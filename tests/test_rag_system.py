"""
RAG 系统单元测试
================
测试各核心模块的功能，无需真实 LLM API 调用（使用 Mock）。
"""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# 确保测试时不需要真实 API Key
os.environ.setdefault("LLM_API_KEY", "test-key-placeholder")
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")


# ============================================================
# 配置模块测试
# ============================================================

class TestSettings:
    """测试全局配置管理"""

    def test_default_settings(self):
        """测试默认配置加载"""
        from rag_system.config.settings import Settings
        settings = Settings()
        assert settings.llm.model_name == "gpt-4o-mini"
        assert settings.llm.temperature == 0.0
        assert settings.vector_store.top_k == 10
        assert settings.rerank.method == "cross_encoder"
        assert settings.memory.max_history == 20

    def test_get_settings_singleton(self):
        """测试配置单例模式"""
        from rag_system.config.settings import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_env_override(self):
        """测试环境变量覆盖配置"""
        from rag_system.config.settings import Settings
        with patch.dict(os.environ, {"LLM_MODEL_NAME": "gpt-4o"}):
            settings = Settings()
            assert settings.llm.model_name == "gpt-4o"


# ============================================================
# Agent 状态与基类测试
# ============================================================

class TestAgentBase:
    """测试 Agent 基类和状态模型"""

    def test_agent_state_fields(self):
        """测试 AgentState TypedDict 字段定义"""
        from rag_system.agents.base import AgentState
        state: AgentState = {
            "user_query": "测试查询",
            "session_id": "test-session",
            "retry_count": 0,
            "retrieved_docs": [],
            "metadata": {},
        }
        assert state["user_query"] == "测试查询"
        assert state["session_id"] == "test-session"

    def test_retrieved_document_model(self):
        """测试 RetrievedDocument Pydantic 模型"""
        from rag_system.agents.base import RetrievedDocument
        doc = RetrievedDocument(
            doc_id="test-001",
            content="这是测试内容",
            source="vector",
            score=0.85,
        )
        assert doc.doc_id == "test-001"
        assert doc.source == "vector"
        assert doc.score == 0.85
        assert doc.metadata == {}

    def test_base_agent_error_handling(self):
        """测试 Agent 基类的错误处理"""
        from rag_system.agents.base import BaseAgent, AgentState

        class FailingAgent(BaseAgent):
            def _execute(self, state: AgentState) -> AgentState:
                raise ValueError("测试错误")

        agent = FailingAgent("FailingAgent")
        state: AgentState = {"user_query": "test", "retrieved_docs": [], "metadata": {}}
        result = agent.run(state)
        assert "error" in result
        assert "测试错误" in result["error"]


# ============================================================
# 检索策略 Agent 测试
# ============================================================

class TestRetrievalStrategyAgent:
    """测试检索策略 Agent（无需 LLM）"""

    def test_tool_to_source_mapping(self):
        """测试工具到检索来源的映射"""
        from rag_system.agents.retrieval_strategy_agent import RetrievalStrategyAgent
        from rag_system.agents.base import AgentState

        agent = RetrievalStrategyAgent()
        state: AgentState = {
            "user_query": "test",
            "selected_tools": ["vector_search", "web_search"],
            "rag_strategy": "hybrid",
            "retrieved_docs": [],
            "metadata": {},
        }
        result = agent._execute(state)
        # vector_search → vector, web_search → web（若 web_search 启用）
        assert "vector" in result["retrieval_sources"]

    def test_fallback_to_vector_when_empty(self):
        """测试无可用来源时降级为向量检索"""
        from rag_system.agents.retrieval_strategy_agent import RetrievalStrategyAgent
        from rag_system.agents.base import AgentState

        agent = RetrievalStrategyAgent()
        state: AgentState = {
            "user_query": "test",
            "selected_tools": [],
            "rag_strategy": "simple",
            "retrieved_docs": [],
            "metadata": {},
        }
        result = agent._execute(state)
        assert "vector" in result["retrieval_sources"]


# ============================================================
# 自适应检索控制器测试
# ============================================================

class TestAdaptiveRetrievalController:
    """测试自适应检索控制器"""

    def test_complexity_assessment_factual(self):
        """测试简单事实性查询的复杂度评分"""
        from rag_system.agents.adaptive_retrieval_controller import AdaptiveRetrievalController

        controller = AdaptiveRetrievalController()
        score = controller._assess_complexity("什么是 RAG？", "factual", [])
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # 简单问题复杂度应较低

    def test_complexity_assessment_multi_hop(self):
        """测试多跳查询的复杂度评分"""
        from rag_system.agents.adaptive_retrieval_controller import AdaptiveRetrievalController

        controller = AdaptiveRetrievalController()
        sub_queries = ["子问题1", "子问题2", "子问题3"]
        score = controller._assess_complexity(
            "这是一个非常复杂的多跳查询问题需要多步推理", "multi_hop", sub_queries
        )
        assert score > 0.5  # 多跳问题复杂度应较高

    def test_performance_update(self):
        """测试性能记录更新（指数移动平均）"""
        from rag_system.agents.adaptive_retrieval_controller import AdaptiveRetrievalController

        controller = AdaptiveRetrievalController()
        old_score = controller._source_performance["vector"]
        controller.update_performance("vector", 0.9)
        new_score = controller._source_performance["vector"]
        # 分数应该向 0.9 方向移动
        assert new_score != old_score

    def test_adaptive_select_expands_for_high_complexity(self):
        """测试高复杂度查询时自动扩展检索来源"""
        from rag_system.agents.adaptive_retrieval_controller import AdaptiveRetrievalController
        from rag_system.config.settings import get_settings

        # 启用 web 搜索
        controller = AdaptiveRetrievalController()
        controller._cfg.web_search.enabled = True
        expanded = controller._adaptive_select(["vector"], complexity=0.8)
        assert "web" in expanded


# ============================================================
# 多检索器 Agent 测试
# ============================================================

class TestMultiRetrieverAgents:
    """测试多检索器并行执行（使用 Mock 检索器）"""

    def test_deduplicate(self):
        """测试文档去重功能"""
        from rag_system.agents.multi_retriever_agents import MultiRetrieverAgents
        from rag_system.agents.base import RetrievedDocument

        agent = MultiRetrieverAgents.__new__(MultiRetrieverAgents)
        agent.name = "test"
        from loguru import logger
        agent._logger = logger.bind(agent="test")

        docs = [
            RetrievedDocument(doc_id="1", content="相同内容", source="vector", score=0.9),
            RetrievedDocument(doc_id="2", content="相同内容", source="graph", score=0.8),
            RetrievedDocument(doc_id="3", content="不同内容", source="web", score=0.7),
        ]
        unique = agent._deduplicate(docs)
        assert len(unique) == 2  # 去除一个重复文档


# ============================================================
# 上下文工程 Agent 测试
# ============================================================

class TestContextEngineeringAgent:
    """测试上下文工程 Agent"""

    def test_filter_low_score_docs(self):
        """测试低分文档过滤"""
        from rag_system.agents.context_engineering_agent import ContextEngineeringAgent
        from rag_system.agents.base import RetrievedDocument

        agent = ContextEngineeringAgent()
        docs = [
            RetrievedDocument(doc_id="1", content="高质量内容", source="vector", score=0.9),
            RetrievedDocument(doc_id="2", content="低质量内容", source="web", score=0.05),
        ]
        filtered = agent._filter_docs(docs)
        assert len(filtered) == 1
        assert filtered[0].doc_id == "1"

    def test_build_context_format(self):
        """测试上下文格式化输出"""
        from rag_system.agents.context_engineering_agent import ContextEngineeringAgent
        from rag_system.agents.base import RetrievedDocument

        agent = ContextEngineeringAgent()
        docs = [
            RetrievedDocument(doc_id="1", content="文档内容", source="vector", score=0.85),
        ]
        context = agent._build_context(docs)
        assert "[文档 1]" in context
        assert "知识库" in context
        assert "文档内容" in context

    def test_context_truncation(self):
        """测试超长上下文截断"""
        from rag_system.agents.context_engineering_agent import (
            ContextEngineeringAgent, MAX_CONTEXT_CHARS
        )
        from rag_system.agents.base import RetrievedDocument, AgentState

        agent = ContextEngineeringAgent()
        # 创建超长文档
        long_docs = [
            RetrievedDocument(
                doc_id=str(i),
                content="A" * 2000,
                source="vector",
                score=0.8,
            )
            for i in range(20)
        ]
        state: AgentState = {
            "user_query": "test",
            "reranked_docs": long_docs,
            "retrieved_docs": [],
            "metadata": {},
        }
        result = agent._execute(state)
        assert len(result["context"]) <= MAX_CONTEXT_CHARS + 20  # 允许截断标记的额外字符


# ============================================================
# 记忆模块测试
# ============================================================

class TestMemoryRetrieval:
    """测试记忆检索模块"""

    def test_empty_session(self):
        """测试空会话的记忆检索"""
        from rag_system.memory.memory_retrieval import MemoryRetrieval
        from rag_system.agents.base import AgentState

        memory = MemoryRetrieval.__new__(MemoryRetrieval)
        memory._sessions = {}
        memory._summaries = {}
        from loguru import logger
        memory._logger = logger.bind(module="test")
        from collections import defaultdict
        memory._sessions = defaultdict(list)
        from rag_system.config.settings import get_settings
        memory._cfg = get_settings()

        context = memory._build_memory_context("new-session", "test query")
        assert context == "无历史记录"

    def test_add_and_retrieve_history(self):
        """测试添加历史并检索"""
        from rag_system.memory.memory_retrieval import MemoryRetrieval

        memory = MemoryRetrieval.__new__(MemoryRetrieval)
        from collections import defaultdict
        from loguru import logger
        memory._sessions = defaultdict(list)
        memory._summaries = {}
        memory._logger = logger.bind(module="test")
        from rag_system.config.settings import get_settings
        memory._cfg = get_settings()

        memory.add_turn("session-1", "用户问题", "助手回答")
        history = memory.get_session_history("session-1")
        assert len(history) == 2
        assert history[0] == ("human", "用户问题")
        assert history[1] == ("assistant", "助手回答")

    def test_clear_session(self):
        """测试清除会话记忆"""
        from rag_system.memory.memory_retrieval import MemoryRetrieval

        memory = MemoryRetrieval.__new__(MemoryRetrieval)
        from collections import defaultdict
        from loguru import logger
        memory._sessions = defaultdict(list)
        memory._summaries = {}
        memory._logger = logger.bind(module="test")

        memory.add_turn("session-1", "问题", "回答")
        memory.clear_session("session-1")
        history = memory.get_session_history("session-1")
        assert len(history) == 0


# ============================================================
# Web 检索器测试
# ============================================================

class TestWebRetriever:
    """测试 Web 检索器"""

    def test_mock_retrieve(self):
        """测试 Mock 检索返回正确格式"""
        from rag_system.retrieval.web_retriever import WebRetriever
        from rag_system.config.settings import get_settings

        cfg = get_settings()
        retriever = WebRetriever(settings=cfg)
        docs = retriever._mock_retrieve("测试查询")
        assert len(docs) > 0
        assert docs[0].source == "web"
        assert "测试查询" in docs[0].content

    def test_disabled_retriever_returns_empty(self):
        """测试禁用 Web 检索时返回空列表"""
        from rag_system.retrieval.web_retriever import WebRetriever
        from rag_system.config.settings import get_settings

        cfg = get_settings()
        cfg.web_search.enabled = False
        retriever = WebRetriever(settings=cfg)
        docs = retriever.retrieve("测试查询")
        assert docs == []


# ============================================================
# 评估模块测试
# ============================================================

class TestRAGEvaluator:
    """测试 RAGAS 评估器"""

    def test_heuristic_evaluate(self):
        """测试启发式评估（无需 RAGAS 库）"""
        from rag_system.evaluation.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        scores = evaluator._heuristic_evaluate(
            question="什么是 RAG？",
            answer="RAG 是检索增强生成技术",
            context="RAG（Retrieval Augmented Generation）是一种将检索与生成结合的技术",
        )
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_evaluator_skips_when_disabled(self):
        """测试禁用评估时跳过评估"""
        from rag_system.evaluation.evaluator import RAGEvaluator
        from rag_system.agents.base import AgentState
        from rag_system.config.settings import get_settings

        evaluator = RAGEvaluator()
        evaluator._cfg.evaluation.enabled = False
        state: AgentState = {
            "user_query": "test",
            "answer": "test answer",
            "context": "test context",
            "retrieved_docs": [],
            "metadata": {},
        }
        result = evaluator.run(state)
        assert result["evaluation_scores"] == {}


# ============================================================
# 反思 Agent 测试
# ============================================================

class TestReflectionAgent:
    """测试反思 Agent 的 JSON 解析"""

    def test_parse_valid_response(self):
        """测试解析有效的 JSON 反思响应"""
        from rag_system.agents.reflection_agent import ReflectionAgent

        agent = ReflectionAgent.__new__(ReflectionAgent)
        agent.name = "test"
        from loguru import logger
        agent._logger = logger.bind(agent="test")

        valid_json = '{"result": "passed", "groundedness_score": 0.9, "feedback": ""}'
        result = agent._parse_response(valid_json)
        assert result["result"] == "passed"
        assert result["groundedness_score"] == 0.9

    def test_parse_invalid_response_defaults_to_passed(self):
        """测试解析失败时默认通过（避免无限循环）"""
        from rag_system.agents.reflection_agent import ReflectionAgent

        agent = ReflectionAgent.__new__(ReflectionAgent)
        agent.name = "test"
        from loguru import logger
        agent._logger = logger.bind(agent="test")

        result = agent._parse_response("这不是 JSON 格式")
        assert result["result"] == "passed"


# ============================================================
# 反馈优化器测试
# ============================================================

class TestFeedbackOptimizer:
    """测试反馈优化器"""

    def test_optimization_stats_empty(self):
        """测试空历史时的统计信息"""
        from rag_system.evaluation.feedback import FeedbackOptimizer

        optimizer = FeedbackOptimizer()
        stats = optimizer.get_optimization_stats()
        assert stats["total_feedback"] == 0
        assert stats["avg_scores"] == {}

    def test_analyze_high_scores(self):
        """测试高分时不触发优化"""
        from rag_system.evaluation.feedback import FeedbackOptimizer
        from rag_system.agents.base import AgentState

        optimizer = FeedbackOptimizer()
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.85}
        state: AgentState = {
            "user_query": "test",
            "retrieved_docs": [],
            "metadata": {},
        }
        applied = optimizer._analyze_and_optimize(scores, state)
        assert applied is False  # 高分不触发优化
