"""
RAG Workflow（LangGraph 工作流编排）
=====================================
使用 LangGraph 构建完整的 RAG Agent 有向图工作流。

系统整体工作流程：
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     RAG 对话系统工作流（LangGraph）                    │
  │                                                                     │
  │  START                                                              │
  │    │                                                                │
  │    ▼                                                                │
  │  [memory_retrieval]      读取历史记忆上下文                           │
  │    │                                                                │
  │    ▼                                                                │
  │  [query_understanding]   查询改写/分类/分解                           │
  │    │                                                                │
  │    ▼                                                                │
  │  [task_planning]         工具选择/RAG策略制定                         │
  │    │                                                                │
  │    ▼                                                                │
  │  [retrieval_strategy]    确定检索来源                                 │
  │    │                                                                │
  │    ▼                                                                │
  │  [adaptive_retrieval]    动态调整检索策略                             │
  │    │                                                                │
  │    ▼                                                                │
  │  [multi_retrieval]       并行多路检索（向量/图谱/Web）                  │
  │    │                                                                │
  │    ▼                                                                │
  │  [rerank]                Cross Encoder / ColBERT 重排序              │
  │    │                                                                │
  │    ▼                                                                │
  │  [context_engineering]   上下文过滤/压缩/拼接                         │
  │    │                                                                │
  │    ▼                                                                │
  │  [reasoning]             CoT 推理/答案生成                           │
  │    │                                                                │
  │    ▼                                                                │
  │  [reflection]            答案验证/幻觉检测                            │
  │    │                                                                │
  │    ▼  (条件路由)                                                     │
  │    ├── reflection_result=="passed" ──→ [memory_update] → [evaluate] │
  │    └── reflection_result=="failed" ──→ [reasoning] (重试)           │
  │                                                                     │
  │  [memory_update]         更新对话记忆                                 │
  │    │                                                                │
  │    ▼                                                                │
  │  [evaluate]              RAGAS 评估                                 │
  │    │                                                                │
  │    ▼                                                                │
  │  [feedback]              DSPy 反馈优化                               │
  │    │                                                                │
  │    ▼                                                                │
  │  END                                                                │
  └─────────────────────────────────────────────────────────────────────┘

LangGraph 核心概念：
  - StateGraph  : 有状态图，每个节点共享 AgentState
  - Node        : 图节点，对应一个 Agent 的 run() 方法
  - Edge        : 有向边，定义节点间的执行顺序
  - Conditional Edge : 条件边，根据 State 动态选择下一节点（用于反思重试）
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Literal, Optional

from loguru import logger

from rag_system.agents.base import AgentState
from rag_system.agents.query_understanding_agent import QueryUnderstandingAgent
from rag_system.agents.task_planning_agent import TaskPlanningAgent
from rag_system.agents.retrieval_strategy_agent import RetrievalStrategyAgent
from rag_system.agents.adaptive_retrieval_controller import AdaptiveRetrievalController
from rag_system.agents.multi_retriever_agents import MultiRetrieverAgents
from rag_system.agents.rerank_agent import RerankAgent
from rag_system.agents.context_engineering_agent import ContextEngineeringAgent
from rag_system.agents.reasoning_agent import ReasoningAgent
from rag_system.agents.reflection_agent import ReflectionAgent
from rag_system.memory.memory_retrieval import MemoryRetrieval
from rag_system.memory.memory_update import MemoryUpdate
from rag_system.evaluation.evaluator import RAGEvaluator
from rag_system.evaluation.feedback import FeedbackOptimizer
from rag_system.config.settings import get_settings


class RAGWorkflow:
    """
    RAG 对话系统主工作流
    ====================
    使用 LangGraph StateGraph 编排所有 Agent 节点，
    构建完整的端到端 RAG 对话处理流水线。

    关键设计决策：
    1. **状态传递**    : 所有 Agent 共享 AgentState TypedDict，
                        通过字段读写实现解耦通信
    2. **条件路由**    : reflection_result 字段驱动重试逻辑，
                        避免了复杂的控制流代码
    3. **懒加载模型**  : 重排和 LLM 模型在首次调用时才加载，
                        加快系统启动速度
    4. **错误隔离**    : 每个 Agent 捕获自己的异常，
                        单个节点失败不导致整个流程崩溃

    使用示例：
    ```python
    from rag_system import RAGWorkflow

    workflow = RAGWorkflow()
    result = workflow.run("什么是 RAG？", session_id="user_001")
    print(result["answer"])
    ```
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._logger = logger.bind(module="RAGWorkflow")

        # 初始化所有 Agent 和模块
        self._memory_retrieval = MemoryRetrieval()
        self._query_understanding = QueryUnderstandingAgent()
        self._task_planning = TaskPlanningAgent()
        self._retrieval_strategy = RetrievalStrategyAgent()
        self._adaptive_retrieval = AdaptiveRetrievalController()
        self._multi_retriever = MultiRetrieverAgents()
        self._rerank = RerankAgent()
        self._context_engineering = ContextEngineeringAgent()
        self._reasoning = ReasoningAgent()
        self._reflection = ReflectionAgent()
        self._memory_update = MemoryUpdate(memory_retrieval=self._memory_retrieval)
        self._evaluator = RAGEvaluator()
        self._feedback_optimizer = FeedbackOptimizer()

        # 构建 LangGraph 工作流
        self._graph = self._build_graph()
        self._logger.info("RAG 工作流初始化完成")

    def _build_graph(self):
        """
        构建 LangGraph StateGraph。

        节点注册：将每个 Agent 的 run() 方法注册为图节点。
        边定义：连接节点并设置条件路由。
        """
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(AgentState)

            # ── 注册节点 ──────────────────────────────────────────────────
            graph.add_node("memory_retrieval",    self._memory_retrieval.run)
            graph.add_node("query_understanding", self._query_understanding.run)
            graph.add_node("task_planning",       self._task_planning.run)
            graph.add_node("retrieval_strategy",  self._retrieval_strategy.run)
            graph.add_node("adaptive_retrieval",  self._adaptive_retrieval.run)
            graph.add_node("multi_retrieval",     self._multi_retriever.run)
            graph.add_node("rerank",              self._rerank.run)
            graph.add_node("context_engineering", self._context_engineering.run)
            graph.add_node("reasoning",           self._reasoning.run)
            graph.add_node("reflection",          self._reflection.run)
            graph.add_node("memory_update",       self._memory_update.run)
            graph.add_node("evaluate",            self._evaluator.run)
            graph.add_node("feedback",            self._feedback_optimizer.run)

            # ── 设置入口节点 ───────────────────────────────────────────────
            graph.set_entry_point("memory_retrieval")

            # ── 定义线性执行边 ─────────────────────────────────────────────
            graph.add_edge("memory_retrieval",    "query_understanding")
            graph.add_edge("query_understanding", "task_planning")
            graph.add_edge("task_planning",       "retrieval_strategy")
            graph.add_edge("retrieval_strategy",  "adaptive_retrieval")
            graph.add_edge("adaptive_retrieval",  "multi_retrieval")
            graph.add_edge("multi_retrieval",     "rerank")
            graph.add_edge("rerank",              "context_engineering")
            graph.add_edge("context_engineering", "reasoning")
            graph.add_edge("reasoning",           "reflection")

            # ── 条件路由：反思结果驱动重试逻辑 ─────────────────────────────
            # reflection_result == "passed"  → memory_update → evaluate → feedback → END
            # reflection_result == "failed"  → reasoning（重新生成，增加 retry_count）
            graph.add_conditional_edges(
                "reflection",
                self._reflection_router,
                {
                    "retry":    "reasoning",
                    "continue": "memory_update",
                },
            )

            graph.add_edge("memory_update", "evaluate")
            graph.add_edge("evaluate",      "feedback")
            graph.add_edge("feedback",      END)

            return graph.compile()

        except ImportError:
            self._logger.warning("LangGraph 未安装，使用顺序执行模式")
            return None

    def _reflection_router(
        self, state: AgentState
    ) -> Literal["retry", "continue"]:
        """
        条件路由函数：根据反思结果决定下一步。

        路由逻辑：
        - reflection_result == "passed"   → continue（进入记忆更新和评估）
        - reflection_result == "failed"   → retry（重新推理，限制最大重试次数）
        - reflection_result == "uncertain" → continue（不确定时放行，避免无限循环）
        - retry_count >= max_retries      → 强制 continue（超过重试限制）
        """
        result = state.get("reflection_result", "passed")
        retry_count = state.get("retry_count", 0)

        if result == "failed" and retry_count < self._cfg.max_retries:
            self._logger.info(
                f"反思未通过，触发重试（第 {retry_count + 1}/{self._cfg.max_retries} 次）"
            )
            # 增加重试计数
            state["retry_count"] = retry_count + 1
            return "retry"
        else:
            return "continue"

    def run(
        self,
        user_query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的 RAG 对话流程。

        Args:
            user_query : 用户输入的问题
            session_id : 会话 ID（用于记忆隔离，默认自动生成）

        Returns:
            包含 answer、evaluation_scores 等字段的结果字典
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # 初始化状态
        initial_state: AgentState = {
            "user_query": user_query,
            "session_id": session_id,
            "retry_count": 0,
            "retrieved_docs": [],
            "metadata": {"workflow_version": "1.0.0"},
        }

        self._logger.info(f"开始处理查询: '{user_query[:80]}' (session={session_id})")

        if self._graph is not None:
            # LangGraph 图执行
            final_state = self._graph.invoke(initial_state)
        else:
            # 降级：顺序执行（LangGraph 不可用时）
            final_state = self._sequential_run(initial_state)

        self._logger.info(
            f"查询处理完成: answer_len={len(final_state.get('answer', ''))}, "
            f"eval_scores={final_state.get('evaluation_scores', {})}"
        )

        return final_state

    def _sequential_run(self, state: AgentState) -> AgentState:
        """
        顺序执行模式（LangGraph 不可用时的降级方案）。
        按顺序依次调用所有 Agent，不支持条件路由和并行执行。
        """
        self._logger.warning("使用顺序执行模式（LangGraph 不可用）")
        pipeline = [
            self._memory_retrieval.run,
            self._query_understanding.run,
            self._task_planning.run,
            self._retrieval_strategy.run,
            self._adaptive_retrieval.run,
            self._multi_retriever.run,
            self._rerank.run,
            self._context_engineering.run,
            self._reasoning.run,
            self._reflection.run,
            self._memory_update.run,
            self._evaluator.run,
            self._feedback_optimizer.run,
        ]
        for step in pipeline:
            try:
                state = step(state)
            except Exception as e:
                self._logger.error(f"步骤 {step.__self__.__class__.__name__} 失败: {e}")
                state["error"] = str(e)
        return state

    def get_session_history(self, session_id: str) -> list:
        """获取指定会话的对话历史"""
        return self._memory_retrieval.get_session_history(session_id)

    def clear_session(self, session_id: str) -> None:
        """清除指定会话的记忆"""
        self._memory_retrieval.clear_session(session_id)
        self._logger.info(f"会话 {session_id} 记忆已清除")
