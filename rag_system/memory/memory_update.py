"""
Memory Update（记忆更新模块）
==============================
将当前轮对话结果写入长期记忆系统，实现知识积累。

工作流位置：ReflectionAgent (passed) → [MemoryUpdate] → Evaluation

调用关系：
  - 输入 state 字段  : session_id, user_query, answer, context, evaluation_scores
  - 输出 state 字段  : （无直接输出，副作用写入记忆）

写入策略：
  - 无论答案质量，均写入对话历史（用于上下文连贯性）
  - 仅当评估分数 > 阈值时，将本轮知识写入长期知识库
  - 支持向量化存储（预留 LlamaIndex 接口）
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from rag_system.agents.base import AgentState
from rag_system.config.settings import get_settings


class MemoryUpdate:
    """
    记忆更新模块
    ============
    在每轮对话结束后将新知识和对话内容持久化到记忆系统。

    更新流程：
    1. 写入会话历史（短期记忆）
    2. 若答案质量达标，写入长期知识库（向量索引）
    3. 触发异步持久化（生产环境用，避免阻塞响应）

    [预留接口]：
    - _write_to_long_term_memory() 可接入 LlamaIndex 向量索引写入逻辑
    - _persist_to_disk() 可接入 Redis / PostgreSQL 等持久化存储
    """

    def __init__(self, memory_retrieval=None) -> None:
        """
        Args:
            memory_retrieval: MemoryRetrieval 实例，用于访问会话历史存储
        """
        self._cfg = get_settings()
        self._logger = logger.bind(module="MemoryUpdate")
        self._memory_retrieval = memory_retrieval  # 共享同一个 MemoryRetrieval 实例
        # 长期记忆向量索引（预留，实际使用 LlamaIndex 实现）
        self._long_term_index = None

    def run(self, state: AgentState) -> AgentState:
        """
        记忆更新入口（LangGraph 节点函数）。
        将本轮对话结果写入记忆系统。
        """
        session_id: str = state.get("session_id", "default")
        user_query: str = state.get("user_query", "")
        answer: str = state.get("answer", "")
        eval_scores: dict = state.get("evaluation_scores", {})

        if not answer:
            self._logger.warning("答案为空，跳过记忆更新")
            return state

        # Step 1: 写入短期会话历史
        if self._memory_retrieval:
            self._memory_retrieval.add_turn(session_id, user_query, answer)
            self._logger.debug(f"会话历史已更新: session={session_id}")

        # Step 2: 质量达标时写入长期记忆
        quality_score = self._compute_quality_score(eval_scores)
        if quality_score >= self._cfg.evaluation.feedback_threshold:
            self._write_to_long_term_memory(session_id, user_query, answer)

        self._logger.info(
            f"记忆更新完成: session={session_id}, "
            f"quality_score={quality_score:.3f}"
        )

        return state

    def _compute_quality_score(self, eval_scores: dict) -> float:
        """计算综合质量分数（多个 RAGAS 指标的加权平均）"""
        if not eval_scores:
            return 0.5  # 无评估分数时使用默认值

        weights = {
            "faithfulness": 0.4,
            "answer_relevancy": 0.4,
            "context_precision": 0.2,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for metric, weight in weights.items():
            if metric in eval_scores:
                weighted_sum += eval_scores[metric] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _write_to_long_term_memory(
        self, session_id: str, query: str, answer: str
    ) -> None:
        """
        将高质量的问答对写入长期记忆向量索引。
        [预留接口] 生产环境接入 LlamaIndex VectorStoreIndex 实现：

        ```python
        from llama_index.core import Document
        doc = Document(
            text=f"Q: {query}\nA: {answer}",
            metadata={"session_id": session_id, "type": "qa_pair"},
        )
        self._long_term_index.insert(doc)
        ```
        """
        self._logger.debug(
            f"[预留] 长期记忆写入: query='{query[:50]}', answer_len={len(answer)}"
        )
