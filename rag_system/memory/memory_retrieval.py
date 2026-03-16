"""
Memory Retrieval（记忆检索模块）
=================================
从长期记忆系统中检索与当前查询相关的历史对话和知识。

工作流位置：[MemoryRetrieval] → QueryUnderstandingAgent（提供历史上下文）

调用关系：
  - 输入 state 字段  : user_query, session_id
  - 输出 state 字段  : memory_context（历史对话摘要/相关记忆）

记忆存储架构：
  - 短期记忆（Short-term）: 当前会话对话历史（存储在内存中）
  - 长期记忆（Long-term）  : 跨会话的重要知识（持久化存储，向量检索）
  - 摘要记忆（Summary）   : 超过 summary_threshold 轮后对历史对话进行摘要压缩

实现说明：
  - 使用字典模拟会话历史存储（生产环境应替换为 Redis / PostgreSQL）
  - 使用 LlamaIndex 向量索引实现语义化记忆检索（当前为简化实现）
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from rag_system.agents.base import AgentState
from rag_system.config.settings import get_settings


# 会话历史条目类型
HistoryEntry = Tuple[str, str]  # (role, content)


class MemoryRetrieval:
    """
    记忆检索模块
    ============
    管理对话历史并为当前查询提供相关历史上下文。

    核心功能：
    1. 会话历史管理  : 按 session_id 存储对话轮次
    2. 历史上下文构建: 格式化最近 N 轮对话供 Agent 使用
    3. 长期记忆检索  : 语义搜索历史对话（预留向量检索接口）
    4. 摘要压缩      : 历史过长时，使用 LLM 生成摘要保持上下文可管理
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._logger = logger.bind(module="MemoryRetrieval")
        # 内存中的会话历史：{session_id: [(role, content), ...]}
        self._sessions: Dict[str, List[HistoryEntry]] = defaultdict(list)
        # 会话摘要：{session_id: summary_text}
        self._summaries: Dict[str, str] = {}
        self._llm = None  # 懒加载

    def _get_llm(self):
        """懒加载 LLM 实例（仅在摘要压缩时使用）"""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self._cfg.llm.model_name,
                temperature=0.0,
                api_key=self._cfg.llm.api_key,
                base_url=self._cfg.llm.api_base,
            )
        return self._llm

    def run(self, state: AgentState) -> AgentState:
        """
        记忆检索入口（LangGraph 节点函数）。
        读取历史对话，构建 memory_context 字段。
        """
        session_id: str = state.get("session_id", "default")
        user_query: str = state.get("user_query", "")

        memory_context = self._build_memory_context(session_id, user_query)
        self._logger.info(f"记忆检索完成: session={session_id}, context_len={len(memory_context)}")

        return {**state, "memory_context": memory_context}

    def _build_memory_context(self, session_id: str, current_query: str) -> str:
        """
        构建记忆上下文字符串。
        优先使用摘要 + 最近 N 轮对话的组合策略，在上下文长度和信息完整性间取得平衡。
        """
        history = self._sessions.get(session_id, [])
        if not history:
            return "无历史记录"

        # 检查是否需要摘要压缩
        if len(history) > self._cfg.memory.summary_threshold * 2:
            self._maybe_summarize(session_id)

        context_parts = []

        # 添加历史摘要（若有）
        summary = self._summaries.get(session_id)
        if summary:
            context_parts.append(f"【历史摘要】\n{summary}")

        # 添加最近 N 轮对话
        recent = history[-self._cfg.memory.max_history * 2:]
        if recent:
            formatted = []
            for role, content in recent:
                prefix = "用户" if role == "human" else "助手"
                formatted.append(f"{prefix}: {content[:300]}")
            context_parts.append("【近期对话】\n" + "\n".join(formatted))

        return "\n\n".join(context_parts) if context_parts else "无历史记录"

    def _maybe_summarize(self, session_id: str) -> None:
        """
        若历史对话超过阈值，使用 LLM 生成摘要并压缩历史。
        摘要后只保留最近 summary_threshold 轮对话。
        """
        history = self._sessions.get(session_id, [])
        if len(history) <= self._cfg.memory.summary_threshold * 2:
            return

        # 选取需要摘要的历史（保留最近 summary_threshold 轮）
        to_summarize = history[: -self._cfg.memory.summary_threshold * 2]
        formatted = "\n".join(
            f"{'用户' if r == 'human' else '助手'}: {c[:200]}"
            for r, c in to_summarize
        )

        try:
            from langchain.schema import HumanMessage, SystemMessage
            response = self._get_llm().invoke(
                [
                    SystemMessage(content="请简洁地总结以下对话历史，保留关键信息。"),
                    HumanMessage(content=formatted),
                ]
            )
            self._summaries[session_id] = response.content
            # 保留最近 N 轮历史
            self._sessions[session_id] = history[-self._cfg.memory.summary_threshold * 2:]
            self._logger.debug(f"会话 {session_id} 历史已压缩摘要")
        except Exception as e:
            self._logger.warning(f"历史摘要失败: {e}")

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """
        添加一轮对话到会话历史。
        由 MemoryUpdate 模块调用。
        """
        self._sessions[session_id].append(("human", user_msg))
        self._sessions[session_id].append(("assistant", assistant_msg))

    def get_session_history(self, session_id: str) -> List[HistoryEntry]:
        """获取指定会话的完整历史"""
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """清除指定会话的记忆"""
        self._sessions.pop(session_id, None)
        self._summaries.pop(session_id, None)
