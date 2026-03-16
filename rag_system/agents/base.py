"""
Agent 基类与状态定义
====================
定义所有 Agent 共享的状态模型（AgentState）和抽象基类（BaseAgent）。

AgentState 是 LangGraph 工作流的核心数据载体，贯穿整个 Agent 调用链，
每个 Agent 读取并更新 State 中的特定字段，实现解耦通信。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================
# 检索文档模型
# ============================================================

class RetrievedDocument(BaseModel):
    """检索到的单个文档片段"""

    doc_id: str = Field(description="文档唯一标识符")
    content: str = Field(description="文档文本内容")
    source: str = Field(description="来源：vector / graph / web")
    score: float = Field(default=0.0, description="相关性得分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


# ============================================================
# LangGraph 核心状态（TypedDict）
# ============================================================

class AgentState(TypedDict, total=False):
    """
    RAG 系统全局状态
    ================
    贯穿 LangGraph 工作流所有节点的共享状态对象。
    每个 Agent 节点读取所需字段并写回更新后的字段。

    字段说明：
    ----------
    user_query          : 原始用户查询
    session_id          : 会话 ID，用于记忆隔离
    rewritten_query     : 经过改写的查询
    query_type          : 查询分类（factual / analytical / conversational / multi_hop）
    sub_queries         : 分解后的子查询列表（用于复杂问题）
    selected_tools      : 任务规划选择的工具列表
    rag_strategy        : 选定的 RAG 策略名称
    retrieval_sources   : 启用的检索来源列表（vector / graph / web）
    retrieved_docs      : 所有检索到的原始文档列表（使用 operator.add 合并）
    reranked_docs       : 重排序后的文档列表
    context             : 上下文工程处理后的最终上下文字符串
    reasoning_steps     : 推理过程的 Chain-of-Thought 步骤
    answer              : 最终生成的答案
    reflection_result   : 反思验证结果（passed / failed）
    reflection_feedback : 反思模块的改进建议
    memory_context      : 从历史记忆检索到的上下文
    evaluation_scores   : RAGAS / DSPy 评估分数字典
    feedback_applied    : 是否已应用评估反馈
    error               : 运行时错误信息（若有）
    retry_count         : 当前重试次数
    metadata            : 系统元数据（耗时、模型版本等）
    """

    # --- 输入 ---
    user_query: str
    session_id: str

    # --- 查询理解 ---
    rewritten_query: str
    query_type: str  # factual / analytical / conversational / multi_hop
    sub_queries: List[str]

    # --- 任务规划 ---
    selected_tools: List[str]
    rag_strategy: str  # simple / hybrid / multi_hop / graph

    # --- 检索策略 ---
    retrieval_sources: List[str]  # ["vector", "graph", "web"]

    # --- 检索结果
    # 使用 Annotated + operator.add 实现 LangGraph 并行节点自动合并：
    # 多个并行检索节点各自向 retrieved_docs 追加结果，LangGraph 自动通过 operator.add
    # 将各节点返回的列表连接为一个完整列表，无需手动合并。
    retrieved_docs: Annotated[List[RetrievedDocument], operator.add]

    # --- 重排与上下文 ---
    reranked_docs: List[RetrievedDocument]
    context: str

    # --- 推理 ---
    reasoning_steps: List[str]
    answer: str

    # --- 反思 ---
    reflection_result: str  # passed / failed / uncertain
    reflection_feedback: str

    # --- 记忆 ---
    memory_context: str

    # --- 评估 ---
    evaluation_scores: Dict[str, float]
    feedback_applied: bool

    # --- 系统字段 ---
    error: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]


# ============================================================
# Agent 抽象基类
# ============================================================

class BaseAgent(ABC):
    """
    所有 Agent 的抽象基类
    =====================
    定义统一的 Agent 接口：
    - run(state) : 同步执行入口，供 LangGraph 节点调用
    - _execute(state) : 子类实现具体业务逻辑
    - _handle_error(state, error) : 统一错误处理

    LangGraph 节点函数签名约定：
        def node_func(state: AgentState) -> AgentState
    每个 Agent 的 run 方法满足此约定，可直接注册为图节点。
    """

    def __init__(self, name: str, settings: Optional[Any] = None) -> None:
        self.name = name
        self.settings = settings
        self._logger = logger.bind(agent=name)

    def run(self, state: AgentState) -> AgentState:
        """
        Agent 执行入口（LangGraph 节点函数）。
        计时、日志、错误处理均在此封装，子类只需实现 _execute。
        """
        start_time = time.perf_counter()
        self._logger.info(f"[{self.name}] 开始执行，query='{state.get('user_query', '')[:80]}'")

        try:
            updated_state = self._execute(state)
            elapsed = time.perf_counter() - start_time
            self._logger.info(f"[{self.name}] 执行完成，耗时 {elapsed:.3f}s")
            # 写入执行耗时到 metadata
            meta = updated_state.get("metadata", {})
            meta[f"{self.name}_elapsed_s"] = round(elapsed, 3)
            updated_state["metadata"] = meta
            return updated_state
        except Exception as exc:
            self._logger.exception(f"[{self.name}] 执行失败: {exc}")
            return self._handle_error(state, exc)

    @abstractmethod
    def _execute(self, state: AgentState) -> AgentState:
        """子类实现具体业务逻辑"""
        ...

    def _handle_error(self, state: AgentState, error: Exception) -> AgentState:
        """统一错误处理：写入 error 字段并返回当前状态"""
        return {**state, "error": str(error)}
