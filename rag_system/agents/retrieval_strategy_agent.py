"""
Retrieval Strategy Agent（检索策略 Agent）
==========================================
负责根据任务规划结果确定具体的检索来源组合：
  - Vector Retrieval : 向量数据库语义检索
  - Graph Retrieval  : 知识图谱检索
  - Web Retrieval    : 实时 Web 搜索

工作流位置：TaskPlanningAgent → [RetrievalStrategyAgent] → AdaptiveRetrievalController

调用关系：
  - 输入 state 字段  : selected_tools, rag_strategy, query_type, rewritten_query
  - 输出 state 字段  : retrieval_sources（已确定的检索来源列表）
"""

from __future__ import annotations

from typing import List

from rag_system.agents.base import AgentState, BaseAgent
from rag_system.config.settings import get_settings


# 工具到检索来源的映射关系
TOOL_TO_SOURCE_MAP = {
    "vector_search": "vector",
    "graph_search": "graph",
    "web_search": "web",
}

# RAG 策略到默认检索来源的映射（在工具映射缺失时使用）
STRATEGY_SOURCE_MAP = {
    "simple": ["vector"],
    "hybrid": ["vector", "web"],
    "multi_hop": ["vector", "graph"],
    "graph": ["graph", "vector"],
}


class RetrievalStrategyAgent(BaseAgent):
    """
    检索策略 Agent
    ==============
    将任务规划层的抽象工具选择转化为具体的检索来源配置。

    核心逻辑：
    1. 将 selected_tools 映射为对应的检索来源（vector/graph/web）
    2. 根据系统配置过滤不可用的来源（如 GraphRAG 未启用则移除 graph）
    3. 保证至少有一个检索来源可用（最低保障为 vector）

    与 AdaptiveRetrievalController 的分工：
    - RetrievalStrategyAgent   : 静态策略决策（基于查询类型和工具选择）
    - AdaptiveRetrievalController : 动态运行时调整（基于实际检索效果反馈）
    """

    def __init__(self, settings=None) -> None:
        super().__init__("RetrievalStrategyAgent", settings)
        self._cfg = get_settings()

    def _execute(self, state: AgentState) -> AgentState:
        selected_tools: List[str] = state.get("selected_tools", ["vector_search"])
        rag_strategy: str = state.get("rag_strategy", "simple")

        # Step 1: 将工具选择映射为检索来源
        sources: List[str] = []
        for tool in selected_tools:
            source = TOOL_TO_SOURCE_MAP.get(tool)
            if source and source not in sources:
                sources.append(source)

        # Step 2: 若无法从工具映射，使用策略默认来源
        if not sources:
            sources = STRATEGY_SOURCE_MAP.get(rag_strategy, ["vector"])

        # Step 3: 按配置过滤不可用来源
        sources = self._filter_available_sources(sources)

        # Step 4: 保障最低可用性
        if not sources:
            self._logger.warning("所有检索来源均不可用，降级为向量检索")
            sources = ["vector"]

        self._logger.info(f"确定检索来源: {sources}（策略: {rag_strategy}）")

        return {**state, "retrieval_sources": sources}

    def _filter_available_sources(self, sources: List[str]) -> List[str]:
        """根据系统配置过滤不可用的检索来源"""
        available = []
        for source in sources:
            if source == "vector":
                available.append(source)  # 向量检索始终可用
            elif source == "graph" and self._cfg.graph_rag.enabled:
                available.append(source)
            elif source == "web" and self._cfg.web_search.enabled:
                available.append(source)
            else:
                self._logger.debug(f"检索来源 '{source}' 未启用，已跳过")
        return available
