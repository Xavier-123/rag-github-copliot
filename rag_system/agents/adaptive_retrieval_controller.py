"""
Adaptive Retrieval Controller（自适应检索控制器）
================================================
动态调整检索策略，根据查询复杂度、检索效果和历史表现
自适应选择最优检索方式。

工作流位置：RetrievalStrategyAgent → [AdaptiveRetrievalController] → MultiRetrieverAgents

调用关系：
  - 输入 state 字段  : retrieval_sources, rewritten_query, query_type, rag_strategy
  - 输出 state 字段  : retrieval_sources（动态调整后）, metadata（含调整原因）

核心机制：
  1. 查询复杂度评估（Complexity Assessment）
     - 基于 token 数量、子查询数量、查询类型计算复杂度分数
  2. 历史表现感知（Performance-Aware）
     - 读取历史检索质量记录，优先选择表现好的来源
  3. 动态扩展（Dynamic Expansion）
     - 若查询复杂度高，自动增加检索来源
  4. 置信度阈值（Confidence Threshold）
     - 若预估检索效果低于阈值，触发降级或扩展策略

注意：完整的 RL/Bandit 算法实现较为复杂，此处提供基于规则的简化实现，
预留 _adaptive_select() 接口供后续接入强化学习模型。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rag_system.agents.base import AgentState, BaseAgent
from rag_system.config.settings import get_settings


# ============================================================
# 复杂度评分权重
# ============================================================

COMPLEXITY_WEIGHTS = {
    "token_length_per_10": 0.1,   # 每 10 个 token 增加 0.1 复杂度
    "sub_query_count": 0.2,       # 每个子查询增加 0.2 复杂度
    "type_score": {
        "factual": 0.1,
        "conversational": 0.0,
        "analytical": 0.4,
        "multi_hop": 0.8,
    },
}


class AdaptiveRetrievalController(BaseAgent):
    """
    自适应检索控制器
    ================
    在静态检索策略基础上，通过动态分析进一步优化检索来源选择。

    自适应机制说明：
    ---------------
    1. **复杂度评估**：综合 query 长度、子查询数量、查询类型计算 [0,1] 区间复杂度分数。
    2. **来源扩展**：复杂度 > 0.6 时，自动追加 web 和 graph 来源（若系统支持）。
    3. **历史过滤**（预留接口）：_get_source_performance() 可接入性能追踪数据库，
       过滤历史表现差的来源。
    4. **强化学习接口**（预留）：_adaptive_select() 可替换为 Bandit/RL 模型，
       通过 epsilon-greedy 或 UCB 算法动态探索最优策略。
    """

    def __init__(self, settings=None) -> None:
        super().__init__("AdaptiveRetrievalController", settings)
        self._cfg = get_settings()
        # 历史性能记录（source -> avg_score），实际部署中应持久化存储
        self._source_performance: Dict[str, float] = {
            "vector": 0.75,
            "graph": 0.70,
            "web": 0.65,
        }

    def _execute(self, state: AgentState) -> AgentState:
        sources: List[str] = state.get("retrieval_sources", ["vector"])
        query: str = state.get("rewritten_query", state.get("user_query", ""))
        query_type: str = state.get("query_type", "factual")
        sub_queries: List[str] = state.get("sub_queries", [])

        # 1. 评估查询复杂度
        complexity_score = self._assess_complexity(query, query_type, sub_queries)
        self._logger.debug(f"查询复杂度评分: {complexity_score:.2f}")

        # 2. 自适应调整来源
        adjusted_sources = self._adaptive_select(sources, complexity_score)

        # 3. 记录调整信息到 metadata
        meta = state.get("metadata", {})
        meta["complexity_score"] = round(complexity_score, 3)
        meta["original_sources"] = sources
        meta["adjusted_sources"] = adjusted_sources

        self._logger.info(
            f"自适应调整: {sources} → {adjusted_sources}（复杂度: {complexity_score:.2f}）"
        )

        return {**state, "retrieval_sources": adjusted_sources, "metadata": meta}

    def _assess_complexity(
        self, query: str, query_type: str, sub_queries: List[str]
    ) -> float:
        """
        评估查询复杂度（返回 0.0 ~ 1.0 的分数）

        算法：
        - token 长度得分 : min(len(tokens) / 100, 1.0) * weight
        - 子查询数量得分 : min(sub_count * weight, 0.6)
        - 类型基础得分   : 根据查询类型预设权重
        """
        tokens = query.split()
        token_score = min(len(tokens) / 100, 1.0) * COMPLEXITY_WEIGHTS["token_length_per_10"] * 10
        sub_score = min(len(sub_queries) * COMPLEXITY_WEIGHTS["sub_query_count"], 0.6)
        type_score = COMPLEXITY_WEIGHTS["type_score"].get(query_type, 0.2)
        return min(token_score + sub_score + type_score, 1.0)

    def _adaptive_select(
        self, current_sources: List[str], complexity: float
    ) -> List[str]:
        """
        根据复杂度动态调整检索来源。

        扩展规则：
        - complexity > 0.6 : 高复杂度，追加 web 和 graph（如系统支持）
        - complexity > 0.4 : 中等复杂度，确保 vector 存在
        - complexity ≤ 0.4 : 简单查询，按原来源执行

        [预留接口]：可替换此方法为 Bandit/RL 模型的在线决策逻辑。
        """
        expanded = list(current_sources)

        if complexity > 0.6:
            if self._cfg.web_search.enabled and "web" not in expanded:
                expanded.append("web")
            if self._cfg.graph_rag.enabled and "graph" not in expanded:
                expanded.append("graph")
        elif complexity > 0.4:
            if "vector" not in expanded:
                expanded.insert(0, "vector")

        # 过滤历史表现极差的来源（阈值 0.3）
        filtered = [
            s for s in expanded
            if self._source_performance.get(s, 0.5) >= 0.3
        ]
        return filtered if filtered else ["vector"]

    def update_performance(self, source: str, score: float) -> None:
        """
        更新检索来源的历史性能记录（指数移动平均）。
        [预留接口]：由 Evaluation Agent 在每轮对话后调用。

        Args:
            source: 检索来源名称（vector/graph/web）
            score: 本次检索质量分数（0.0~1.0）
        """
        alpha = 0.1  # 学习率
        old_score = self._source_performance.get(source, 0.5)
        self._source_performance[source] = (1 - alpha) * old_score + alpha * score
        self._logger.debug(
            f"性能更新: {source} {old_score:.3f} → {self._source_performance[source]:.3f}"
        )
