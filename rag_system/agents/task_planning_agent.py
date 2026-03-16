"""
Task Planning Agent（任务规划 Agent）
=====================================
根据查询理解结果，制定最优的执行策略：
  1. 工具选择（Tool Selection）  : 决定使用哪些工具（向量检索/图检索/Web/计算器等）
  2. RAG 策略选择（RAG Strategy）: 选择检索增强策略（simple/hybrid/multi_hop/graph）

工作流位置：QueryUnderstandingAgent → [TaskPlanningAgent] → RetrievalStrategyAgent

调用关系：
  - 输入 state 字段  : rewritten_query, query_type, sub_queries
  - 输出 state 字段  : selected_tools, rag_strategy
"""

from __future__ import annotations

import json

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_system.agents.base import AgentState, BaseAgent
from rag_system.config.settings import get_settings


# ============================================================
# 可用工具与策略定义
# ============================================================

AVAILABLE_TOOLS = {
    "vector_search": "向量数据库语义检索，适合事实性问题",
    "graph_search": "知识图谱检索，适合实体关系和多跳推理",
    "web_search": "实时 Web 搜索，适合最新信息查询",
    "calculator": "数学计算工具",
    "code_executor": "代码执行工具",
    "summarizer": "文档摘要工具",
}

RAG_STRATEGIES = {
    "simple": "单一向量检索，适合简单事实性问题",
    "hybrid": "向量 + 关键词混合检索，适合综合查询",
    "multi_hop": "多跳检索，先检索再根据结果继续检索",
    "graph": "图增强检索，利用知识图谱进行关系推理",
}

SYSTEM_PROMPT = "你是一个智能任务规划助手，根据查询类型为 RAG 系统选择最佳工具和检索策略。"

PLANNING_PROMPT_TEMPLATE = """
## 查询信息
- 原始查询: {user_query}
- 改写查询: {rewritten_query}
- 查询类型: {query_type}
- 子查询数量: {sub_query_count}

## 可用工具
{tools_description}

## 可用 RAG 策略
{strategies_description}

## 任务
根据查询信息，选择最合适的工具组合和 RAG 策略。
返回 JSON 格式：
{{
  "selected_tools": ["tool1", "tool2"],
  "rag_strategy": "strategy_name",
  "reasoning": "选择理由（简短）"
}}
"""


# ============================================================
# TaskPlanningAgent 实现
# ============================================================

class TaskPlanningAgent(BaseAgent):
    """
    任务规划 Agent
    ==============
    基于查询理解结果，通过 LLM 推理选择最优工具组合和 RAG 策略。

    规划逻辑：
    - factual        → vector_search + simple RAG
    - analytical     → vector_search + web_search + hybrid RAG
    - conversational → vector_search + simple RAG（轻量）
    - multi_hop      → graph_search + vector_search + multi_hop RAG

    LLM 在此处作为规划器，利用 Few-shot 推理能力超越硬编码规则。
    同时提供基于规则的快速路径（fast path）作为 LLM 调用失败的降级方案。
    """

    def __init__(self, settings=None) -> None:
        super().__init__("TaskPlanningAgent", settings)
        cfg = get_settings()
        self._llm = ChatOpenAI(
            model=cfg.llm.model_name,
            temperature=0.0,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.api_base,
        )

    def _execute(self, state: AgentState) -> AgentState:
        query_type: str = state.get("query_type", "factual")
        sub_queries = state.get("sub_queries", [])

        # 快速路径：对于简单查询类型直接使用规则，避免 LLM 开销
        if query_type == "conversational":
            return {
                **state,
                "selected_tools": ["vector_search"],
                "rag_strategy": "simple",
            }

        # LLM 路径：复杂查询使用 LLM 进行规划
        tools_desc = "\n".join(f"- {k}: {v}" for k, v in AVAILABLE_TOOLS.items())
        strategies_desc = "\n".join(f"- {k}: {v}" for k, v in RAG_STRATEGIES.items())

        prompt = PLANNING_PROMPT_TEMPLATE.format(
            user_query=state.get("user_query", ""),
            rewritten_query=state.get("rewritten_query", ""),
            query_type=query_type,
            sub_query_count=len(sub_queries),
            tools_description=tools_desc,
            strategies_description=strategies_desc,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self._llm.invoke(messages)
        plan = self._parse_response(response.content, query_type)

        self._logger.debug(
            f"任务规划: tools={plan['selected_tools']}, strategy={plan['rag_strategy']}"
        )

        return {
            **state,
            "selected_tools": plan["selected_tools"],
            "rag_strategy": plan["rag_strategy"],
        }

    def _parse_response(self, content: str, query_type: str) -> dict:
        """解析 LLM 规划结果，失败时使用规则降级"""
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            result = json.loads(cleaned.strip())
            return {
                "selected_tools": result.get("selected_tools", ["vector_search"]),
                "rag_strategy": result.get("rag_strategy", "simple"),
            }
        except (json.JSONDecodeError, KeyError):
            # 规则降级策略
            return self._rule_based_fallback(query_type)

    def _rule_based_fallback(self, query_type: str) -> dict:
        """基于查询类型的规则降级策略"""
        fallback_map = {
            "factual": {"selected_tools": ["vector_search"], "rag_strategy": "simple"},
            "analytical": {"selected_tools": ["vector_search", "web_search"], "rag_strategy": "hybrid"},
            "multi_hop": {"selected_tools": ["graph_search", "vector_search"], "rag_strategy": "multi_hop"},
            "conversational": {"selected_tools": ["vector_search"], "rag_strategy": "simple"},
        }
        return fallback_map.get(
            query_type,
            {"selected_tools": ["vector_search"], "rag_strategy": "simple"},
        )
