"""
Query Understanding Agent（查询理解 Agent）
==========================================
负责对用户原始查询进行三步处理：
  1. 查询改写（Query Rewrite）    : 消除歧义，补充上下文信息
  2. 查询分类（Query Classify）   : 判断问题类型（factual/analytical/conversational/multi_hop）
  3. 查询分解（Query Decompose）  : 将复杂多跳问题拆解为子问题列表

工作流位置：User Query → [QueryUnderstandingAgent] → TaskPlanningAgent

调用关系：
  - 输入 state 字段  : user_query, memory_context
  - 输出 state 字段  : rewritten_query, query_type, sub_queries
"""

from __future__ import annotations

import json
from typing import List

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_system.agents.base import AgentState, BaseAgent
from rag_system.config.settings import get_settings


# ============================================================
# 提示词模板
# ============================================================

SYSTEM_PROMPT = """你是一个专业的查询理解助手，负责对用户问题进行分析和优化。
请严格按照指定的 JSON 格式返回结果，不要包含额外文字。"""

UNDERSTANDING_PROMPT_TEMPLATE = """
请对以下用户查询进行分析，并以 JSON 格式返回结果。

## 历史上下文
{memory_context}

## 用户查询
{user_query}

## 分析任务
1. **查询改写（rewritten_query）**: 结合历史上下文，消除指代词歧义，使查询更加清晰完整。
2. **查询分类（query_type）**: 从以下类型中选择一个：
   - `factual`       : 需要从知识库中检索事实性答案
   - `analytical`    : 需要分析推理，综合多个信息来源
   - `conversational`: 闲聊或上下文依赖型对话
   - `multi_hop`     : 需要多步推理，将问题拆解为子问题
3. **子查询分解（sub_queries）**: 若类型为 `multi_hop`，将问题分解为 2-4 个子问题；其他类型返回空列表。

## 输出格式（JSON）
{{
  "rewritten_query": "改写后的查询",
  "query_type": "factual|analytical|conversational|multi_hop",
  "sub_queries": ["子问题1", "子问题2"]
}}
"""


# ============================================================
# QueryUnderstandingAgent 实现
# ============================================================

class QueryUnderstandingAgent(BaseAgent):
    """
    查询理解 Agent
    ==============
    通过 LLM 对用户查询进行改写、分类和分解，为后续 Agent 提供高质量的结构化查询信息。

    核心算法：
    - 使用 LLM Few-shot Prompting 实现三合一分析（改写+分类+分解）
    - 单次 LLM 调用完成所有分析，减少延迟
    - 对 LLM 输出进行 JSON 解析与降级处理（fallback）
    """

    def __init__(self, settings=None) -> None:
        super().__init__("QueryUnderstandingAgent", settings)
        cfg = get_settings()
        self._llm = ChatOpenAI(
            model=cfg.llm.model_name,
            temperature=0.0,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.api_base,
        )

    def _execute(self, state: AgentState) -> AgentState:
        user_query: str = state.get("user_query", "")
        memory_context: str = state.get("memory_context", "无历史记录")

        # 构建 LLM 提示
        prompt = UNDERSTANDING_PROMPT_TEMPLATE.format(
            user_query=user_query,
            memory_context=memory_context,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 调用 LLM
        response = self._llm.invoke(messages)
        result = self._parse_response(response.content, user_query)

        self._logger.debug(
            f"查询理解结果: type={result['query_type']}, "
            f"rewritten='{result['rewritten_query'][:60]}', "
            f"sub_queries={len(result['sub_queries'])}"
        )

        return {
            **state,
            "rewritten_query": result["rewritten_query"],
            "query_type": result["query_type"],
            "sub_queries": result["sub_queries"],
        }

    def _parse_response(self, content: str, original_query: str) -> dict:
        """
        解析 LLM 返回的 JSON 响应。
        若解析失败，使用降级策略：保留原始查询，类型设为 factual。
        """
        try:
            # 清理 Markdown 代码块标记
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            result = json.loads(cleaned.strip())
            return {
                "rewritten_query": result.get("rewritten_query", original_query),
                "query_type": result.get("query_type", "factual"),
                "sub_queries": result.get("sub_queries", []),
            }
        except (json.JSONDecodeError, KeyError) as e:
            self._logger.warning(f"JSON 解析失败，使用降级策略: {e}")
            return {
                "rewritten_query": original_query,
                "query_type": "factual",
                "sub_queries": [],
            }
