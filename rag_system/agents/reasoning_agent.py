"""
Reasoning Agent（推理与答案生成 Agent）
======================================
使用大语言模型基于上下文进行推理并生成最终答案：
  - Chain-of-Thought (CoT) : 逐步推理，提高复杂问题准确性
  - Tool-Augmented Reasoning : 结合工具调用增强推理能力

工作流位置：ContextEngineeringAgent → [ReasoningAgent] → ReflectionAgent

调用关系：
  - 输入 state 字段  : context, rewritten_query, query_type, memory_context, selected_tools
  - 输出 state 字段  : answer, reasoning_steps

推理策略：
  - factual / conversational : 标准 RAG 回答（单步推理）
  - analytical / multi_hop   : Chain-of-Thought 多步推理
"""

from __future__ import annotations

import re
from typing import List, Tuple

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_system.agents.base import AgentState, BaseAgent
from rag_system.config.settings import get_settings


# ============================================================
# 推理系统提示
# ============================================================

RAG_SYSTEM_PROMPT = """你是一个专业的知识问答助手。
请基于提供的参考文档回答用户问题。
- 回答要准确、有据可依
- 若文档中无相关信息，请明确说明，不要编造
- 使用中文回答"""

COT_SYSTEM_PROMPT = """你是一个擅长逐步推理的专业助手。
请使用 Chain-of-Thought 方法，通过分步思考来解决复杂问题。
格式要求：
<thinking>
步骤1: ...
步骤2: ...
步骤N: ...
</thinking>
<answer>
最终答案
</answer>"""

RAG_PROMPT_TEMPLATE = """## 参考文档
{context}

## 历史对话记录
{memory_context}

## 用户问题
{query}

请基于参考文档回答用户问题。若文档信息不足，请说明。"""

COT_PROMPT_TEMPLATE = """## 参考文档
{context}

## 历史对话记录
{memory_context}

## 用户问题（需要逐步推理）
{query}

请先在 <thinking> 标签内逐步分析，再在 <answer> 标签内给出最终答案。"""


class ReasoningAgent(BaseAgent):
    """
    推理 Agent
    ==========
    根据查询类型选择合适的推理策略生成高质量答案。

    推理策略选择：
    - factual/conversational → 标准 RAG 推理（效率优先）
    - analytical/multi_hop   → Chain-of-Thought 推理（准确性优先）

    CoT 解析机制：
    - 从 LLM 输出中提取 <thinking> 和 <answer> 两个标签内容
    - reasoning_steps 存储中间推理步骤，便于调试和反思验证
    - 若 CoT 解析失败，使用完整输出作为 answer（降级策略）
    """

    def __init__(self, settings=None) -> None:
        super().__init__("ReasoningAgent", settings)
        cfg = get_settings()
        self._llm = ChatOpenAI(
            model=cfg.llm.model_name,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.api_base,
        )

    def _execute(self, state: AgentState) -> AgentState:
        context: str = state.get("context", "")
        query: str = state.get("rewritten_query", state.get("user_query", ""))
        query_type: str = state.get("query_type", "factual")
        memory_context: str = state.get("memory_context", "无历史记录")

        # 根据查询类型选择推理策略
        use_cot = query_type in ("analytical", "multi_hop")

        if use_cot:
            answer, reasoning_steps = self._cot_reasoning(query, context, memory_context)
        else:
            answer, reasoning_steps = self._standard_reasoning(query, context, memory_context)

        self._logger.info(
            f"推理完成: 策略={'CoT' if use_cot else '标准'}, "
            f"答案长度={len(answer)}, 推理步骤={len(reasoning_steps)}"
        )

        return {**state, "answer": answer, "reasoning_steps": reasoning_steps}

    def _standard_reasoning(
        self, query: str, context: str, memory_context: str
    ) -> Tuple[str, List[str]]:
        """标准 RAG 推理：直接基于上下文生成答案"""
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context or "（无相关参考文档）",
            memory_context=memory_context,
            query=query,
        )
        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self._llm.invoke(messages)
        return response.content, []

    def _cot_reasoning(
        self, query: str, context: str, memory_context: str
    ) -> Tuple[str, List[str]]:
        """
        Chain-of-Thought 推理
        ======================
        通过 <thinking>...</thinking><answer>...</answer> 结构化输出，
        分离推理过程和最终答案，提高复杂问题的准确性。
        """
        prompt = COT_PROMPT_TEMPLATE.format(
            context=context or "（无相关参考文档）",
            memory_context=memory_context,
            query=query,
        )
        messages = [
            SystemMessage(content=COT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self._llm.invoke(messages)
        return self._parse_cot_response(response.content)

    def _parse_cot_response(self, content: str) -> Tuple[str, List[str]]:
        """
        解析 CoT 格式输出。
        提取 <thinking> 中的推理步骤和 <answer> 中的最终答案。
        """
        # 提取推理步骤
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

        if thinking_match and answer_match:
            thinking_text = thinking_match.group(1).strip()
            # 按步骤分割推理文本
            steps = [
                step.strip()
                for step in re.split(r"步骤\d+[:：]|\n\n", thinking_text)
                if step.strip()
            ]
            answer = answer_match.group(1).strip()
            return answer, steps
        else:
            # 降级：使用完整输出作为答案
            self._logger.warning("CoT 格式解析失败，使用完整输出")
            return content, []
