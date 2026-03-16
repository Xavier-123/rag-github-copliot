"""
Reflection Agent（反思验证 Agent）
===================================
对推理结果进行质量验证，检测幻觉并决定是否需要重新检索或生成：
  1. Answer Verification    : 验证答案是否有文档依据
  2. Hallucination Detection: 检测答案中是否包含文档未支持的断言
  3. Self-Reflection        : 评估答案完整性和相关性

工作流位置：ReasoningAgent → [ReflectionAgent] → MemoryUpdate (passed) / 重试 (failed)

调用关系：
  - 输入 state 字段  : answer, context, rewritten_query, reasoning_steps
  - 输出 state 字段  : reflection_result（passed/failed/uncertain）, reflection_feedback

反思判断流程：
  1. LLM 判断答案是否基于提供的上下文
  2. 若通过，写入 reflection_result="passed"
  3. 若发现幻觉或答案不完整，写入 reflection_result="failed" 和改进建议
  4. LangGraph 路由函数根据 reflection_result 决定：继续 or 重试（最多 max_retries 次）
"""

from __future__ import annotations

import json

from rag_system.agents.base import AgentState, BaseAgent
from rag_system.config.settings import get_settings


REFLECTION_SYSTEM_PROMPT = """你是一个严格的答案质量审核员。
请客观评估 AI 助手的回答质量，检查是否存在幻觉或与参考文档不符的内容。"""

REFLECTION_PROMPT_TEMPLATE = """
## 用户问题
{query}

## 参考文档
{context}

## AI 回答
{answer}

## 推理步骤
{reasoning_steps}

## 审核任务
请从以下维度评估答案质量，并以 JSON 格式返回结果：

1. **有依据性（groundedness）**: 答案中的关键断言是否有参考文档支撑？
2. **相关性（relevance）**: 答案是否直接回答了用户问题？
3. **完整性（completeness）**: 答案是否充分回答了问题的所有方面？
4. **幻觉检测（hallucination）**: 是否存在文档中未提及但被确信陈述的内容？

## 输出格式（JSON）
{{
  "result": "passed|failed|uncertain",
  "groundedness_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "hallucination_detected": true|false,
  "feedback": "改进建议（若 result != passed）",
  "issues": ["具体问题1", "具体问题2"]
}}

判断标准：
- passed    : 三项分数均 ≥ 0.7 且无幻觉
- failed    : 任一分数 < 0.5 或存在幻觉
- uncertain : 介于两者之间，需要更多信息
"""


class ReflectionAgent(BaseAgent):
    """
    反思 Agent
    ==========
    通过 LLM 自评估机制验证答案质量，实现 Self-RAG 式的循环优化。

    Self-RAG 流程：
    1. ReasoningAgent 生成初始答案
    2. ReflectionAgent 评估答案质量
    3. 若质量不达标（failed），触发 LangGraph 重试路由
    4. 重新执行检索+生成（最多 max_retries 次）
    5. 最终通过或超过重试次数时结束

    幻觉检测机制：
    - 使用 LLM 作为判断器（LLM-as-Judge）评估事实一致性
    - 相比关键词匹配，LLM 评判更能理解语义层面的不一致性
    """

    def __init__(self, settings=None) -> None:
        super().__init__("ReflectionAgent", settings)
        cfg = get_settings()
        self._cfg = cfg
        self._max_retries: int = cfg.max_retries
        self._llm = None  # 懒加载

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self._cfg.llm.model_name,
                temperature=0.0,
                api_key=self._cfg.llm.api_key,
                base_url=self._cfg.llm.api_base,
            )
        return self._llm

    def _execute(self, state: AgentState) -> AgentState:
        answer: str = state.get("answer", "")
        context: str = state.get("context", "")
        query: str = state.get("rewritten_query", state.get("user_query", ""))
        reasoning_steps = state.get("reasoning_steps", [])
        retry_count: int = state.get("retry_count", 0)

        if not answer:
            return {
                **state,
                "reflection_result": "failed",
                "reflection_feedback": "答案为空，需要重新生成",
            }

        # 若已超过最大重试次数，强制通过（避免无限循环）
        if retry_count >= self._max_retries:
            self._logger.warning(f"已达最大重试次数 {self._max_retries}，强制通过")
            return {
                **state,
                "reflection_result": "passed",
                "reflection_feedback": f"达到最大重试次数 {self._max_retries}，接受当前答案",
            }

        # LLM 评估
        steps_text = "\n".join(f"- {s}" for s in reasoning_steps) if reasoning_steps else "（无推理步骤）"
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            query=query,
            context=context[:4000] if context else "（无参考文档）",
            answer=answer,
            reasoning_steps=steps_text,
        )

        from langchain.schema import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self._get_llm().invoke(messages)
        eval_result = self._parse_response(response.content)

        self._logger.info(
            f"反思评估: result={eval_result['result']}, "
            f"groundedness={eval_result.get('groundedness_score', 'N/A')}, "
            f"hallucination={eval_result.get('hallucination_detected', False)}"
        )

        return {
            **state,
            "reflection_result": eval_result["result"],
            "reflection_feedback": eval_result.get("feedback", ""),
        }

    def _parse_response(self, content: str) -> dict:
        """解析反思评估 JSON 结果"""
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, KeyError) as e:
            self._logger.warning(f"反思结果解析失败，默认 passed: {e}")
            return {"result": "passed", "feedback": ""}
