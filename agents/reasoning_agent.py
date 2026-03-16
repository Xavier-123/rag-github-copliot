"""
Reasoning Agent.

Responsibilities
----------------
Perform complex, multi-step reasoning over the prepared context before
answer generation.  Supported reasoning modes:

1. **Chain-of-Thought (CoT)** – the LLM is prompted to reason step-by-step
   before giving an answer.  This improves accuracy on analytical and complex
   queries.
2. **Multi-hop Reasoning** – iteratively retrieve → reason → retrieve again
   until the answer can be confidently derived or the iteration limit is
   reached.
3. **Tool Use** – the Reasoning Agent can invoke external tools (calculator,
   code executor, database lookup) during its reasoning loop.

Algorithm / Calling Relationship
----------------------------------
    ContextEngineeringAgent
        └─▶ ReasoningAgent.execute(AgentInput)
                ├─▶ (if multi-hop) loop:
                │     _reason_step(query, context) → intermediate_result
                │     _need_more_retrieval(intermediate) → bool
                │     if True: calls MultiRetrieverAgent again (via pipeline)
                ├─▶ _final_reasoning(query, context, chain) → reasoning_output
                └─▶ returns AgentOutput(result=ReasoningOutput)

The agent has access to the full task plan so it knows whether multi-hop
reasoning is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from config.settings import settings
from utils.helpers import build_llm_client, get_model_name, is_llm_available, truncate_text


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ReasoningOutput:
    """
    Structured output from the Reasoning Agent.

    Attributes:
        reasoning_chain: Ordered list of reasoning steps (thoughts).
        intermediate_answers: Partial answers produced during multi-hop.
        final_reasoning:      Consolidated reasoning narrative.
        requires_more_info:   True if the agent determined retrieval is incomplete.
        follow_up_queries:    Suggested follow-up retrieval queries.
    """

    reasoning_chain: List[str] = field(default_factory=list)
    intermediate_answers: List[str] = field(default_factory=list)
    final_reasoning: str = ""
    requires_more_info: bool = False
    follow_up_queries: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReasoningAgent(BaseAgent):
    """
    Multi-step reasoning over retrieved context.

    The agent supports three reasoning modes selected automatically based on
    task plan flags:

    - **single-step CoT**  – for simple / moderate queries.
    - **multi-hop**        – for complex queries requiring iterative retrieval.
    - **tool-augmented**   – for queries where external tool calls are needed
                             (placeholder; extend ``_call_tool`` to add tools).
    """

    def __init__(self) -> None:
        super().__init__("ReasoningAgent")
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Execute the reasoning pipeline.

        Args:
            agent_input: ``query`` is the user query; ``context`` (as a string
                         in ``metadata["context_str"]``) is the prepared context;
                         ``metadata["task_plan"]`` drives reasoning mode selection.

        Returns:
            :class:`AgentOutput` whose ``result`` is a :class:`ReasoningOutput`.
        """
        query = agent_input.query
        context_str: str = agent_input.metadata.get("context_str", "")
        task_plan: Dict[str, Any] = agent_input.metadata.get("task_plan", {})

        requires_multi_hop: bool = task_plan.get("requires_multi_hop", False)
        max_iters = settings.max_reasoning_iterations

        if requires_multi_hop:
            output = self._multi_hop_reasoning(query, context_str, max_iters)
        else:
            output = self._chain_of_thought(query, context_str)

        self.logger.info(
            "Reasoning complete: steps=%d, requires_more_info=%s",
            len(output.reasoning_chain), output.requires_more_info,
        )

        return AgentOutput(
            result=output,
            metadata={
                "reasoning_steps": len(output.reasoning_chain),
                "requires_more_info": output.requires_more_info,
                "follow_up_queries": output.follow_up_queries,
            },
            confidence=0.85,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm_client()
        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        if not is_llm_available():
            return "__MOCK_REASONING__"
        client = self._get_llm()
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _chain_of_thought(self, query: str, context_str: str) -> ReasoningOutput:
        """
        Single-pass chain-of-thought reasoning.

        The LLM is prompted with ``<think>`` markers to encourage step-by-step
        reasoning before producing an answer.

        Args:
            query:       User query.
            context_str: Formatted context from the Context Engineering Agent.

        Returns:
            :class:`ReasoningOutput` with the reasoning chain populated.
        """
        system_prompt = (
            "You are an expert reasoning assistant. "
            "Use the provided context to reason step-by-step about the question. "
            "Structure your response as:\n"
            "THOUGHT 1: ...\nTHOUGHT 2: ...\n...\nFINAL REASONING: ...\n"
            "Be precise and cite specific documents when relevant."
        )
        context_truncated = truncate_text(context_str, max_tokens=6000)
        user_prompt = (
            f"Context:\n{context_truncated}\n\n"
            f"Question: {query}\n\n"
            "Reason step-by-step:"
        )

        raw = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

        if raw == "__MOCK_REASONING__":
            return ReasoningOutput(
                reasoning_chain=["Mock reasoning step 1", "Mock reasoning step 2"],
                final_reasoning=f"Mock reasoning for: {query}",
            )

        # Parse the response into structured chain
        chain, final = self._parse_reasoning(raw)
        return ReasoningOutput(reasoning_chain=chain, final_reasoning=final)

    def _multi_hop_reasoning(
        self, query: str, context_str: str, max_iters: int
    ) -> ReasoningOutput:
        """
        Iterative multi-hop reasoning: reason → check if more info needed → repeat.

        At each iteration the agent either:
        - Produces a confident intermediate answer → continues to next hop.
        - Determines it has enough information → finalises reasoning.
        - Requests a follow-up query → signals the pipeline to re-retrieve.

        Args:
            query:     User query.
            context_str: Initial context.
            max_iters:   Maximum reasoning iterations.

        Returns:
            :class:`ReasoningOutput` with full chain and follow-up queries.
        """
        chain: List[str] = []
        intermediate_answers: List[str] = []
        current_context = context_str

        for iteration in range(1, max_iters + 1):
            self.logger.debug("Multi-hop iteration %d/%d", iteration, max_iters)

            step_prompt = (
                "You are performing multi-hop reasoning. "
                "Given the current context and previous reasoning steps, "
                "either:\n"
                "1. Provide the next reasoning step and a partial answer.\n"
                "2. If insufficient information, output: NEED_MORE_INFO: <follow-up query>\n"
                "3. If you can give the final answer, output: DONE: <final reasoning>\n"
            )
            history_str = "\n".join(f"Step {i}: {s}" for i, s in enumerate(chain, 1))
            user_msg = (
                f"Original query: {query}\n\n"
                f"Context:\n{truncate_text(current_context, 4000)}\n\n"
                f"Previous steps:\n{history_str}\n\n"
                "Next:"
            )

            raw = self._call_llm(step_prompt, user_msg, max_tokens=512)

            if raw == "__MOCK_REASONING__":
                chain.append(f"Mock hop {iteration}")
                break

            if raw.startswith("DONE:"):
                final_reasoning = raw[5:].strip()
                chain.append(f"[Final] {final_reasoning}")
                return ReasoningOutput(
                    reasoning_chain=chain,
                    intermediate_answers=intermediate_answers,
                    final_reasoning=final_reasoning,
                    requires_more_info=False,
                )

            if raw.startswith("NEED_MORE_INFO:"):
                follow_up = raw[15:].strip()
                chain.append(f"[Need more info] {follow_up}")
                return ReasoningOutput(
                    reasoning_chain=chain,
                    intermediate_answers=intermediate_answers,
                    final_reasoning="",
                    requires_more_info=True,
                    follow_up_queries=[follow_up],
                )

            chain.append(raw)
            intermediate_answers.append(raw)

        # Max iterations reached – synthesise final reasoning
        final = self._synthesise_chain(query, chain)
        return ReasoningOutput(
            reasoning_chain=chain,
            intermediate_answers=intermediate_answers,
            final_reasoning=final,
        )

    def _synthesise_chain(self, query: str, chain: List[str]) -> str:
        """
        Produce a final reasoning summary from the accumulated reasoning chain.

        Args:
            query: The original user query.
            chain: Accumulated reasoning steps.

        Returns:
            Concise final reasoning string.
        """
        steps_str = "\n".join(f"{i}. {s}" for i, s in enumerate(chain, 1))
        raw = self._call_llm(
            "Summarise the following reasoning steps into a concise final reasoning narrative.",
            f"Query: {query}\n\nSteps:\n{steps_str}",
            max_tokens=512,
        )
        return raw if raw != "__MOCK_REASONING__" else steps_str

    @staticmethod
    def _parse_reasoning(raw: str) -> tuple[List[str], str]:
        """
        Parse a CoT LLM response into a chain and final reasoning.

        The expected format is::

            THOUGHT 1: ...
            THOUGHT 2: ...
            FINAL REASONING: ...

        Args:
            raw: Raw LLM response text.

        Returns:
            ``(chain, final_reasoning)`` tuple.
        """
        lines = raw.strip().split("\n")
        chain: List[str] = []
        final = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("FINAL REASONING:"):
                final = line.split(":", 1)[1].strip()
            elif line.upper().startswith("THOUGHT"):
                thought = line.split(":", 1)[1].strip() if ":" in line else line
                chain.append(thought)
        return chain, final or raw  # fallback: use raw as final reasoning
