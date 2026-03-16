"""
Reflection Agent.

Responsibilities
----------------
The Reflection Agent is the **quality gate** of the pipeline.  After the
Reasoning Agent produces a candidate answer, this agent evaluates it on three
dimensions:

1. **Answer Verification** – does the answer actually address the original query?
2. **Hallucination Detection** – are all factual claims in the answer supported
   by the retrieved context?  Unsupported claims are flagged.
3. **Confidence Scoring** – a scalar confidence score in ``[0, 1]`` is assigned.

If confidence is below ``settings.confidence_threshold``, the agent signals
the pipeline to retry (e.g. re-retrieve with different strategy or re-reason).

Algorithm / Calling Relationship
----------------------------------
    ReasoningAgent
        └─▶ ReflectionAgent.execute(AgentInput)
                └─▶  _verify_answer(query, answer, context) → bool
                └─▶  _detect_hallucinations(answer, context) → List[str]
                └─▶  _score_confidence(verified, hallucinations) → float
                └─▶  returns AgentOutput(result=ReflectionResult)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from config.settings import settings
from utils.helpers import build_llm_client, get_model_name, truncate_text


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ReflectionResult:
    """
    Output produced by the :class:`ReflectionAgent`.

    Attributes:
        answer:              The candidate answer being evaluated.
        is_verified:         True if the answer addresses the query.
        hallucinations:      List of potentially unsupported claims.
        confidence:          Overall confidence score in ``[0, 1]``.
        should_retry:        True if confidence < threshold; pipeline should retry.
        retry_reason:        Human-readable reason for requesting a retry.
        improvement_hints:   Suggestions for improving retrieval / reasoning.
    """

    answer: str
    is_verified: bool = True
    hallucinations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    should_retry: bool = False
    retry_reason: str = ""
    improvement_hints: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReflectionAgent(BaseAgent):
    """
    Evaluate the quality of a candidate answer and decide whether to retry.

    This agent is crucial for preventing hallucinations and ensuring that
    answers are grounded in the retrieved context.
    """

    def __init__(self) -> None:
        super().__init__("ReflectionAgent")
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Evaluate the candidate answer.

        Args:
            agent_input: ``metadata["candidate_answer"]`` holds the draft answer;
                         ``metadata["context_str"]`` holds the source context;
                         ``query`` is the original user query.

        Returns:
            :class:`AgentOutput` whose ``result`` is a :class:`ReflectionResult`.
        """
        query = agent_input.query
        answer: str = agent_input.metadata.get("candidate_answer", "")
        context_str: str = agent_input.metadata.get("context_str", "")

        if not answer:
            return AgentOutput(
                result=ReflectionResult(
                    answer="",
                    is_verified=False,
                    confidence=0.0,
                    should_retry=True,
                    retry_reason="Empty candidate answer.",
                ),
                confidence=0.0,
            )

        is_verified = self._verify_answer(query, answer, context_str)
        hallucinations = self._detect_hallucinations(answer, context_str)
        confidence = self._score_confidence(is_verified, hallucinations)
        should_retry = confidence < settings.confidence_threshold

        hints = self._generate_hints(query, answer, hallucinations, confidence)

        result = ReflectionResult(
            answer=answer,
            is_verified=is_verified,
            hallucinations=hallucinations,
            confidence=confidence,
            should_retry=should_retry,
            retry_reason=(
                f"Confidence {confidence:.2f} below threshold {settings.confidence_threshold}"
                if should_retry else ""
            ),
            improvement_hints=hints,
        )

        self.logger.info(
            "Reflection: verified=%s, hallucinations=%d, confidence=%.2f, retry=%s",
            is_verified, len(hallucinations), confidence, should_retry,
        )

        return AgentOutput(
            result=result,
            metadata={
                "is_verified": is_verified,
                "hallucination_count": len(hallucinations),
                "confidence": confidence,
                "should_retry": should_retry,
            },
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm_client()
        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        api_key = settings.openai_api_key or settings.azure_openai_api_key
        if not api_key:
            return "__MOCK__"
        client = self._get_llm()
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def _verify_answer(self, query: str, answer: str, context_str: str) -> bool:
        """
        Check whether the answer addresses the query.

        The LLM is asked to verify with a binary yes/no response.
        Falls back to True if the LLM is unavailable.

        Args:
            query:       Original user query.
            answer:      Candidate answer.
            context_str: Retrieved context (used as evidence).

        Returns:
            True if the answer is relevant and addresses the query.
        """
        system_prompt = (
            "You are a quality evaluation assistant. "
            "Given a query and an answer, determine whether the answer "
            "adequately addresses the query. "
            "Reply with exactly 'YES' or 'NO'."
        )
        user_prompt = f"Query: {query}\n\nAnswer: {answer}"
        raw = self._call_llm(system_prompt, user_prompt)
        if raw == "__MOCK__":
            return True
        return raw.strip().upper().startswith("YES")

    def _detect_hallucinations(
        self, answer: str, context_str: str
    ) -> List[str]:
        """
        Identify claims in the answer that are not grounded in the context.

        The LLM is asked to list any claims in the answer that cannot be
        verified from the context, returning a JSON list of strings.

        Args:
            answer:      Candidate answer.
            context_str: Retrieved context to check against.

        Returns:
            List of potentially hallucinated claim strings.
        """
        if not context_str:
            return []

        context_truncated = truncate_text(context_str, max_tokens=3000)
        system_prompt = (
            "You are a hallucination detection assistant. "
            "Given a context and an answer, identify any factual claims in the answer "
            "that are NOT supported by the context. "
            "Return a JSON array of strings (the unsupported claims). "
            "If all claims are supported, return an empty array []."
        )
        user_prompt = (
            f"Context:\n{context_truncated}\n\n"
            f"Answer:\n{answer}"
        )
        raw = self._call_llm(system_prompt, user_prompt)
        if raw == "__MOCK__":
            return []
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

    def _score_confidence(
        self, is_verified: bool, hallucinations: List[str]
    ) -> float:
        """
        Compute a confidence score based on verification and hallucination count.

        Scoring formula:
        - Base: 1.0
        - ``is_verified`` False: subtract 0.3
        - Each hallucination: subtract 0.15 (capped at 0.6 total deduction)

        Args:
            is_verified:    Whether the answer addresses the query.
            hallucinations: List of detected hallucinated claims.

        Returns:
            Confidence score in ``[0, 1]``.
        """
        score = 1.0
        if not is_verified:
            score -= 0.3
        hallucination_penalty = min(len(hallucinations) * 0.15, 0.6)
        score -= hallucination_penalty
        return max(0.0, round(score, 3))

    def _generate_hints(
        self,
        query: str,
        answer: str,
        hallucinations: List[str],
        confidence: float,
    ) -> List[str]:
        """
        Generate actionable improvement hints for the Self-Improving layer.

        Args:
            query:          Original user query.
            answer:         Candidate answer.
            hallucinations: Detected hallucinated claims.
            confidence:     Computed confidence score.

        Returns:
            List of hint strings.
        """
        hints: List[str] = []
        if confidence < 0.5:
            hints.append("Consider broadening retrieval to include more diverse sources.")
        if hallucinations:
            hints.append(
                f"Detected {len(hallucinations)} unsupported claim(s); "
                "try retrieving more specific context."
            )
        if confidence < settings.confidence_threshold:
            hints.append(
                "Consider using hybrid retrieval or switching to web search "
                "for more up-to-date information."
            )
        return hints
