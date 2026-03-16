"""
LLM-as-Judge Evaluator.

Uses an LLM to score answers across four standard RAG evaluation dimensions.
This is the default evaluation backend and requires no additional dependencies
beyond the OpenAI client.

Algorithm
---------
For each metric, a carefully designed prompt is sent to the LLM which returns
a JSON object with a score and brief explanation:

1. **Faithfulness**: is every claim in the answer grounded in the context?
2. **Answer Relevance**: does the answer address the user's question?
3. **Context Precision**: do the retrieved documents primarily contain relevant
   information (low noise)?
4. **Context Recall**: does the retrieved context cover all aspects needed to
   answer the question?

Scoring
-------
All metrics are normalised to ``[0, 1]``.  Each prompt instructs the LLM to
return a score on a 0–10 scale which is divided by 10.

Reference
---------
- Zheng et al., 2023 – "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config.settings import settings
from evaluation.evaluator import EvaluationResult
from utils.helpers import build_llm_client, get_model_name, truncate_text


class LLMJudge:
    """
    LLM-based evaluation using a single model as the judge.

    All four metrics are evaluated in separate LLM calls to ensure focused,
    accurate scoring.
    """

    def __init__(self) -> None:
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        answer: str,
        context: List[str],
        reference_answer: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate the answer using LLM-as-Judge.

        Args:
            query:            User query.
            answer:           Generated answer to evaluate.
            context:          List of context strings used during generation.
            reference_answer: Optional gold-standard answer.

        Returns:
            :class:`EvaluationResult` with individual metric scores.
        """
        context_str = "\n\n".join(context)

        faithfulness = self._score_faithfulness(answer, context_str)
        answer_relevance = self._score_answer_relevance(query, answer)
        context_precision = self._score_context_precision(query, context_str)
        context_recall = self._score_context_recall(
            query, context_str, reference_answer or answer
        )

        return EvaluationResult(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm_client()
        return self._llm

    def _judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Call the judge LLM and return a parsed JSON response.

        The response format expected from the LLM is::

            {"score": 8, "explanation": "..."}

        Falls back to a mock score of 7 when no API key is configured.

        Args:
            system_prompt: System instruction for the judge.
            user_prompt:   Content to evaluate.

        Returns:
            Dict with ``"score"`` (0–10) and ``"explanation"`` keys.
        """
        api_key = settings.openai_api_key or settings.azure_openai_api_key
        if not api_key:
            return {"score": 7, "explanation": "Mock evaluation (no API key)."}

        try:
            client = self._get_llm()
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:  # noqa: BLE001
            return {"score": 5, "explanation": "Evaluation error."}

    def _normalise(self, result: Dict[str, Any]) -> float:
        """Convert a 0–10 score dict to a 0–1 float."""
        return min(1.0, max(0.0, result.get("score", 5) / 10.0))

    def _score_faithfulness(self, answer: str, context_str: str) -> float:
        system_prompt = (
            "You are an evaluation assistant assessing factual faithfulness. "
            "Rate how well every claim in the answer is supported by the context. "
            "Return JSON: {\"score\": <0-10>, \"explanation\": \"...\"}. "
            "10 = all claims fully supported; 0 = answer is entirely hallucinated."
        )
        ctx = truncate_text(context_str, max_tokens=2000)
        user_prompt = f"Context:\n{ctx}\n\nAnswer:\n{answer}"
        return self._normalise(self._judge(system_prompt, user_prompt))

    def _score_answer_relevance(self, query: str, answer: str) -> float:
        system_prompt = (
            "You are an evaluation assistant assessing answer relevance. "
            "Rate how well the answer addresses the user's question. "
            "Return JSON: {\"score\": <0-10>, \"explanation\": \"...\"}. "
            "10 = perfectly on-topic; 0 = completely off-topic."
        )
        user_prompt = f"Question: {query}\n\nAnswer: {answer}"
        return self._normalise(self._judge(system_prompt, user_prompt))

    def _score_context_precision(self, query: str, context_str: str) -> float:
        system_prompt = (
            "You are an evaluation assistant assessing retrieval precision. "
            "Rate how much of the retrieved context is actually relevant to the query. "
            "Return JSON: {\"score\": <0-10>, \"explanation\": \"...\"}. "
            "10 = all context is highly relevant; 0 = all context is noise."
        )
        ctx = truncate_text(context_str, max_tokens=2000)
        user_prompt = f"Query: {query}\n\nContext:\n{ctx}"
        return self._normalise(self._judge(system_prompt, user_prompt))

    def _score_context_recall(
        self, query: str, context_str: str, reference: str
    ) -> float:
        system_prompt = (
            "You are an evaluation assistant assessing retrieval recall. "
            "Rate how well the retrieved context covers all information needed "
            "to fully answer the reference answer. "
            "Return JSON: {\"score\": <0-10>, \"explanation\": \"...\"}. "
            "10 = context contains everything needed; 0 = critical information missing."
        )
        ctx = truncate_text(context_str, max_tokens=2000)
        user_prompt = (
            f"Query: {query}\n\n"
            f"Reference Answer: {reference[:500]}\n\n"
            f"Retrieved Context:\n{ctx}"
        )
        return self._normalise(self._judge(system_prompt, user_prompt))
