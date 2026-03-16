"""
Evaluation Layer – Coordinator.

This module provides the top-level :class:`Evaluator` that dispatches to the
appropriate evaluation framework based on ``settings.evaluation_framework``:

- ``"ragas"``    → :class:`~evaluation.ragas_evaluator.RagasEvaluator`
- ``"llm_judge"``→ :class:`~evaluation.llm_judge.LLMJudge`
- ``"dspy"``     → DSPy Evaluate (placeholder, extend as needed)

The Evaluator produces a standardised :class:`EvaluationResult` that is
consumed by the Feedback Learning Layer.

Evaluation Metrics
------------------
- **Faithfulness**: are the claims in the answer supported by the context?
- **Answer Relevance**: how relevant is the answer to the question?
- **Context Precision**: how precise are the retrieved documents?
- **Context Recall**: how well does the context cover the gold answer?
- **Overall Score**: weighted average of the above metrics.

Calling Relationship
--------------------
    RAGPipeline (after ReflectionAgent)
        └─▶ Evaluator.evaluate(query, answer, context, reference) → EvaluationResult
                └─▶ RagasEvaluator or LLMJudge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings
from utils.helpers import get_logger


logger = get_logger("Evaluator")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Standardised evaluation result returned by all evaluation frameworks.

    Attributes:
        faithfulness:      Score for factual groundedness (0–1).
        answer_relevance:  Score for answer-query relevance (0–1).
        context_precision: Score for retrieval precision (0–1).
        context_recall:    Score for retrieval recall (0–1).
        overall_score:     Weighted average of all metrics (0–1).
        framework:         Name of the evaluation framework used.
        details:           Arbitrary extra details from the evaluator.
        error:             Non-empty if evaluation failed.
    """

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    framework: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Top-level evaluation coordinator.

    Selects and invokes the appropriate backend evaluator, then normalises
    the result into a uniform :class:`EvaluationResult`.
    """

    # Metric weights for overall score computation
    _WEIGHTS = {
        "faithfulness": 0.35,
        "answer_relevance": 0.35,
        "context_precision": 0.15,
        "context_recall": 0.15,
    }

    def __init__(self) -> None:
        self._framework = settings.evaluation_framework
        self._backend = self._load_backend()

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
        Evaluate the quality of an answer against the query and context.

        Args:
            query:            The user query.
            answer:           The generated answer to evaluate.
            context:          List of context strings used to generate the answer.
            reference_answer: Optional gold-standard answer for recall computation.

        Returns:
            :class:`EvaluationResult` with all metric scores populated.
        """
        if not settings.evaluation_enabled:
            logger.info("Evaluation disabled by configuration.")
            return EvaluationResult(overall_score=1.0, framework="disabled")

        try:
            result = self._backend.evaluate(query, answer, context, reference_answer)
            result.overall_score = self._compute_overall(result)
            result.framework = self._framework
            logger.info(
                "Evaluation complete: overall=%.3f, faithfulness=%.3f, relevance=%.3f",
                result.overall_score, result.faithfulness, result.answer_relevance,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Evaluation failed: %s", exc)
            return EvaluationResult(error=str(exc), framework=self._framework)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_backend(self) -> Any:
        """Instantiate the evaluation backend specified in settings."""
        if self._framework == "ragas":
            from evaluation.ragas_evaluator import RagasEvaluator  # noqa: PLC0415
            return RagasEvaluator()
        if self._framework == "llm_judge":
            from evaluation.llm_judge import LLMJudge  # noqa: PLC0415
            return LLMJudge()
        # Default / DSPy placeholder
        logger.warning(
            "Unknown evaluation framework '%s'; falling back to LLM-as-Judge.",
            self._framework,
        )
        from evaluation.llm_judge import LLMJudge  # noqa: PLC0415
        return LLMJudge()

    def _compute_overall(self, result: EvaluationResult) -> float:
        """
        Compute a weighted average of individual metric scores.

        Args:
            result: Evaluation result with individual metric scores.

        Returns:
            Overall score in ``[0, 1]``.
        """
        scores = {
            "faithfulness": result.faithfulness,
            "answer_relevance": result.answer_relevance,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        weighted_sum = sum(
            scores[metric] * weight
            for metric, weight in self._WEIGHTS.items()
        )
        return round(weighted_sum, 4)
