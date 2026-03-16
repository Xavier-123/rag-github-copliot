"""
RAGAS Evaluator.

Wraps the `ragas` library to evaluate RAG pipelines using its standard
metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall.

This evaluator requires the ``ragas`` package:
    pip install ragas

If ``ragas`` is not installed, this class falls back to :class:`LLMJudge`
with a warning.

Reference
---------
- Es et al., 2023 – "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  https://arxiv.org/abs/2309.15217
"""

from __future__ import annotations

from typing import Any, List, Optional

from config.settings import settings
from evaluation.evaluator import EvaluationResult
from utils.helpers import get_logger


logger = get_logger("RagasEvaluator")


class RagasEvaluator:
    """
    RAGAS-based RAG evaluation.

    Attempts to use the ``ragas`` library; falls back to
    :class:`~evaluation.llm_judge.LLMJudge` if the library is not available.
    """

    def __init__(self) -> None:
        self._ragas_available = self._check_ragas()
        if not self._ragas_available:
            logger.warning(
                "ragas package not found; falling back to LLM-as-Judge. "
                "Install with: pip install ragas"
            )
            from evaluation.llm_judge import LLMJudge  # noqa: PLC0415
            self._fallback = LLMJudge()

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
        Run RAGAS evaluation metrics.

        Args:
            query:            User query.
            answer:           Generated answer.
            context:          List of retrieved context strings.
            reference_answer: Gold-standard answer (improves recall scoring).

        Returns:
            :class:`EvaluationResult` with metric scores.
        """
        if not self._ragas_available:
            return self._fallback.evaluate(query, answer, context, reference_answer)

        return self._run_ragas(query, answer, context, reference_answer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ragas() -> bool:
        """Check whether the ragas library is installed."""
        try:
            import ragas  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False

    def _run_ragas(
        self,
        query: str,
        answer: str,
        context: List[str],
        reference_answer: Optional[str],
    ) -> EvaluationResult:
        """
        Execute RAGAS evaluation using its dataset-based API.

        RAGAS expects a HuggingFace ``Dataset`` object with columns:
        ``question``, ``answer``, ``contexts``, ``ground_truth``.

        Args:
            query:            User query.
            answer:           Generated answer.
            context:          Retrieved context strings.
            reference_answer: Gold-standard answer.

        Returns:
            :class:`EvaluationResult` with RAGAS scores.
        """
        try:
            from datasets import Dataset  # noqa: PLC0415
            from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
            from ragas.metrics import (  # noqa: PLC0415
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [context],
                "ground_truth": [reference_answer or answer],
            }
            dataset = Dataset.from_dict(data)

            api_key = settings.openai_api_key or settings.azure_openai_api_key
            if not api_key:
                logger.warning("No API key; RAGAS will use mock scores.")
                return EvaluationResult(
                    faithfulness=0.7,
                    answer_relevance=0.7,
                    context_precision=0.7,
                    context_recall=0.7,
                )

            result = ragas_evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            scores = result.to_pandas().iloc[0].to_dict()

            return EvaluationResult(
                faithfulness=float(scores.get("faithfulness", 0)),
                answer_relevance=float(scores.get("answer_relevancy", 0)),
                context_precision=float(scores.get("context_precision", 0)),
                context_recall=float(scores.get("context_recall", 0)),
                details={"ragas_raw": scores},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("RAGAS evaluation failed: %s", exc)
            return EvaluationResult(error=str(exc))
