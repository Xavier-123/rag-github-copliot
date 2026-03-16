"""
Feedback Learning.

Closes the self-improvement loop by analysing evaluation results and
episodic memory to update system parameters:

1. **Retrieval Strategy** – if a particular strategy consistently scores
   poorly, prefer others for similar query intents.
2. **Prompt Tuning** – maintain a prompt registry and select the best-
   performing prompt variant for each task.
3. **Reranker Tuning** – adjust the ``rerank_top_k`` and score threshold
   based on observed precision/recall trade-offs.
4. **Tool Usage** – track which tool calls improve scores and promote them.

Algorithm
---------
    EvaluationLayer
        └─▶ FeedbackLearning.learn(evaluation_result, episode_context)
                └─▶ _analyse_retrieval_performance()
                └─▶ _update_retrieval_preference()
                └─▶ _update_prompt_preference()
                └─▶ returns LearningUpdate

The :class:`LearningUpdate` is applied by :class:`~optimization.system_optimizer.SystemOptimizer`.

Calling Relationship
--------------------
    RAGPipeline
        └─▶ FeedbackLearning.learn(eval_result, episode_ctx)
        └─▶ SystemOptimizer.apply(learning_update)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.evaluator import EvaluationResult
from memory.episodic_memory import EpisodicMemory
from utils.helpers import get_logger


logger = get_logger("FeedbackLearning")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class LearningUpdate:
    """
    Encapsulates recommended system parameter updates derived from feedback.

    Attributes:
        preferred_strategies: Mapping of query_intent → recommended strategy.
        discouraged_strategies: Strategies that performed poorly.
        prompt_updates:    Dict of prompt_key → improved prompt string.
        rerank_top_k:      Suggested new value for rerank_top_k (or None to keep).
        min_rerank_score:  Suggested new score filter threshold.
        notes:             Human-readable explanation of the updates.
    """

    preferred_strategies: Dict[str, str] = field(default_factory=dict)
    discouraged_strategies: List[str] = field(default_factory=list)
    prompt_updates: Dict[str, str] = field(default_factory=dict)
    rerank_top_k: Optional[int] = None
    min_rerank_score: Optional[float] = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FeedbackLearning
# ---------------------------------------------------------------------------

class FeedbackLearning:
    """
    Analyse evaluation results and episodic memory to produce learning updates.
    """

    # Minimum number of episodes before drawing conclusions
    _MIN_EPISODES = 5

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        """
        Initialise with a reference to the shared episodic memory store.

        Args:
            episodic_memory: Shared :class:`~memory.episodic_memory.EpisodicMemory`
                             instance containing past experience episodes.
        """
        from config.settings import settings  # noqa: PLC0415

        self._memory = episodic_memory
        # Use configurable threshold from settings (default 0.5)
        self._poor_score_threshold: float = settings.feedback_poor_score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(
        self,
        eval_result: EvaluationResult,
        episode_context: Optional[Dict[str, Any]] = None,
    ) -> LearningUpdate:
        """
        Produce a :class:`LearningUpdate` based on the latest evaluation and
        historical episodes.

        Args:
            eval_result:      The evaluation result for the most recent answer.
            episode_context:  Optional context about the episode (query, strategy, …).

        Returns:
            :class:`LearningUpdate` with recommended parameter changes.
        """
        update = LearningUpdate()
        ctx = episode_context or {}

        # Record the episode
        if ctx.get("session_id"):
            self._memory.add_episode(
                session_id=ctx["session_id"],
                query=ctx.get("query", ""),
                strategy_used=ctx.get("strategy_used", "unknown"),
                evaluation_score=eval_result.overall_score,
                metadata={"intent": ctx.get("intent", "")},
            )

        # Analyse historical performance
        self._update_strategy_preferences(update, ctx.get("intent", ""))
        self._update_rerank_params(update, eval_result)
        self._flag_poor_performance(update, eval_result)

        logger.info(
            "FeedbackLearning update: preferred=%s, discouraged=%s",
            update.preferred_strategies,
            update.discouraged_strategies,
        )
        return update

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_strategy_preferences(
        self, update: LearningUpdate, query_intent: str
    ) -> None:
        """
        Update preferred retrieval strategies based on episodic memory.

        Algorithm:
        - Retrieve the top-3 strategies from episodic memory.
        - If the best strategy has ≥ ``_MIN_EPISODES`` and a significantly
          higher average score, recommend it.

        Args:
            update:       The :class:`LearningUpdate` being built.
            query_intent: Current query intent label.
        """
        best_strategies = self._memory.get_best_strategies(
            query_intent=query_intent or None, top_n=3
        )
        if not best_strategies:
            return

        best = best_strategies[0]
        if best["count"] >= self._MIN_EPISODES and best["avg_score"] >= 0.7:
            if query_intent:
                update.preferred_strategies[query_intent] = best["strategy"]
            update.notes.append(
                f"Strategy '{best['strategy']}' has avg score {best['avg_score']:.2f} "
                f"over {best['count']} episodes."
            )

        # Flag poor strategies
        for strat_info in best_strategies:
            if (
                strat_info["count"] >= self._MIN_EPISODES
                and strat_info["avg_score"] < self._poor_score_threshold
            ):
                update.discouraged_strategies.append(strat_info["strategy"])
                update.notes.append(
                    f"Strategy '{strat_info['strategy']}' flagged as poor "
                    f"(avg={strat_info['avg_score']:.2f})."
                )

    def _update_rerank_params(
        self, update: LearningUpdate, eval_result: EvaluationResult
    ) -> None:
        """
        Suggest rerank parameter adjustments based on context precision/recall.

        - Low precision → reduce rerank_top_k (keep fewer, better docs).
        - Low recall → increase rerank_top_k (retrieve more docs).

        Args:
            update:      The :class:`LearningUpdate` being built.
            eval_result: Latest evaluation result.
        """
        from config.settings import settings as s  # noqa: PLC0415

        if eval_result.context_precision < 0.4:
            new_k = max(3, s.rerank_top_k - 1)
            update.rerank_top_k = new_k
            update.notes.append(
                f"Low context precision ({eval_result.context_precision:.2f}); "
                f"reducing rerank_top_k to {new_k}."
            )
        elif eval_result.context_recall < 0.4:
            new_k = min(15, s.rerank_top_k + 2)
            update.rerank_top_k = new_k
            update.notes.append(
                f"Low context recall ({eval_result.context_recall:.2f}); "
                f"increasing rerank_top_k to {new_k}."
            )

    def _flag_poor_performance(
        self, update: LearningUpdate, eval_result: EvaluationResult
    ) -> None:
        """
        Add general notes if the overall evaluation score is poor.

        Args:
            update:      The :class:`LearningUpdate` being built.
            eval_result: Latest evaluation result.
        """
        if eval_result.overall_score < self._poor_score_threshold:
            update.notes.append(
                f"Poor overall score ({eval_result.overall_score:.2f}); "
                "consider switching retrieval strategy or reviewing prompt quality."
            )
        if eval_result.faithfulness < 0.4:
            update.notes.append(
                "Low faithfulness detected – possible hallucination. "
                "Consider tightening context filtering."
            )
