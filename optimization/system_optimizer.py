"""
System Optimizer.

Applies the :class:`~optimization.feedback_learning.LearningUpdate` produced
by the Feedback Learning module to the live system configuration, closing the
self-improvement loop.

Optimisation targets
--------------------
- **Prompt Optimization**  – swap in better-performing prompt variants.
- **Retriever Tuning**     – update ``rerank_top_k`` and retrieval preferences.
- **Reranker Tuning**      – adjust score thresholds.
- **Strategy Overrides**   – register query-intent → strategy mappings that
                             the Retrieval Planning Agent can look up.

Inspiration / Related Work
--------------------------
- **DSPy** (Khattab et al., 2023) – automated prompt optimisation.
- **AutoRAG** – automated RAG pipeline tuning.
- **Self-RAG** – adaptive retrieval and self-reflection.

This implementation provides the framework interface; specific ML-based tuning
steps (DSPy compilation, AutoRAG hyperparameter search) can be plugged in via
the extension points marked as ``# TODO: Extend with <technique>``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings
from optimization.feedback_learning import LearningUpdate
from utils.helpers import get_logger


logger = get_logger("SystemOptimizer")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """
    Record of system changes made during an optimisation cycle.

    Attributes:
        applied_changes: Human-readable list of changes applied.
        skipped_changes: Changes that were considered but skipped.
        timestamp:       Unix timestamp of the optimisation run.
        cycle_id:        Monotonically increasing optimisation cycle counter.
    """

    applied_changes: List[str] = field(default_factory=list)
    skipped_changes: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    cycle_id: int = 0


# ---------------------------------------------------------------------------
# SystemOptimizer
# ---------------------------------------------------------------------------

class SystemOptimizer:
    """
    Apply learned system parameter updates from :class:`LearningUpdate`.

    The optimizer maintains a mutable parameter store that overrides the
    static ``settings`` values.  Downstream agents query this store (via
    :meth:`get_strategy_override` and :meth:`get_rerank_top_k`) to use the
    dynamically optimised values.
    """

    def __init__(self) -> None:
        self._cycle_count: int = 0
        # Dynamic overrides applied by optimization
        self._strategy_overrides: Dict[str, str] = {}
        self._rerank_top_k: int = settings.rerank_top_k
        self._min_rerank_score: float = 2.0
        self._prompt_registry: Dict[str, str] = {}
        self._optimization_history: List[OptimizationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, update: LearningUpdate) -> OptimizationResult:
        """
        Apply a :class:`LearningUpdate` to the system configuration.

        Args:
            update: Recommended system parameter changes.

        Returns:
            :class:`OptimizationResult` detailing what was applied.
        """
        self._cycle_count += 1
        result = OptimizationResult(cycle_id=self._cycle_count)

        # Apply strategy overrides
        for intent, strategy in update.preferred_strategies.items():
            self._strategy_overrides[intent] = strategy
            result.applied_changes.append(
                f"Strategy override: intent='{intent}' → '{strategy}'"
            )

        # Apply rerank_top_k adjustment
        if update.rerank_top_k is not None:
            old_k = self._rerank_top_k
            self._rerank_top_k = update.rerank_top_k
            result.applied_changes.append(
                f"rerank_top_k: {old_k} → {self._rerank_top_k}"
            )

        # Apply min_rerank_score adjustment
        if update.min_rerank_score is not None:
            old_score = self._min_rerank_score
            self._min_rerank_score = update.min_rerank_score
            result.applied_changes.append(
                f"min_rerank_score: {old_score:.2f} → {self._min_rerank_score:.2f}"
            )

        # Apply prompt updates
        for key, prompt in update.prompt_updates.items():
            self._prompt_registry[key] = prompt
            result.applied_changes.append(f"Prompt updated: key='{key}'")

        # Log notes from the learning update
        for note in update.notes:
            logger.info("[OptimizationNote] %s", note)

        self._optimization_history.append(result)
        logger.info(
            "Optimization cycle %d: %d changes applied, %d skipped.",
            self._cycle_count, len(result.applied_changes), len(result.skipped_changes),
        )
        return result

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_strategy_override(self, query_intent: str) -> Optional[str]:
        """
        Return the dynamically optimised retrieval strategy for an intent.

        Args:
            query_intent: Query intent label (e.g. ``"factual"``).

        Returns:
            Preferred strategy name, or ``None`` if no override is set.
        """
        return self._strategy_overrides.get(query_intent)

    def get_rerank_top_k(self) -> int:
        """Return the current (possibly optimised) rerank_top_k value."""
        return self._rerank_top_k

    def get_min_rerank_score(self) -> float:
        """Return the current (possibly optimised) minimum rerank score."""
        return self._min_rerank_score

    def get_prompt(self, key: str, default: str = "") -> str:
        """
        Retrieve a prompt from the registry.

        Args:
            key:     Prompt identifier.
            default: Fallback prompt string if the key is not registered.

        Returns:
            Registered or default prompt string.
        """
        return self._prompt_registry.get(key, default)

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Return the full optimisation history."""
        return list(self._optimization_history)

    def get_status(self) -> Dict[str, Any]:
        """
        Return a status dict summarising the current optimisation state.

        Returns:
            Dict with current parameter values and cycle count.
        """
        return {
            "cycle_count": self._cycle_count,
            "strategy_overrides": dict(self._strategy_overrides),
            "rerank_top_k": self._rerank_top_k,
            "min_rerank_score": self._min_rerank_score,
            "registered_prompts": list(self._prompt_registry.keys()),
        }
