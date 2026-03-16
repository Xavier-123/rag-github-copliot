"""
Episodic Memory.

Stores and retrieves experiences gained by the system over time.  These
experiences include:

- Which **retrieval strategies** worked well for particular query types.
- Which **prompts** produced high-quality answers.
- **User feedback** signals (thumbs up/down, rating).
- **Evaluation scores** (RAGAS, LLM-as-Judge) associated with past queries.

The Self-Improving Layer reads from Episodic Memory to drive Feedback
Learning and System Optimization.

Data Model
----------
Each episode is a dict::

    {
        "episode_id":     str,
        "session_id":     str,
        "query":          str,
        "strategy_used":  str,           # e.g. "hybrid"
        "evaluation_score": float,       # 0-1
        "user_rating":    Optional[int], # 1-5 stars
        "metadata":       dict,
        "timestamp":      float,
    }

Calling Relationship
--------------------
    RAGPipeline (after Evaluation Layer)
        └─▶ EpisodicMemory.add_episode(episode)

    FeedbackLearning
        └─▶ EpisodicMemory.get_best_strategies(query_type, n)
        └─▶ EpisodicMemory.get_episodes(filters)
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from config.settings import settings
from utils.helpers import get_logger


logger = get_logger("EpisodicMemory")


class EpisodicMemory:
    """
    In-memory episodic memory with optional persistence to Redis.

    Episodes are stored in a bounded deque-like structure; when the maximum
    capacity is reached the oldest episodes are evicted.
    """

    def __init__(self) -> None:
        self._max_entries = settings.episodic_memory_max_entries
        self._episodes: List[Dict[str, Any]] = []
        self._redis: Optional[object] = None
        self._use_redis = self._init_redis()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_episode(
        self,
        session_id: str,
        query: str,
        strategy_used: str,
        evaluation_score: float,
        user_rating: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a new experience episode.

        Args:
            session_id:        Current session identifier.
            query:             The user query that was answered.
            strategy_used:     Retrieval strategy name (e.g. ``"hybrid"``).
            evaluation_score:  Automated quality score in ``[0, 1]``.
            user_rating:       Optional user-provided rating (1–5).
            metadata:          Arbitrary extra data.

        Returns:
            Unique episode ID.
        """
        episode_id = str(uuid.uuid4())
        episode: Dict[str, Any] = {
            "episode_id": episode_id,
            "session_id": session_id,
            "query": query,
            "strategy_used": strategy_used,
            "evaluation_score": evaluation_score,
            "user_rating": user_rating,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        if self._use_redis:
            self._redis_add(episode)
        else:
            if len(self._episodes) >= self._max_entries:
                self._episodes.pop(0)  # Evict oldest
            self._episodes.append(episode)

        logger.debug("Episode recorded: id=%s, score=%.2f", episode_id, evaluation_score)
        return episode_id

    def get_episodes(
        self,
        strategy: Optional[str] = None,
        min_score: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve episodes, optionally filtered by strategy and minimum score.

        Args:
            strategy:  Filter to episodes using this retrieval strategy.
            min_score: Minimum evaluation score threshold.
            limit:     Maximum number of episodes to return (most recent first).

        Returns:
            List of episode dicts sorted by timestamp descending.
        """
        episodes = self._episodes if not self._use_redis else self._redis_get_all()
        if strategy:
            episodes = [e for e in episodes if e.get("strategy_used") == strategy]
        if min_score is not None:
            episodes = [e for e in episodes if e.get("evaluation_score", 0) >= min_score]
        # Sort by timestamp descending
        episodes = sorted(episodes, key=lambda e: e.get("timestamp", 0), reverse=True)
        return episodes[:limit]

    def get_best_strategies(
        self, query_intent: Optional[str] = None, top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Return the top-N retrieval strategies by average evaluation score.

        Optionally filtered to episodes where the query intent matches.

        Args:
            query_intent: Optional intent label to filter by (uses metadata).
            top_n:        Number of top strategies to return.

        Returns:
            List of ``{"strategy": str, "avg_score": float, "count": int}``
            dicts sorted by average score descending.
        """
        episodes = self.get_episodes(limit=500)

        if query_intent:
            episodes = [
                e for e in episodes
                if e.get("metadata", {}).get("intent") == query_intent
            ]

        strategy_scores: Dict[str, List[float]] = defaultdict(list)
        for ep in episodes:
            strategy = ep.get("strategy_used", "unknown")
            score = ep.get("evaluation_score", 0.0)
            strategy_scores[strategy].append(score)

        ranked = [
            {
                "strategy": strategy,
                "avg_score": sum(scores) / len(scores),
                "count": len(scores),
            }
            for strategy, scores in strategy_scores.items()
        ]
        return sorted(ranked, key=lambda x: x["avg_score"], reverse=True)[:top_n]

    def add_user_rating(self, episode_id: str, rating: int) -> bool:
        """
        Attach a user rating to an existing episode.

        Args:
            episode_id: Target episode ID.
            rating:     Integer rating in 1–5.

        Returns:
            True if the episode was found and updated; False otherwise.
        """
        for ep in self._episodes:
            if ep["episode_id"] == episode_id:
                ep["user_rating"] = max(1, min(5, rating))
                return True
        return False

    def episode_count(self) -> int:
        """Return total number of stored episodes."""
        return len(self._episodes)

    # ------------------------------------------------------------------
    # Redis backend
    # ------------------------------------------------------------------

    def _init_redis(self) -> bool:
        """Attempt to connect to Redis; return True on success."""
        try:
            import json  # noqa: PLC0415
            import redis  # noqa: PLC0415
            client = redis.from_url(settings.redis_url, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            self._json = json
            logger.info("EpisodicMemory: using Redis backend.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("EpisodicMemory: Redis unavailable (%s); using in-memory.", exc)
            return False

    def _redis_add(self, episode: Dict[str, Any]) -> None:
        import json  # noqa: PLC0415
        key = "rag:episodes"
        self._redis.rpush(key, json.dumps(episode))  # type: ignore[union-attr]
        # Trim to max size
        self._redis.ltrim(key, -self._max_entries, -1)  # type: ignore[union-attr]

    def _redis_get_all(self) -> List[Dict[str, Any]]:
        import json  # noqa: PLC0415
        key = "rag:episodes"
        raw = self._redis.lrange(key, 0, -1)  # type: ignore[union-attr]
        return [json.loads(r) for r in raw]
