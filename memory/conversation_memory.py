"""
Conversation Memory.

Maintains a sliding window of recent conversation turns for each session.

Two backends are supported:
1. **In-Memory** (default) – fast, no dependencies, but not persistent across
   process restarts.  Suitable for development and single-node deployments.
2. **Redis** – persistent, distributed, suitable for production.  Activated
   when ``settings.redis_url`` is reachable.

The conversation history is serialised as a JSON list of
``{"role": "...", "content": "..."}`` dicts (OpenAI message format).

Calling Relationship
--------------------
    RAGPipeline
        ├─▶ ConversationMemory.get_history(session_id) → List[Dict]
        └─▶ ConversationMemory.add_turn(session_id, role, content)
"""

from __future__ import annotations

import json
from collections import deque
from typing import Deque, Dict, List, Optional

from config.settings import settings
from utils.helpers import get_logger


logger = get_logger("ConversationMemory")


class ConversationMemory:
    """
    Sliding-window conversation memory with Redis and in-memory backends.

    The memory window is bounded by ``settings.conversation_window`` turns.
    Older turns are automatically evicted.
    """

    def __init__(self) -> None:
        self._window = settings.conversation_window
        # In-memory fallback: session_id → deque of message dicts
        self._store: Dict[str, Deque[Dict[str, str]]] = {}
        self._redis: Optional[object] = None
        self._use_redis = self._init_redis()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the recent conversation history for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            Ordered list of ``{"role": ..., "content": ...}`` dicts,
            from oldest to most recent, capped at the window size.
        """
        if self._use_redis:
            return self._redis_get(session_id)
        return list(self._store.get(session_id, deque()))

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """
        Append a single conversation turn to the session history.

        Args:
            session_id: Unique session identifier.
            role:       Speaker role – ``"user"`` or ``"assistant"``.
            content:    Message text.
        """
        message = {"role": role, "content": content}
        if self._use_redis:
            self._redis_add(session_id, message)
        else:
            if session_id not in self._store:
                self._store[session_id] = deque(maxlen=self._window)
            self._store[session_id].append(message)

    def clear(self, session_id: str) -> None:
        """
        Clear all history for a session.

        Args:
            session_id: Unique session identifier.
        """
        if self._use_redis:
            self._redis.delete(self._redis_key(session_id))  # type: ignore[union-attr]
        else:
            self._store.pop(session_id, None)

    def get_all_sessions(self) -> List[str]:
        """Return all active session IDs (in-memory backend only)."""
        return list(self._store.keys())

    # ------------------------------------------------------------------
    # Redis backend
    # ------------------------------------------------------------------

    def _init_redis(self) -> bool:
        """Attempt to connect to Redis; return True on success."""
        try:
            import redis  # noqa: PLC0415
            client = redis.from_url(settings.redis_url, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            logger.info("ConversationMemory: using Redis backend (%s).", settings.redis_url)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "ConversationMemory: Redis unavailable (%s); using in-memory backend.", exc
            )
            return False

    def _redis_key(self, session_id: str) -> str:
        return f"rag:conversation:{session_id}"

    def _redis_get(self, session_id: str) -> List[Dict[str, str]]:
        key = self._redis_key(session_id)
        raw_list = self._redis.lrange(key, -self._window, -1)  # type: ignore[union-attr]
        return [json.loads(item) for item in raw_list]

    def _redis_add(self, session_id: str, message: Dict[str, str]) -> None:
        key = self._redis_key(session_id)
        self._redis.rpush(key, json.dumps(message))  # type: ignore[union-attr]
        # Trim to window size
        self._redis.ltrim(key, -self._window, -1)  # type: ignore[union-attr]
        # Set TTL of 24 hours
        self._redis.expire(key, 86400)  # type: ignore[union-attr]
