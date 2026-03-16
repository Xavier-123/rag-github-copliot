"""
Base agent abstraction for the Agentic RAG system.

All concrete agents inherit from :class:`BaseAgent` and must implement the
:meth:`BaseAgent.run` method.  This design enforces a uniform interface while
allowing each agent to encapsulate its own logic independently.

Design goals
------------
- **Decoupled**: agents communicate via plain :class:`AgentInput` /
  :class:`AgentOutput` data classes; no agent holds a direct reference to
  another.
- **Observable**: every agent emits structured log messages for monitoring.
- **Extendable**: override ``run`` to swap in different strategies without
  touching upstream callers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.helpers import get_logger


# ---------------------------------------------------------------------------
# Data classes shared across all agents
# ---------------------------------------------------------------------------

@dataclass
class AgentInput:
    """
    Standardised input payload passed into every agent.

    Attributes:
        query:          The original (or rewritten) user query string.
        context:        Retrieved / processed document chunks available so far.
        conversation_history: Prior turns in the current conversation session.
        metadata:       Arbitrary extra data (e.g. task plan, retrieval config).
        session_id:     Unique identifier for the current conversation session.
    """

    query: str
    context: List[Dict[str, Any]] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""


@dataclass
class AgentOutput:
    """
    Standardised output payload returned by every agent.

    Attributes:
        result:       The primary output (answer, plan, ranked docs, …).
        metadata:     Supplementary structured data produced by the agent.
        confidence:   Self-assessed confidence score in ``[0, 1]``.
        agent_name:   Name of the agent that produced this output.
        elapsed_ms:   Wall-clock execution time in milliseconds.
        error:        Non-empty string if the agent encountered a recoverable error.
    """

    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    agent_name: str = ""
    elapsed_ms: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base class for all RAG agents.

    Subclasses must implement :meth:`run`.  The public entry-point is
    :meth:`execute`, which adds timing and error handling around ``run``.

    Attributes:
        name:    Human-readable agent name (used in logs and output metadata).
        logger:  Pre-configured :class:`logging.Logger` instance.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name or self.__class__.__name__
        self.logger = get_logger(self.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """
        Execute the agent with timing and error capture.

        This wrapper calls :meth:`run` and ensures that:
        - ``agent_name`` is set on the output.
        - ``elapsed_ms`` reflects actual wall-clock duration.
        - Unhandled exceptions are caught, logged, and surfaced via the
          ``error`` field rather than propagating to the caller.

        Args:
            agent_input: Standardised input payload.

        Returns:
            :class:`AgentOutput` with all metadata fields populated.
        """
        self.logger.info("Starting | session=%s | query=%.80s", agent_input.session_id, agent_input.query)
        start = time.perf_counter()
        try:
            output = self.run(agent_input)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.logger.exception("Agent %s raised an exception: %s", self.name, exc)
            output = AgentOutput(
                result=None,
                error=str(exc),
                agent_name=self.name,
                elapsed_ms=elapsed_ms,
                confidence=0.0,
            )
        else:
            output.elapsed_ms = (time.perf_counter() - start) * 1000
            output.agent_name = self.name

        self.logger.info(
            "Done | elapsed=%.1fms | confidence=%.2f | error=%s",
            output.elapsed_ms,
            output.confidence,
            output.error or "none",
        )
        return output

    @abstractmethod
    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Core agent logic – to be implemented by every concrete subclass.

        Args:
            agent_input: Standardised input payload.

        Returns:
            :class:`AgentOutput` with the agent's results.
        """
        ...
