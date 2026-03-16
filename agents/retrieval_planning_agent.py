"""
Retrieval Planning Agent.

Responsibilities
----------------
Given the :class:`~agents.task_planning_agent.TaskPlan`, this agent decides
**how** documents should be retrieved for each ``"retrieve"`` task by selecting
the optimal retrieval strategy:

- **Vector Search**  – dense semantic search via embedding similarity.
- **GraphRAG**       – knowledge-graph traversal (good for relationship queries).
- **Web Search**     – live internet search for recent / external knowledge.
- **Hybrid Search**  – combine vector + keyword (BM25) search.

Algorithm / Calling Relationship
----------------------------------
    TaskPlanningAgent
        └─▶ RetrievalPlanningAgent.execute(AgentInput)
                └─▶  _select_strategy(task, query_understanding) → RetrievalStrategy
                └─▶  returns AgentOutput(result=List[RetrievalStrategy])

The agent uses the intent, complexity, and task type from upstream agents to
make an informed selection, falling back to the global default if the LLM
is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from agents.task_planning_agent import Task, TaskPlan
from config.settings import RetrievalMode, settings
from utils.helpers import build_llm_client, get_model_name


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RetrievalStrategy:
    """
    Retrieval configuration for a single retrieve task.

    Attributes:
        task_id:       References the :class:`~agents.task_planning_agent.Task`.
        query:         The sub-query to retrieve documents for.
        mode:          Selected :class:`~config.settings.RetrievalMode`.
        top_k:         Number of candidate documents to fetch.
        filters:       Optional metadata filters (e.g. date range, source).
        rationale:     Why this strategy was chosen (for transparency / logging).
    """

    task_id: int
    query: str
    mode: RetrievalMode
    top_k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RetrievalPlanningAgent(BaseAgent):
    """
    Map each retrieve task in the :class:`~agents.task_planning_agent.TaskPlan`
    to a concrete :class:`RetrievalStrategy`.

    Selection heuristics
    --------------------
    +------------------+-------------------+
    | Intent           | Preferred mode    |
    +==================+===================+
    | factual          | vector            |
    | analytical       | hybrid            |
    | creative         | vector            |
    | procedural       | hybrid            |
    | conversational   | vector            |
    +------------------+-------------------+

    Additionally:
    - Recent-events keywords → ``web``
    - Relationship / graph keywords → ``graph``
    - Complex multi-hop → ``hybrid``
    """

    _RELATIONSHIP_KEYWORDS = {
        "relationship", "related", "connected", "link", "graph",
        "dependency", "hierarchy", "ancestor", "parent", "child",
    }
    _RECENCY_KEYWORDS = {
        "latest", "recent", "today", "current", "news",
        "2024", "2025", "2026", "yesterday", "this week",
    }

    def __init__(self) -> None:
        super().__init__("RetrievalPlanningAgent")
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Produce a list of :class:`RetrievalStrategy` objects – one per
        retrieve task in the incoming :class:`TaskPlan`.

        Args:
            agent_input: ``metadata["task_plan"]`` should be a serialised
                         :class:`TaskPlan` dict; ``metadata["query_understanding"]``
                         provides intent / complexity context.

        Returns:
            :class:`AgentOutput` whose ``result`` is
            ``List[RetrievalStrategy]``.
        """
        plan_dict: Dict[str, Any] = agent_input.metadata.get("task_plan", {})
        qu_dict: Dict[str, Any] = agent_input.metadata.get("query_understanding", {})

        # Reconstruct TaskPlan
        if plan_dict:
            tasks = [Task(**t) for t in plan_dict.get("tasks", [])]
            plan = TaskPlan(
                tasks=tasks,
                requires_multi_hop=plan_dict.get("requires_multi_hop", False),
                requires_tool_use=plan_dict.get("requires_tool_use", False),
            )
        else:
            # Fallback: single retrieve task
            plan = TaskPlan(
                tasks=[
                    Task(
                        task_id=1,
                        task_type="retrieve",
                        description="Retrieve documents",
                        query=agent_input.query,
                    )
                ]
            )

        intent: str = qu_dict.get("intent", "factual")
        complexity: str = qu_dict.get("complexity", "simple")

        strategies: List[RetrievalStrategy] = []
        for task in plan.tasks:
            if task.task_type == "retrieve":
                strategy = self._select_strategy(
                    task, intent, complexity, plan.requires_multi_hop
                )
                strategies.append(strategy)

        self.logger.debug(
            "Retrieval strategies: %s",
            [(s.task_id, s.mode.value) for s in strategies],
        )

        return AgentOutput(
            result=strategies,
            metadata={
                "strategies": [asdict(s) for s in strategies],
                "num_strategies": len(strategies),
            },
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_strategy(
        self,
        task: Task,
        intent: str,
        complexity: str,
        requires_multi_hop: bool,
    ) -> RetrievalStrategy:
        """
        Heuristically choose the best :class:`RetrievalMode` for a task.

        Decision order
        --------------
        1. If query contains recency keywords → ``web``.
        2. If query contains relationship keywords → ``graph``.
        3. If multi-hop or complex → ``hybrid``.
        4. Use intent-based mapping.
        5. Fall back to the global ``settings.default_retrieval_mode``.

        Args:
            task:              The retrieve task.
            intent:            Query intent from Query Understanding Agent.
            complexity:        Estimated complexity.
            requires_multi_hop: Whether the overall plan involves multi-hop.

        Returns:
            :class:`RetrievalStrategy` for the task.
        """
        query_lower = task.query.lower()

        # Rule-based selection
        if any(kw in query_lower for kw in self._RECENCY_KEYWORDS):
            mode = RetrievalMode.WEB
            rationale = "Query contains recency keywords – using web search."
        elif any(kw in query_lower for kw in self._RELATIONSHIP_KEYWORDS):
            mode = RetrievalMode.GRAPH
            rationale = "Query involves relationships – using GraphRAG."
        elif requires_multi_hop or complexity == "complex":
            mode = RetrievalMode.HYBRID
            rationale = "Complex multi-hop query – using hybrid search."
        else:
            intent_map: Dict[str, RetrievalMode] = {
                "factual": RetrievalMode.VECTOR,
                "analytical": RetrievalMode.HYBRID,
                "creative": RetrievalMode.VECTOR,
                "procedural": RetrievalMode.HYBRID,
                "conversational": RetrievalMode.VECTOR,
            }
            mode = intent_map.get(intent, settings.default_retrieval_mode)
            rationale = f"Intent '{intent}' mapped to {mode.value}."

        return RetrievalStrategy(
            task_id=task.task_id,
            query=task.query,
            mode=mode,
            top_k=settings.retrieval_top_k,
            rationale=rationale,
        )
