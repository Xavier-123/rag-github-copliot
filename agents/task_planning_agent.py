"""
Task Planning Agent.

Responsibilities
----------------
Given the :class:`~agents.query_understanding_agent.QueryUnderstanding`
produced by the previous stage, this agent decides **what** needs to be done
to answer the query.  It outputs a structured :class:`TaskPlan` that the
downstream Retrieval Planning Agent and Reasoning Agent execute.

Key decisions made here
-----------------------
- Whether pure retrieval is sufficient, or whether tool use / multi-step
  reasoning is required.
- How many retrieval rounds are needed (single vs. iterative).
- Which specialised sub-tasks to schedule (summarisation, comparison, …).

Algorithm / Calling Relationship
---------------------------------
    QueryUnderstandingAgent
        └─▶ TaskPlanningAgent.execute(AgentInput)
                └─▶  _plan_tasks(query_understanding) → List[Task]
                └─▶  returns AgentOutput(result=TaskPlan)

The task list is represented as an ordered sequence of :class:`Task` objects
that can be serialised to JSON for logging and replay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from agents.query_understanding_agent import QueryUnderstanding
from utils.helpers import build_llm_client, get_model_name, is_llm_available


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A single atomic task in the execution plan.

    Attributes:
        task_id:     Sequential identifier within the plan.
        task_type:   Category – ``"retrieve"``, ``"reason"``, ``"summarise"``,
                     ``"compare"``, ``"tool_call"``, or ``"generate_answer"``.
        description: Human-readable description of what this task does.
        query:       The (sub-)query this task should address.
        depends_on:  List of ``task_id`` values that must complete first.
        parameters:  Extra task-specific configuration.
    """

    task_id: int
    task_type: str
    description: str
    query: str
    depends_on: List[int] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """
    Ordered sequence of tasks returned by :class:`TaskPlanningAgent`.

    Attributes:
        tasks:              Ordered list of :class:`Task` objects.
        requires_multi_hop: Whether multi-hop retrieval is needed.
        requires_tool_use:  Whether external tool calls are needed.
    """

    tasks: List[Task]
    requires_multi_hop: bool = False
    requires_tool_use: bool = False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TaskPlanningAgent(BaseAgent):
    """
    Produce an ordered :class:`TaskPlan` from structured query understanding.

    The plan drives the rest of the pipeline: the Retrieval Planning Agent
    selects backends for each ``"retrieve"`` task; the Reasoning Agent
    executes ``"reason"`` tasks; and the Answer Generation step handles the
    final ``"generate_answer"`` task.
    """

    def __init__(self) -> None:
        super().__init__("TaskPlanningAgent")
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Build a task plan from query understanding metadata.

        Args:
            agent_input: ``metadata`` should contain a serialised
                         :class:`QueryUnderstanding` dict under the key
                         ``"query_understanding"``.

        Returns:
            :class:`AgentOutput` whose ``result`` is a :class:`TaskPlan`.
        """
        qu_dict: Dict[str, Any] = agent_input.metadata.get("query_understanding", {})
        # Reconstruct QueryUnderstanding if passed as dict
        if isinstance(qu_dict, dict) and qu_dict:
            qu = QueryUnderstanding(**qu_dict)
        else:
            # Fallback: minimal understanding from the raw query
            qu = QueryUnderstanding(
                original_query=agent_input.query,
                rewritten_query=agent_input.query,
                intent="factual",
                complexity="simple",
                sub_queries=[agent_input.query],
            )

        plan = self._build_plan(qu)
        self.logger.debug("TaskPlan: tasks=%d, multi_hop=%s", len(plan.tasks), plan.requires_multi_hop)

        return AgentOutput(
            result=plan,
            metadata={
                "num_tasks": len(plan.tasks),
                "requires_multi_hop": plan.requires_multi_hop,
                "requires_tool_use": plan.requires_tool_use,
                "tasks": [asdict(t) for t in plan.tasks],
            },
            confidence=0.85,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm_client()
        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if not is_llm_available():
            return "__MOCK__"
        client = self._get_llm()
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _build_plan(self, qu: QueryUnderstanding) -> TaskPlan:
        """
        Construct a :class:`TaskPlan` based on query complexity.

        Strategy
        --------
        - **simple**:   1 retrieve task → 1 generate_answer task.
        - **moderate**: N retrieve tasks (one per sub-query) → 1 reason task
                        → 1 generate_answer task.
        - **complex**:  N iterative retrieve tasks → N reason tasks →
                        1 summarise task → 1 generate_answer task.

        For complex queries the LLM is consulted to refine the plan; for
        simpler ones the heuristic above is applied deterministically.

        Args:
            qu: Structured query understanding.

        Returns:
            :class:`TaskPlan` ready for downstream agents.
        """
        if qu.complexity == "simple":
            return self._simple_plan(qu)
        if qu.complexity == "moderate":
            return self._moderate_plan(qu)
        return self._complex_plan(qu)

    def _simple_plan(self, qu: QueryUnderstanding) -> TaskPlan:
        tasks = [
            Task(
                task_id=1,
                task_type="retrieve",
                description=f"Retrieve documents for: {qu.rewritten_query}",
                query=qu.rewritten_query,
            ),
            Task(
                task_id=2,
                task_type="generate_answer",
                description="Generate the final answer using retrieved context.",
                query=qu.rewritten_query,
                depends_on=[1],
            ),
        ]
        return TaskPlan(tasks=tasks, requires_multi_hop=False, requires_tool_use=False)

    def _moderate_plan(self, qu: QueryUnderstanding) -> TaskPlan:
        tasks: List[Task] = []
        retrieve_ids: List[int] = []
        for i, sub_q in enumerate(qu.sub_queries, start=1):
            tasks.append(
                Task(
                    task_id=i,
                    task_type="retrieve",
                    description=f"Retrieve documents for sub-query: {sub_q}",
                    query=sub_q,
                )
            )
            retrieve_ids.append(i)

        reason_id = len(tasks) + 1
        tasks.append(
            Task(
                task_id=reason_id,
                task_type="reason",
                description="Synthesise evidence from all retrieved documents.",
                query=qu.rewritten_query,
                depends_on=retrieve_ids,
            )
        )
        tasks.append(
            Task(
                task_id=reason_id + 1,
                task_type="generate_answer",
                description="Generate the final answer.",
                query=qu.rewritten_query,
                depends_on=[reason_id],
            )
        )
        return TaskPlan(tasks=tasks, requires_multi_hop=True, requires_tool_use=False)

    def _complex_plan(self, qu: QueryUnderstanding) -> TaskPlan:
        """
        Build a complex multi-hop plan, optionally enriched by the LLM.

        For complex queries the LLM is asked to return a JSON task list.
        If that fails, we fall back to the deterministic moderate-plan strategy
        plus an extra summarisation step.
        """
        # Try LLM-driven planning
        system_prompt = (
            "You are a task planning assistant for a RAG system. "
            "Given a complex query and its sub-questions, produce an ordered "
            "JSON array of tasks. Each task must have: "
            "task_id (int), task_type (one of: retrieve, reason, summarise, "
            "compare, tool_call, generate_answer), description (str), "
            "query (str), depends_on (list of int). "
            "The last task must always be generate_answer."
        )
        user_prompt = (
            f"Original query: {qu.original_query}\n"
            f"Rewritten query: {qu.rewritten_query}\n"
            f"Sub-questions: {json.dumps(qu.sub_queries)}\n\n"
            "Return the task list as a JSON array."
        )
        raw = self._call_llm(system_prompt, user_prompt)

        if raw != "__MOCK__":
            try:
                start = raw.index("[")
                end = raw.rindex("]") + 1
                task_dicts = json.loads(raw[start:end])
                tasks = [Task(**td) for td in task_dicts]
                return TaskPlan(tasks=tasks, requires_multi_hop=True, requires_tool_use=True)
            except Exception:  # noqa: BLE001
                self.logger.warning("Could not parse LLM task plan; using fallback.")

        # Deterministic fallback
        base_plan = self._moderate_plan(qu)
        # Insert a summarise step before generate_answer
        gen_task = base_plan.tasks[-1]
        reason_task = base_plan.tasks[-2]
        summarise_id = reason_task.task_id + 1
        gen_task.task_id = summarise_id + 1
        gen_task.depends_on = [summarise_id]
        base_plan.tasks.insert(
            -1,
            Task(
                task_id=summarise_id,
                task_type="summarise",
                description="Summarise and de-duplicate all evidence before answer generation.",
                query=qu.rewritten_query,
                depends_on=[reason_task.task_id],
            ),
        )
        base_plan.requires_tool_use = True
        return base_plan
