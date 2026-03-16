"""
Query Understanding Agent.

Responsibilities
----------------
This is the **first** agent in the Agentic RAG pipeline.  Given a raw user
query it produces a structured understanding object that all downstream agents
can rely on.  Specifically it performs:

1. **Query Rewriting** – reformulate the query for better retrieval precision.
2. **Intent Detection** – classify the query (factual, analytical, creative, …).
3. **Complexity Estimation** – estimate whether multi-hop or single-hop retrieval
   is needed (simple / moderate / complex).
4. **Query Decomposition** – for complex queries, break them into independent
   sub-queries that can be retrieved and answered separately.

Algorithm / Calling Relationship
---------------------------------
    RAGPipeline
        └─▶ QueryUnderstandingAgent.execute(AgentInput)
                └─▶  _rewrite_query(query) → str
                └─▶  _detect_intent(query) → str
                └─▶  _estimate_complexity(query) → str
                └─▶  _decompose_query(query, complexity) → List[str]
                └─▶  returns AgentOutput(result=QueryUnderstanding)

The agent calls the LLM once per step using structured prompts; it falls back
to sensible defaults if the LLM is unavailable (mock mode).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from config.settings import settings
from utils.helpers import build_llm_client, get_model_name


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class QueryUnderstanding:
    """
    Structured result produced by :class:`QueryUnderstandingAgent`.

    Attributes:
        original_query:  The raw user query as received.
        rewritten_query: Reformulated query optimised for retrieval.
        intent:          Detected query intent label.
        complexity:      Estimated complexity – ``"simple"``, ``"moderate"``, or
                         ``"complex"``.
        sub_queries:     For complex queries, a list of atomic sub-questions.
    """

    original_query: str
    rewritten_query: str
    intent: str
    complexity: str
    sub_queries: List[str]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QueryUnderstandingAgent(BaseAgent):
    """
    Analyse a raw user query and return structured understanding metadata.

    This agent is the gateway into the pipeline; every subsequent agent
    uses its output to make informed decisions about planning and retrieval.
    """

    # Intent categories used in few-shot classification
    _INTENT_LABELS = [
        "factual",        # single verifiable fact
        "analytical",     # requires reasoning / comparison
        "creative",       # open-ended generation
        "procedural",     # how-to / step-by-step
        "conversational", # casual chat / follow-up
    ]

    def __init__(self) -> None:
        super().__init__("QueryUnderstandingAgent")
        # Lazy-initialised LLM client; None when running in mock mode.
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Full query understanding pipeline.

        Steps
        -----
        1. Rewrite the query for retrieval quality.
        2. Detect intent.
        3. Estimate complexity.
        4. Decompose if complex.

        Args:
            agent_input: Contains the raw ``query`` string.

        Returns:
            :class:`AgentOutput` whose ``result`` is a
            :class:`QueryUnderstanding` dataclass.
        """
        query = agent_input.query.strip()
        history = agent_input.conversation_history

        rewritten = self._rewrite_query(query, history)
        intent = self._detect_intent(rewritten)
        complexity = self._estimate_complexity(rewritten)
        sub_queries = self._decompose_query(rewritten, complexity)

        understanding = QueryUnderstanding(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            complexity=complexity,
            sub_queries=sub_queries,
        )

        self.logger.debug("QueryUnderstanding: %s", asdict(understanding))

        return AgentOutput(
            result=understanding,
            metadata=asdict(understanding),
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        """Lazily initialise the LLM client."""
        if self._llm is None:
            self._llm = build_llm_client()
        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the configured LLM and return the text response.

        Falls back to a clearly labelled mock string if the API key is absent.
        """
        api_key = (
            settings.openai_api_key or settings.azure_openai_api_key
        )
        if not api_key:
            self.logger.warning("No API key configured – using mock LLM response.")
            return "__MOCK__"

        client = self._get_llm()
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    # ---- Step 1 --------------------------------------------------------

    def _rewrite_query(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite the query to improve retrieval quality.

        Incorporates conversation history to resolve co-references
        (e.g. "tell me more about it") into explicit queries.

        Args:
            query:   Raw user query.
            history: Prior conversation turns (list of {role, content} dicts).

        Returns:
            Rewritten query string.
        """
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{t['role'].capitalize()}: {t['content']}"
                for t in history[-4:]  # last 2 turns
            )

        system_prompt = (
            "You are a query rewriting assistant. "
            "Given a conversation history and a new user query, "
            "rewrite the query to be self-contained, specific, and optimised "
            "for semantic document retrieval. "
            "Return ONLY the rewritten query, nothing else."
        )
        user_prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"User query: {query}\n\n"
            "Rewritten query:"
        )

        result = self._call_llm(system_prompt, user_prompt)
        if result == "__MOCK__" or not result:
            return query  # fallback: use original
        return result

    # ---- Step 2 --------------------------------------------------------

    def _detect_intent(self, query: str) -> str:
        """
        Classify the query into one of the predefined intent categories.

        Algorithm:
          - Send a few-shot classification prompt to the LLM.
          - Parse the single-word label returned.
          - Fall back to ``"factual"`` if parsing fails.

        Args:
            query: (Rewritten) user query.

        Returns:
            Intent label string.
        """
        labels_str = ", ".join(self._INTENT_LABELS)
        system_prompt = (
            f"Classify the user query into exactly one of these intent categories: "
            f"{labels_str}. "
            "Return ONLY the category name, lowercase, nothing else."
        )
        result = self._call_llm(system_prompt, query)
        if result == "__MOCK__":
            return "factual"
        label = result.lower().strip()
        return label if label in self._INTENT_LABELS else "factual"

    # ---- Step 3 --------------------------------------------------------

    def _estimate_complexity(self, query: str) -> str:
        """
        Estimate query complexity: ``"simple"``, ``"moderate"``, or ``"complex"``.

        Complexity guides the Task Planning Agent on how many retrieval steps
        and reasoning hops are required.

        - **simple**:   One-hop; a single document likely contains the answer.
        - **moderate**: Requires aggregation of 2–3 documents.
        - **complex**:  Multi-hop; requires decomposition, multiple retrievals,
                        and synthesis.

        Args:
            query: (Rewritten) user query.

        Returns:
            Complexity label string.
        """
        system_prompt = (
            "Estimate the retrieval complexity of the query. "
            "Return ONLY one of: simple, moderate, complex. "
            "- simple: a single document contains the answer. "
            "- moderate: 2-3 documents needed. "
            "- complex: multi-hop reasoning across many documents required."
        )
        result = self._call_llm(system_prompt, query)
        if result == "__MOCK__":
            return "simple"
        label = result.lower().strip()
        return label if label in {"simple", "moderate", "complex"} else "simple"

    # ---- Step 4 --------------------------------------------------------

    def _decompose_query(self, query: str, complexity: str) -> List[str]:
        """
        Decompose a complex query into independent atomic sub-questions.

        For ``"simple"`` and ``"moderate"`` queries the original query is
        returned as the sole sub-query.  For ``"complex"`` queries the LLM
        is asked to produce a JSON list of sub-questions.

        Algorithm:
          - Prompt the LLM for a JSON array of strings.
          - Parse the JSON; fall back to ``[query]`` on any error.

        Args:
            query:      (Rewritten) user query.
            complexity: Estimated complexity label.

        Returns:
            List of sub-query strings.
        """
        if complexity != "complex":
            return [query]

        system_prompt = (
            "You are a query decomposition assistant. "
            "Decompose the complex question into a list of simple, independent "
            "sub-questions that together answer the original question. "
            "Return a valid JSON array of strings, e.g. [\"q1\", \"q2\"]."
        )
        result = self._call_llm(system_prompt, query)
        if result == "__MOCK__":
            return [query]
        try:
            # Extract JSON array from the response (LLM may add prose around it)
            start = result.index("[")
            end = result.rindex("]") + 1
            sub_queries: List[str] = json.loads(result[start:end])
            return sub_queries if sub_queries else [query]
        except (ValueError, json.JSONDecodeError):
            self.logger.warning("Could not parse sub-queries JSON; using original query.")
            return [query]
