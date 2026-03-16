"""
Multi-Retriever Agent.

Responsibilities
----------------
Execute multiple retrieval strategies **in parallel** and merge the results.
This agent acts as an orchestrator over the concrete retriever backends:

- :class:`~retrieval.vector_retriever.VectorRetriever`
- :class:`~retrieval.graph_retriever.GraphRetriever`
- :class:`~retrieval.web_retriever.WebRetriever`
- :class:`~retrieval.hybrid_retriever.HybridRetriever`

Algorithm / Calling Relationship
----------------------------------
    RetrievalPlanningAgent
        └─▶ MultiRetrieverAgent.execute(AgentInput)
                ├─▶  VectorRetriever.retrieve(query, top_k)
                ├─▶  GraphRetriever.retrieve(query, top_k)   (if needed)
                ├─▶  WebRetriever.retrieve(query, top_k)     (if needed)
                └─▶  HybridRetriever.retrieve(query, top_k)  (if needed)
                └─▶  _merge_results(all_results) → List[Document]
                └─▶  returns AgentOutput(result=List[Document])

Parallel execution is achieved via :mod:`concurrent.futures.ThreadPoolExecutor`
so retriever latencies do not stack.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from agents.retrieval_planning_agent import RetrievalStrategy
from config.settings import RetrievalMode
from retrieval.graph_retriever import GraphRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.web_retriever import WebRetriever
from utils.helpers import deduplicate_documents


class MultiRetrieverAgent(BaseAgent):
    """
    Orchestrate parallel retrieval across all configured backends.

    Each :class:`~agents.retrieval_planning_agent.RetrievalStrategy` is
    dispatched to the corresponding concrete retriever.  Results are merged
    and deduplicated before being passed to the Rerank Agent.
    """

    def __init__(self) -> None:
        super().__init__("MultiRetrieverAgent")
        # Instantiate retrievers lazily (they may not be needed on every call)
        self._vector_retriever = VectorRetriever()
        self._graph_retriever = GraphRetriever()
        self._web_retriever = WebRetriever()
        self._hybrid_retriever = HybridRetriever()

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Execute each retrieval strategy in parallel and return merged documents.

        Args:
            agent_input: ``metadata["retrieval_strategies"]`` should be a list
                         of :class:`RetrievalStrategy` dicts.

        Returns:
            :class:`AgentOutput` whose ``result`` is a merged list of document
            dicts, each containing at minimum ``"content"`` and ``"source"``
            keys.
        """
        strategy_dicts: List[Dict[str, Any]] = agent_input.metadata.get(
            "retrieval_strategies", []
        )

        # Reconstruct strategy objects
        if strategy_dicts:
            strategies = [RetrievalStrategy(**s) for s in strategy_dicts]
        else:
            # Fallback: single vector strategy for the raw query
            strategies = [
                RetrievalStrategy(
                    task_id=1,
                    query=agent_input.query,
                    mode=RetrievalMode.VECTOR,
                    top_k=10,
                )
            ]

        all_documents: List[Dict[str, Any]] = []
        errors: List[str] = []

        # ----------------------------------------------------------------
        # Parallel retrieval using ThreadPoolExecutor
        # ----------------------------------------------------------------
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            future_to_strategy = {
                executor.submit(self._retrieve_one, strategy): strategy
                for strategy in strategies
            }
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    docs = future.result()
                    self.logger.debug(
                        "Strategy task_id=%d mode=%s returned %d docs",
                        strategy.task_id, strategy.mode.value, len(docs),
                    )
                    all_documents.extend(docs)
                except Exception as exc:  # noqa: BLE001
                    msg = f"Retriever {strategy.mode.value} failed: {exc}"
                    self.logger.warning(msg)
                    errors.append(msg)

        # Deduplicate across retrievers
        unique_docs = deduplicate_documents(all_documents)

        self.logger.info(
            "Retrieved %d unique documents from %d strategies.",
            len(unique_docs), len(strategies),
        )

        return AgentOutput(
            result=unique_docs,
            metadata={
                "total_retrieved": len(all_documents),
                "unique_retrieved": len(unique_docs),
                "errors": errors,
                "strategies_used": [s.mode.value for s in strategies],
            },
            confidence=1.0 if not errors else 0.7,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_one(self, strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
        """
        Dispatch a single :class:`RetrievalStrategy` to the appropriate retriever.

        Args:
            strategy: The retrieval configuration to execute.

        Returns:
            List of document dicts from the chosen backend.
        """
        retriever_map = {
            RetrievalMode.VECTOR: self._vector_retriever,
            RetrievalMode.GRAPH: self._graph_retriever,
            RetrievalMode.WEB: self._web_retriever,
            RetrievalMode.HYBRID: self._hybrid_retriever,
        }
        retriever = retriever_map[strategy.mode]
        return retriever.retrieve(
            query=strategy.query,
            top_k=strategy.top_k,
            filters=strategy.filters,
        )
