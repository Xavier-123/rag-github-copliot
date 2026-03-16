"""
Rerank Agent.

Responsibilities
----------------
Given the raw candidate documents returned by the Multi-Retriever Agent,
this agent **re-scores and re-orders** them so that the most relevant
documents appear first.  It keeps only the top-K results.

Two re-ranking strategies are supported:
1. **LLM-based re-ranking** (default) – the LLM is asked to score each
   document's relevance to the query on a 0–10 scale; documents are then
   sorted by score descending.
2. **Cross-encoder / embedding re-ranking** (placeholder) – intended to be
   replaced with a dedicated cross-encoder model (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``)
   for production deployments.

Algorithm / Calling Relationship
----------------------------------
    MultiRetrieverAgent
        └─▶ RerankAgent.execute(AgentInput)
                └─▶  _score_documents(query, docs) → List[(score, doc)]
                └─▶  sorted(by score, desc)[:top_k]
                └─▶  returns AgentOutput(result=List[Document])
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from config.settings import settings
from utils.helpers import build_llm_client, get_model_name, is_llm_available


class RerankAgent(BaseAgent):
    """
    Score and re-rank retrieved documents by relevance.

    The top ``settings.rerank_top_k`` documents are forwarded to the
    Context Engineering Agent.
    """

    def __init__(self) -> None:
        super().__init__("RerankAgent")
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Re-rank candidate documents and return the top-K.

        Args:
            agent_input: ``context`` contains the candidate document list;
                         ``query`` is the (rewritten) user query.

        Returns:
            :class:`AgentOutput` whose ``result`` is a list of the top-K
            re-ranked documents, each augmented with a ``"rerank_score"``
            field.
        """
        query = agent_input.query
        documents = agent_input.context or agent_input.metadata.get("retrieved_docs", [])

        if not documents:
            self.logger.warning("No documents to re-rank; returning empty list.")
            return AgentOutput(result=[], metadata={"reranked": 0}, confidence=0.5)

        scored = self._score_documents(query, documents)
        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        top_k = settings.rerank_top_k
        top_docs = [doc for _, doc in scored[:top_k]]

        self.logger.info(
            "Re-ranked %d → %d documents. Top score: %.2f",
            len(documents), len(top_docs),
            scored[0][0] if scored else 0,
        )

        return AgentOutput(
            result=top_docs,
            metadata={
                "reranked": len(top_docs),
                "scores": [score for score, _ in scored[:top_k]],
            },
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm_client()
        return self._llm

    def _score_documents(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Assign a relevance score to each document.

        LLM-based scoring
        -----------------
        Send all documents to the LLM in a single batched prompt.  The LLM
        returns a JSON array of ``[index, score]`` pairs where score ∈ [0, 10].

        Fallback
        --------
        If the LLM is unavailable, documents retain their original order with
        decreasing mock scores (10, 9, 8, …).

        Args:
            query:     The user query to rank against.
            documents: List of candidate document dicts.

        Returns:
            List of ``(score, document)`` tuples.
        """
        if not is_llm_available():
            # Mock: assign descending scores to preserve original order
            return [(10.0 - i, doc) for i, doc in enumerate(documents)]

        # Build a compact representation of documents for the LLM
        doc_summaries = []
        for i, doc in enumerate(documents):
            content_snippet = doc.get("content", "")[:300]
            doc_summaries.append(f"[{i}]: {content_snippet}")

        system_prompt = (
            "You are a relevance scoring assistant. "
            "Given a query and a list of document snippets (each prefixed with [index]), "
            "return a JSON array where each element is [index, score] and score is "
            "an integer from 0 (irrelevant) to 10 (perfectly relevant). "
            "Return ONLY the JSON array, nothing else."
        )
        user_prompt = (
            f"Query: {query}\n\n"
            "Documents:\n" + "\n".join(doc_summaries)
        )

        try:
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
            raw = response.choices[0].message.content.strip()
            start = raw.index("[")
            end = raw.rindex("]") + 1
            score_pairs = json.loads(raw[start:end])

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for pair in score_pairs:
                idx, score = int(pair[0]), float(pair[1])
                if 0 <= idx < len(documents):
                    doc = dict(documents[idx])
                    doc["rerank_score"] = score
                    scored.append((score, doc))
            return scored

        except Exception as exc:  # noqa: BLE001
            self.logger.warning("LLM re-ranking failed (%s); using original order.", exc)
            return [(10.0 - i, doc) for i, doc in enumerate(documents)]
