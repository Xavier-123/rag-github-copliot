"""
Context Engineering Agent.

Responsibilities
----------------
Transform the re-ranked document list into a clean, concise, and
well-structured context string that can be injected directly into the
reasoning / answer-generation prompt.

The following processing steps are applied in order:

1. **Deduplication** – identical or near-identical documents are removed.
2. **Filtering**     – documents with extremely low rerank scores are dropped.
3. **Compression**   – each document is summarised if it exceeds a length
                       threshold (uses the LLM).
4. **Summarisation** – when many documents exist, a single holistic summary
                       is generated (optional, triggered by ``summarise`` tasks).
5. **Context Ordering** – documents are sorted by relevance score then arranged
                          in a reading-friendly order for the LLM prompt.

Algorithm / Calling Relationship
----------------------------------
    RerankAgent
        └─▶ ContextEngineeringAgent.execute(AgentInput)
                └─▶  _deduplicate(docs)
                └─▶  _filter(docs)
                └─▶  _compress(docs, query)
                └─▶  _order(docs)
                └─▶  _format_context(docs) → str
                └─▶  returns AgentOutput(result=str)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from config.settings import settings
from utils.helpers import (
    build_llm_client,
    deduplicate_documents,
    format_documents_for_context,
    get_model_name,
    truncate_text,
)


class ContextEngineeringAgent(BaseAgent):
    """
    Post-process retrieved documents into a clean prompt context.

    The final context string is consumed by the Reasoning Agent and the
    Answer Generation step.
    """

    # Documents shorter than this are not compressed
    _COMPRESSION_THRESHOLD_CHARS = 1500
    # Minimum rerank score to keep a document (0–10 scale; -1 means keep all)
    _MIN_RERANK_SCORE = 2.0

    def __init__(self) -> None:
        super().__init__("ContextEngineeringAgent")
        self._llm: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseAgent.run implementation
    # ------------------------------------------------------------------

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Process and format document context for downstream agents.

        Args:
            agent_input: ``context`` contains the re-ranked document list.

        Returns:
            :class:`AgentOutput` whose ``result`` is a formatted context
            string ready for prompt injection.
        """
        documents = agent_input.context or agent_input.metadata.get("reranked_docs", [])
        query = agent_input.query

        if not documents:
            self.logger.warning("No documents in context; returning empty context.")
            return AgentOutput(
                result="",
                metadata={"num_context_docs": 0},
                confidence=0.3,
            )

        # Pipeline
        docs = deduplicate_documents(documents)
        docs = self._filter(docs)
        docs = self._compress(docs, query)
        docs = self._order(docs)
        context_str = format_documents_for_context(
            docs, max_chars=16000  # safe limit for most LLM context windows
        )

        self.logger.info(
            "Context engineering: %d docs → %d chars",
            len(docs), len(context_str),
        )

        return AgentOutput(
            result=context_str,
            metadata={
                "num_context_docs": len(docs),
                "context_chars": len(context_str),
            },
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm_client()
        return self._llm

    def _filter(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove documents whose ``rerank_score`` is below the minimum threshold.

        If no documents have a score field, all documents are kept.

        Args:
            documents: List of document dicts, optionally with ``rerank_score``.

        Returns:
            Filtered list.
        """
        has_scores = any("rerank_score" in d for d in documents)
        if not has_scores:
            return documents
        filtered = [
            d for d in documents
            if d.get("rerank_score", 0) >= self._MIN_RERANK_SCORE
        ]
        removed = len(documents) - len(filtered)
        if removed:
            self.logger.debug("Filtered %d low-score documents.", removed)
        return filtered or documents  # avoid returning empty list

    def _compress(
        self, documents: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Summarise documents that exceed the length threshold.

        Each long document is compressed to a focused summary relevant to the
        query.  This prevents long-context issues without losing key information.

        Args:
            documents: List of document dicts.
            query:     The user query (used to focus the summary).

        Returns:
            List of documents, some with compressed ``content``.
        """
        api_key = settings.openai_api_key or settings.azure_openai_api_key
        compressed = []
        for doc in documents:
            content = doc.get("content", "")
            if len(content) > self._COMPRESSION_THRESHOLD_CHARS and api_key:
                summary = self._summarise_document(content, query)
                doc = {**doc, "content": summary, "compressed": True}
            compressed.append(doc)
        return compressed

    def _summarise_document(self, content: str, query: str) -> str:
        """
        Use the LLM to produce a concise, query-focused summary of a document.

        Args:
            content: Full document text.
            query:   The user query to focus the summary.

        Returns:
            Summarised text string.
        """
        truncated = truncate_text(content, max_tokens=3000)
        try:
            client = self._get_llm()
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document compression assistant. "
                            "Summarise the following document in 3-5 sentences, "
                            "focusing on information relevant to the query. "
                            "Preserve key facts, numbers, and names."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nDocument:\n{truncated}",
                    },
                ],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Document compression failed (%s); using original.", exc)
            return truncate_text(content, max_tokens=500)

    def _order(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Order documents for optimal prompt performance.

        Research (Liu et al., 2023 "Lost in the Middle") shows that LLMs attend
        best to documents at the start and end of the context window.  We
        therefore place the highest-scoring document first and second-highest
        last, with remaining documents in the middle.

        Args:
            documents: List of document dicts.

        Returns:
            Re-ordered document list.
        """
        if len(documents) <= 2:
            return documents
        # Sort by rerank_score descending; no score → original order preserved
        sorted_docs = sorted(
            documents, key=lambda d: d.get("rerank_score", 0), reverse=True
        )
        if len(sorted_docs) <= 2:
            return sorted_docs
        # Place best at start, second-best at end, rest in middle
        best = sorted_docs[0]
        second_best = sorted_docs[1]
        middle = sorted_docs[2:]
        return [best] + middle + [second_best]
