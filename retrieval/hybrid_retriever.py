"""
Hybrid Retriever.

Combines **dense vector search** with **sparse keyword (BM25) search** using
Reciprocal Rank Fusion (RRF) to merge the result lists.  This provides better
coverage than either method alone:

- Dense search captures semantic similarity.
- Sparse search captures exact keyword matches and rare terms.
- RRF is robust to score scale differences between the two methods.

Algorithm (RRF)
---------------
For each document d, the fused score is::

    RRF_score(d) = Σ_r  1 / (k + rank_r(d))

where ``k = 60`` (standard constant) and ``rank_r(d)`` is the position of
document ``d`` in result list ``r`` (1-indexed).

References
----------
- Cormack et al., 2009 – "Reciprocal Rank Fusion outperforms Condorcet and
  individual Rank Learning Methods"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from config.settings import settings
from retrieval.base_retriever import BaseRetriever
from retrieval.vector_retriever import VectorRetriever
from utils.helpers import deduplicate_documents


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense (vector) and sparse (BM25) search via RRF.

    The BM25 index is maintained in-memory using the ``rank_bm25`` library
    when available, falling back to simple TF-IDF scoring otherwise.
    """

    _RRF_K = 60  # Standard RRF constant

    def __init__(self) -> None:
        super().__init__("HybridRetriever")
        self._vector_retriever = VectorRetriever()
        self._bm25: Optional[Any] = None
        self._corpus: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # BaseRetriever implementation
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid (vector + BM25) search and RRF fusion.

        Args:
            query:   Natural-language search query.
            top_k:   Number of top fused results to return.
            filters: Passed through to the vector retriever.

        Returns:
            Fused, deduplicated list of top-K documents.
        """
        # Retrieve from both backends in parallel (could use threads; keeping
        # simple for clarity)
        k_wide = top_k * 2  # Retrieve more candidates for better fusion
        vector_docs = self._vector_retriever.retrieve(query, k_wide, filters)
        bm25_docs = self._bm25_retrieve(query, k_wide)

        fused = self._reciprocal_rank_fusion([vector_docs, bm25_docs], top_k)
        return fused

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to both the vector store and the BM25 index.

        Args:
            documents: List of document dicts.
        """
        self._vector_retriever.add_documents(documents)
        self._corpus.extend(documents)
        self._bm25 = None  # Reset BM25 so it's rebuilt on next query
        self.logger.info(
            "Added %d documents to hybrid index (corpus size: %d).",
            len(documents), len(self._corpus),
        )

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _get_bm25(self) -> Optional[Any]:
        """Lazily build the BM25 index from the current corpus."""
        if not self._corpus:
            return None
        if self._bm25 is not None:
            return self._bm25
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
            tokenised = [
                doc.get("content", "").lower().split()
                for doc in self._corpus
            ]
            self._bm25 = BM25Okapi(tokenised)
            self.logger.info("BM25 index built with %d documents.", len(self._corpus))
        except ImportError:
            self.logger.warning(
                "rank_bm25 not installed; BM25 retrieval unavailable. "
                "Install with: pip install rank-bm25"
            )
        return self._bm25

    def _bm25_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 keyword scoring.

        Args:
            query:  Search query.
            top_k:  Maximum number of results.

        Returns:
            List of document dicts sorted by BM25 score descending.
        """
        bm25 = self._get_bm25()
        if bm25 is None:
            return []

        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)

        # Pair score with document and sort
        scored = sorted(
            zip(scores, self._corpus),
            key=lambda x: x[0],
            reverse=True,
        )
        results = []
        for score, doc in scored[:top_k]:
            result_doc = dict(doc)
            result_doc["score"] = float(score)
            result_doc["retriever"] = "bm25"
            results.append(result_doc)
        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple ranked result lists using Reciprocal Rank Fusion (RRF).

        Each document is identified by its ``content`` field (first 100 chars)
        as a proxy for a document ID.

        Args:
            result_lists: A list of result lists (one per retriever).
            top_k:        Number of top fused results to return.

        Returns:
            Fused and sorted list of documents.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_registry: Dict[str, Dict[str, Any]] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                # Use content hash as document identifier
                key = doc.get("content", "")[:100]
                rrf_scores[key] += 1.0 / (self._RRF_K + rank)
                if key not in doc_registry:
                    doc_registry[key] = doc

        # Sort by fused RRF score descending
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        fused = []
        for key in sorted_keys[:top_k]:
            doc = dict(doc_registry[key])
            doc["rrf_score"] = round(rrf_scores[key], 6)
            doc["score"] = doc["rrf_score"]
            fused.append(doc)

        return deduplicate_documents(fused)
