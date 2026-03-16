"""
Knowledge Memory.

Manages the persistent knowledge base that backs vector, graph, and hybrid
retrieval.  This is the central document store for the system.

Responsibilities
----------------
- **Document ingestion**: accept raw documents, chunk them, embed, and store
  in configured backends.
- **Document retrieval**: delegate to the appropriate retriever.
- **Index management**: support adding, updating, and listing documents.

The Knowledge Memory acts as a unified facade over all retriever backends,
ensuring documents are indexed consistently across all stores.

Calling Relationship
--------------------
    RAGPipeline / external ingestor
        └─▶ KnowledgeMemory.add_documents(docs)
                ├─▶ VectorRetriever.add_documents(docs)
                └─▶ GraphRetriever.add_documents(docs)  (if graph enabled)
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from retrieval.graph_retriever import GraphRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever
from utils.helpers import get_logger


logger = get_logger("KnowledgeMemory")


class KnowledgeMemory:
    """
    Unified knowledge base facade.

    Maintains a registry of all ingested documents and coordinates indexing
    across vector and graph backends.
    """

    def __init__(
        self,
        enable_graph: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """
        Initialise the knowledge memory.

        Args:
            enable_graph:   Whether to also index documents in Neo4j.
            chunk_size:     Character length of each document chunk.
            chunk_overlap:  Overlap between consecutive chunks.
        """
        self._vector_retriever = VectorRetriever()
        self._hybrid_retriever = HybridRetriever()
        self._graph_retriever = GraphRetriever() if enable_graph else None
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # In-memory registry: doc_id → metadata
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._enable_graph = enable_graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Ingest documents into all configured knowledge stores.

        Each document is optionally chunked before indexing.

        Args:
            documents: List of dicts with at minimum a ``"content"`` key.
                       Optional keys: ``"source"``, ``"metadata"``, ``"id"``.

        Returns:
            Number of document chunks indexed.
        """
        chunks = self._chunk_documents(documents)
        if not chunks:
            return 0

        self._vector_retriever.add_documents(chunks)
        self._hybrid_retriever.add_documents(chunks)

        if self._enable_graph and self._graph_retriever:
            self._graph_retriever.add_documents(chunks)

        # Register chunks
        for chunk in chunks:
            doc_id = chunk.get("id", self._content_hash(chunk["content"]))
            self._registry[doc_id] = {
                "source": chunk.get("source", ""),
                "indexed_at": time.time(),
                "content_preview": chunk["content"][:100],
            }

        logger.info("Indexed %d chunks from %d source documents.", len(chunks), len(documents))
        return len(chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return a summary list of all indexed documents.

        Returns:
            List of registry entry dicts (id, source, indexed_at, preview).
        """
        return [
            {"id": doc_id, **meta}
            for doc_id, meta in self._registry.items()
        ]

    def document_count(self) -> int:
        """Return the total number of indexed document chunks."""
        return len(self._registry)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split documents into fixed-size overlapping chunks.

        Algorithm:
          - Slide a window of ``chunk_size`` characters across the content.
          - Advance by ``chunk_size - chunk_overlap`` characters each step.
          - Each chunk inherits the parent document's source and metadata.

        Args:
            documents: List of raw document dicts.

        Returns:
            List of chunk dicts.
        """
        chunks: List[Dict[str, Any]] = []
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "")
            metadata = doc.get("metadata", {})

            if len(content) <= self._chunk_size:
                chunks.append(doc)
                continue

            step = self._chunk_size - self._chunk_overlap
            for i, start in enumerate(range(0, len(content), step)):
                chunk_text = content[start: start + self._chunk_size]
                if not chunk_text.strip():
                    continue
                chunk_id = self._content_hash(chunk_text)
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,
                    "source": source,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "parent_source": source,
                    },
                })
        return chunks

    @staticmethod
    def _content_hash(content: str) -> str:
        """Generate a short stable ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
