"""
Vector Retriever.

Supports multiple vector store backends via a unified interface:
- **ChromaDB** (default, local or server)
- **FAISS** (local, in-memory / disk-persisted)
- **Milvus** (remote server)
- **Weaviate** (remote server)

The backend is selected from ``settings.vector_store_type``.  Each backend
is initialised lazily so the retriever can be imported without requiring all
libraries to be installed.

Algorithm
---------
1. Embed the query using the configured embedding model.
2. Perform an approximate nearest-neighbour (ANN) search in the vector store.
3. Return top-K documents with similarity scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.settings import settings
from retrieval.base_retriever import BaseRetriever
from utils.helpers import build_embedding_client, get_embedding_model_name, is_llm_available


class VectorRetriever(BaseRetriever):
    """
    Dense semantic retrieval using embedding-based vector search.

    The retriever is stateless with respect to the HTTP connection; the
    backend client is initialised on the first call to :meth:`retrieve`.
    """

    def __init__(self) -> None:
        super().__init__("VectorRetriever")
        self._store_type = settings.vector_store_type.lower()
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embed_client: Optional[Any] = None

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
        Retrieve semantically similar documents for *query*.

        Args:
            query:   Natural-language search query.
            top_k:   Number of documents to retrieve.
            filters: Metadata filters (supported by Chroma and Milvus).

        Returns:
            List of document dicts sorted by similarity (highest first).
        """
        query_embedding = self._embed(query)
        backend = self._get_backend()
        return backend(query, query_embedding, top_k, filters or {})

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.

        Each document's content is embedded and stored alongside its metadata.

        Args:
            documents: List of dicts with ``"content"`` key.
        """
        if self._store_type == "chroma":
            self._add_to_chroma(documents)
        elif self._store_type == "faiss":
            self._add_to_faiss(documents)
        else:
            self.logger.warning(
                "add_documents not fully implemented for backend '%s'.",
                self._store_type,
            )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for *text*.

        Falls back to a zero vector if the API key is not configured.

        Args:
            text: Input text to embed.

        Returns:
            Dense embedding vector.
        """
        if not is_llm_available():
            self.logger.warning("No LLM configured – returning mock zero embedding.")
            return [0.0] * 1536  # text-embedding-3-small dimension

        if self._embed_client is None:
            self._embed_client = build_embedding_client()

        response = self._embed_client.embeddings.create(
            input=text, model=get_embedding_model_name()
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _get_backend(self):
        """Return the retrieve function for the configured backend."""
        dispatch = {
            "chroma": self._retrieve_from_chroma,
            "faiss": self._retrieve_from_faiss,
            "milvus": self._retrieve_from_milvus,
            "weaviate": self._retrieve_from_weaviate,
        }
        backend = dispatch.get(self._store_type)
        if backend is None:
            self.logger.error("Unknown vector store type: %s", self._store_type)
            return self._mock_retrieve
        return backend

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------

    def _get_chroma_collection(self) -> Any:
        """Lazily initialise the ChromaDB client and collection."""
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # noqa: PLC0415
            client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
            self._collection = client.get_or_create_collection(
                settings.chroma_collection
            )
            self.logger.info("Connected to ChromaDB at %s:%d", settings.chroma_host, settings.chroma_port)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("ChromaDB unavailable (%s); using in-memory mock.", exc)
            self._collection = _InMemoryMockStore()
        return self._collection

    def _retrieve_from_chroma(
        self,
        query: str,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        collection = self._get_chroma_collection()
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=filters if filters else None,
            )
            return _chroma_results_to_docs(results)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("ChromaDB query failed (%s).", exc)
            return []

    def _add_to_chroma(self, documents: List[Dict[str, Any]]) -> None:
        collection = self._get_chroma_collection()
        ids, embeddings, metadatas, contents = [], [], [], []
        for i, doc in enumerate(documents):
            ids.append(doc.get("id", f"doc_{i}"))
            embeddings.append(self._embed(doc["content"]))
            contents.append(doc["content"])
            metadatas.append(doc.get("metadata", {"source": doc.get("source", "")}))
        collection.upsert(
            ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas
        )

    # ------------------------------------------------------------------
    # FAISS
    # ------------------------------------------------------------------

    def _retrieve_from_faiss(
        self,
        query: str,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        FAISS retrieval – mock implementation.

        Replace with a real ``faiss.IndexFlatIP`` / ``IndexIVFFlat`` index
        backed by a docstore for production use.
        """
        self.logger.info("FAISS retrieval (mock) for query: %.60s", query)
        return _mock_documents(query, top_k)

    def _add_to_faiss(self, documents: List[Dict[str, Any]]) -> None:
        self.logger.info("FAISS add_documents (mock): %d docs", len(documents))

    # ------------------------------------------------------------------
    # Milvus
    # ------------------------------------------------------------------

    def _retrieve_from_milvus(
        self,
        query: str,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Milvus retrieval – mock implementation.

        Replace with ``pymilvus.Collection.search()`` for production use.
        """
        self.logger.info("Milvus retrieval (mock) for query: %.60s", query)
        return _mock_documents(query, top_k)

    # ------------------------------------------------------------------
    # Weaviate
    # ------------------------------------------------------------------

    def _retrieve_from_weaviate(
        self,
        query: str,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Weaviate retrieval – mock implementation.

        Replace with ``weaviate.Client.query.get().with_near_vector()`` for
        production use.
        """
        self.logger.info("Weaviate retrieval (mock) for query: %.60s", query)
        return _mock_documents(query, top_k)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_retrieve(
        query: str,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return _mock_documents(query, top_k)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chroma_results_to_docs(results: Any) -> List[Dict[str, Any]]:
    """Convert ChromaDB query results to the unified document format."""
    docs = []
    if not results or not results.get("documents"):
        return docs
    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for content, meta, dist in zip(documents, metadatas, distances):
        docs.append({
            "content": content,
            "source": meta.get("source", "") if meta else "",
            "score": 1 - dist,  # Convert distance to similarity
            "metadata": meta or {},
        })
    return docs


def _mock_documents(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Return placeholder documents for testing without a real backend."""
    return [
        {
            "content": f"[Mock document {i+1}] Relevant information about: {query}",
            "source": f"mock://document_{i+1}",
            "score": 1.0 - i * 0.05,
            "metadata": {"mock": True},
        }
        for i in range(min(top_k, 3))
    ]


class _InMemoryMockStore:
    """Minimal in-memory store used when ChromaDB is unavailable."""

    def __init__(self) -> None:
        self._docs: List[Dict] = []

    def query(self, query_embeddings, n_results, where=None):
        docs = self._docs[:n_results]
        return {
            "documents": [[d["content"] for d in docs]],
            "metadatas": [[d.get("metadata", {}) for d in docs]],
            "distances": [[0.1 * i for i in range(len(docs))]],
        }

    def upsert(self, ids, embeddings, documents, metadatas):
        for doc_id, emb, content, meta in zip(ids, embeddings, documents, metadatas):
            self._docs.append({"id": doc_id, "content": content, "metadata": meta})
