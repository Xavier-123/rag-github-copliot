"""
Base retriever abstraction.

All concrete retrieval backends inherit from :class:`BaseRetriever` and must
implement :meth:`retrieve` and :meth:`add_documents`.

The unified :class:`Document` type ensures that all retrievers return data in
the same shape, making it easy for the Multi-Retriever Agent to merge results
regardless of the backend used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.helpers import get_logger


# ---------------------------------------------------------------------------
# Document type
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    Unified document representation used by all retrieval backends.

    Attributes:
        content:  The full text content of the document chunk.
        source:   Identifier of the source (file path, URL, node ID, …).
        score:    Relevance score assigned by the retriever (higher = better).
        metadata: Arbitrary extra metadata (page number, date, author, …).
    """

    content: str
    source: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary for downstream processing."""
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval backends.

    Subclasses must implement:
    - :meth:`retrieve`      – query the index and return matching documents.
    - :meth:`add_documents` – ingest new documents into the index.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-K documents most relevant to *query*.

        Args:
            query:   Natural-language search query.
            top_k:   Maximum number of documents to return.
            filters: Optional metadata filters (backend-specific).

        Returns:
            List of document dicts with at minimum ``"content"`` and
            ``"source"`` keys.
        """
        ...

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever's index / store.

        Args:
            documents: List of document dicts.  Each must have a ``"content"``
                       key; ``"source"`` and ``"metadata"`` are optional.
        """
        ...
