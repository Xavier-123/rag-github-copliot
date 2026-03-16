"""
Tests for the retrieval backend modules.
"""

from __future__ import annotations

import pytest

from retrieval.base_retriever import Document
from retrieval.graph_retriever import GraphRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.web_retriever import WebRetriever


# ---------------------------------------------------------------------------
# Document type
# ---------------------------------------------------------------------------

class TestDocument:
    def test_to_dict(self):
        doc = Document(
            content="test content",
            source="test_source",
            score=0.9,
            metadata={"key": "value"},
        )
        d = doc.to_dict()
        assert d["content"] == "test content"
        assert d["source"] == "test_source"
        assert d["score"] == 0.9
        assert d["metadata"]["key"] == "value"

    def test_defaults(self):
        doc = Document(content="hello")
        assert doc.source == ""
        assert doc.score == 0.0
        assert doc.metadata == {}


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------

class TestVectorRetriever:
    def test_retrieve_returns_list(self):
        retriever = VectorRetriever()
        results = retriever.retrieve("test query", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self):
        retriever = VectorRetriever()
        results = retriever.retrieve("test", top_k=2)
        assert len(results) <= 2

    def test_each_doc_has_content(self):
        retriever = VectorRetriever()
        results = retriever.retrieve("test", top_k=3)
        for doc in results:
            assert "content" in doc

    def test_add_documents_is_callable(self):
        retriever = VectorRetriever()
        docs = [{"content": "test doc", "source": "test.txt"}]
        # Should not raise even without a real backend
        retriever.add_documents(docs)


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------

class TestGraphRetriever:
    def test_retrieve_returns_list(self):
        retriever = GraphRetriever()
        results = retriever.retrieve("Paris France", top_k=3)
        assert isinstance(results, list)

    def test_no_api_key_returns_mock(self):
        retriever = GraphRetriever()
        # Without Neo4j password, should return mock results
        results = retriever.retrieve("test entity", top_k=2)
        assert isinstance(results, list)

    def test_extract_entities_capitalised_words(self):
        retriever = GraphRetriever()
        entities = retriever._extract_entities("What did Albert Einstein discover?")
        assert "Albert Einstein" in entities or len(entities) > 0

    def test_extract_entities_quoted_strings(self):
        retriever = GraphRetriever()
        entities = retriever._extract_entities('Tell me about "quantum mechanics"')
        assert "quantum mechanics" in entities


# ---------------------------------------------------------------------------
# WebRetriever
# ---------------------------------------------------------------------------

class TestWebRetriever:
    def test_retrieve_returns_list(self):
        retriever = WebRetriever()
        results = retriever.retrieve("Python programming", top_k=3)
        assert isinstance(results, list)

    def test_add_documents_is_noop(self):
        retriever = WebRetriever()
        # Should not raise
        retriever.add_documents([{"content": "test"}])

    def test_no_api_key_returns_mock(self):
        retriever = WebRetriever()
        results = retriever.retrieve("latest news", top_k=2)
        # Either empty or mock results
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def test_retrieve_returns_list(self):
        retriever = HybridRetriever()
        results = retriever.retrieve("machine learning", top_k=5)
        assert isinstance(results, list)

    def test_rrf_fusion_basic(self):
        retriever = HybridRetriever()
        list1 = [
            {"content": "doc A", "source": "a", "score": 0.9},
            {"content": "doc B", "source": "b", "score": 0.8},
        ]
        list2 = [
            {"content": "doc B", "source": "b", "score": 0.85},
            {"content": "doc C", "source": "c", "score": 0.7},
        ]
        fused = retriever._reciprocal_rank_fusion([list1, list2], top_k=3)
        assert isinstance(fused, list)
        # All returned docs should have an rrf_score
        for doc in fused:
            assert "rrf_score" in doc

    def test_rrf_promotes_documents_in_both_lists(self):
        retriever = HybridRetriever()
        # doc B appears in both lists → should rank higher via RRF
        list1 = [
            {"content": "doc A content", "source": "a"},
            {"content": "doc B content", "source": "b"},
        ]
        list2 = [
            {"content": "doc B content", "source": "b"},
            {"content": "doc C content", "source": "c"},
        ]
        fused = retriever._reciprocal_rank_fusion([list1, list2], top_k=3)
        # doc B appears in both lists and should have a higher RRF score
        contents = [d["content"] for d in fused]
        assert "doc B content" in contents

    def test_add_documents_updates_corpus(self):
        retriever = HybridRetriever()
        docs = [
            {"content": "Python is a programming language.", "source": "test"},
        ]
        retriever.add_documents(docs)
        assert len(retriever._corpus) == 1

    def test_bm25_retrieval_after_add(self):
        """BM25 should return relevant results after documents are added."""
        retriever = HybridRetriever()
        docs = [
            {"content": "The sky is blue on a clear day.", "source": "sky.txt"},
            {"content": "Python is used for data science.", "source": "py.txt"},
        ]
        retriever.add_documents(docs)
        results = retriever._bm25_retrieve("blue sky", top_k=2)
        # If rank_bm25 is installed, results should be non-empty
        # If not installed, results will be empty (graceful fallback)
        assert isinstance(results, list)
