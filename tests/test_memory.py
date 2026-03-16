"""
Tests for the memory layer modules.
"""

from __future__ import annotations

import time

import pytest

from memory.conversation_memory import ConversationMemory
from memory.episodic_memory import EpisodicMemory
from memory.knowledge_memory import KnowledgeMemory


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class TestConversationMemory:
    def test_add_and_get_turn(self):
        mem = ConversationMemory()
        mem.add_turn("session-1", "user", "Hello!")
        mem.add_turn("session-1", "assistant", "Hi there!")
        history = mem.get_history("session-1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"
        assert history[1]["role"] == "assistant"

    def test_empty_history_for_new_session(self):
        mem = ConversationMemory()
        history = mem.get_history("non-existent-session")
        assert history == []

    def test_sliding_window(self):
        mem = ConversationMemory()
        # Add more turns than the window size
        for i in range(mem._window + 5):
            mem.add_turn("session-w", "user", f"message {i}")
        history = mem.get_history("session-w")
        # Should be bounded by the window size
        assert len(history) <= mem._window

    def test_clear_session(self):
        mem = ConversationMemory()
        mem.add_turn("session-c", "user", "test")
        mem.clear("session-c")
        history = mem.get_history("session-c")
        assert history == []

    def test_multiple_sessions_are_independent(self):
        mem = ConversationMemory()
        mem.add_turn("session-A", "user", "message A")
        mem.add_turn("session-B", "user", "message B")
        history_a = mem.get_history("session-A")
        history_b = mem.get_history("session-B")
        assert history_a[0]["content"] == "message A"
        assert history_b[0]["content"] == "message B"
        assert len(history_a) == 1
        assert len(history_b) == 1


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def test_add_and_retrieve_episode(self):
        mem = EpisodicMemory()
        episode_id = mem.add_episode(
            session_id="s1",
            query="test query",
            strategy_used="vector",
            evaluation_score=0.8,
        )
        assert episode_id
        episodes = mem.get_episodes()
        assert len(episodes) == 1
        assert episodes[0]["strategy_used"] == "vector"

    def test_filter_by_strategy(self):
        mem = EpisodicMemory()
        mem.add_episode("s1", "q1", "vector", 0.8)
        mem.add_episode("s1", "q2", "hybrid", 0.75)
        mem.add_episode("s1", "q3", "vector", 0.9)
        vector_eps = mem.get_episodes(strategy="vector")
        assert all(e["strategy_used"] == "vector" for e in vector_eps)
        assert len(vector_eps) == 2

    def test_filter_by_min_score(self):
        mem = EpisodicMemory()
        mem.add_episode("s1", "q1", "vector", 0.9)
        mem.add_episode("s1", "q2", "vector", 0.3)
        high_score_eps = mem.get_episodes(min_score=0.8)
        assert all(e["evaluation_score"] >= 0.8 for e in high_score_eps)

    def test_get_best_strategies(self):
        mem = EpisodicMemory()
        for _ in range(5):
            mem.add_episode("s", "q", "hybrid", 0.9)
        for _ in range(5):
            mem.add_episode("s", "q", "vector", 0.6)
        best = mem.get_best_strategies(top_n=1)
        assert best[0]["strategy"] == "hybrid"
        assert best[0]["avg_score"] == pytest.approx(0.9)

    def test_episode_count(self):
        mem = EpisodicMemory()
        for i in range(5):
            mem.add_episode("s", f"q{i}", "vector", 0.7)
        assert mem.episode_count() == 5

    def test_add_user_rating(self):
        mem = EpisodicMemory()
        ep_id = mem.add_episode("s", "q", "vector", 0.8)
        result = mem.add_user_rating(ep_id, 5)
        assert result is True
        episodes = mem.get_episodes()
        assert episodes[0]["user_rating"] == 5

    def test_add_user_rating_unknown_id(self):
        mem = EpisodicMemory()
        result = mem.add_user_rating("non-existent-id", 5)
        assert result is False


# ---------------------------------------------------------------------------
# KnowledgeMemory
# ---------------------------------------------------------------------------

class TestKnowledgeMemory:
    @pytest.fixture
    def sample_documents(self):
        return [
            {
                "content": "The Python programming language was created by Guido van Rossum.",
                "source": "python_history.txt",
                "metadata": {"topic": "python"},
            },
            {
                "content": "Machine learning is a subset of artificial intelligence.",
                "source": "ml_intro.txt",
                "metadata": {"topic": "ml"},
            },
        ]

    def test_add_documents_returns_count(self, sample_documents):
        km = KnowledgeMemory()
        count = km.add_documents(sample_documents)
        assert count == len(sample_documents)

    def test_document_count_updates(self, sample_documents):
        km = KnowledgeMemory()
        km.add_documents(sample_documents)
        assert km.document_count() == len(sample_documents)

    def test_list_documents(self, sample_documents):
        km = KnowledgeMemory()
        km.add_documents(sample_documents)
        docs = km.list_documents()
        assert len(docs) == len(sample_documents)
        assert all("id" in d for d in docs)

    def test_chunking_long_document(self):
        km = KnowledgeMemory(chunk_size=100, chunk_overlap=20)
        long_doc = {
            "content": "A" * 350,
            "source": "long_doc.txt",
        }
        count = km.add_documents([long_doc])
        # 350 chars / (100 - 20) = ~4-5 chunks
        assert count >= 3

    def test_add_empty_documents(self):
        km = KnowledgeMemory()
        count = km.add_documents([])
        assert count == 0
