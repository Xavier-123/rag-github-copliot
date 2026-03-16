"""
Integration tests for the full RAG pipeline.

These tests exercise the complete pipeline in mock mode (no API key required),
verifying that all components wire together correctly and that the pipeline
produces expected output shapes.
"""

from __future__ import annotations

import pytest

from evaluation.evaluator import EvaluationResult, Evaluator
from memory.episodic_memory import EpisodicMemory
from optimization.feedback_learning import FeedbackLearning, LearningUpdate
from optimization.system_optimizer import SystemOptimizer
from pipeline.rag_pipeline import ChatResponse, RAGPipeline


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        """Create a single pipeline instance shared across tests in the class."""
        return RAGPipeline()

    def test_instantiation(self, pipeline):
        assert pipeline is not None

    def test_chat_returns_response(self, pipeline):
        response = pipeline.chat("What is RAG?", session_id="test-001")
        assert isinstance(response, ChatResponse)
        assert response.query == "What is RAG?"
        assert response.session_id == "test-001"

    def test_chat_has_answer(self, pipeline):
        response = pipeline.chat("What is the capital of France?", session_id="test-002")
        assert response.answer
        assert len(response.answer) > 0

    def test_chat_tracks_session(self, pipeline):
        session = "session-tracking-test"
        pipeline.chat("Hello", session_id=session)
        pipeline.chat("How are you?", session_id=session)
        history = pipeline._conversation_memory.get_history(session)
        # Should have 2 user turns + 2 assistant turns = 4 entries
        assert len(history) == 4

    def test_auto_session_id(self, pipeline):
        response = pipeline.chat("Test without session ID")
        assert response.session_id
        assert len(response.session_id) > 0

    def test_elapsed_ms_populated(self, pipeline):
        response = pipeline.chat("Quick question", session_id="test-elapsed")
        assert response.elapsed_ms > 0

    def test_metadata_has_expected_keys(self, pipeline):
        response = pipeline.chat("Test metadata", session_id="test-meta")
        assert "intent" in response.metadata
        assert "complexity" in response.metadata
        assert "num_retrieved" in response.metadata

    def test_add_documents(self, pipeline):
        docs = [
            {
                "content": "RAG stands for Retrieval Augmented Generation.",
                "source": "rag_overview.txt",
            }
        ]
        count = pipeline.add_documents(docs)
        assert count >= 1

    def test_optimizer_status_accessible(self, pipeline):
        status = pipeline.get_optimizer_status()
        assert "cycle_count" in status
        assert "rerank_top_k" in status


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_evaluate_returns_result(self):
        evaluator = Evaluator()
        result = evaluator.evaluate(
            query="What is AI?",
            answer="AI stands for Artificial Intelligence.",
            context=["Artificial Intelligence (AI) is the simulation of human intelligence."],
        )
        assert isinstance(result, EvaluationResult)

    def test_scores_in_valid_range(self):
        evaluator = Evaluator()
        result = evaluator.evaluate(
            query="test",
            answer="test answer",
            context=["test context"],
        )
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0
        assert 0.0 <= result.context_precision <= 1.0
        assert 0.0 <= result.context_recall <= 1.0
        assert 0.0 <= result.overall_score <= 1.0

    def test_disabled_evaluation(self, monkeypatch):
        from config.settings import settings as s
        monkeypatch.setattr(s, "evaluation_enabled", False)
        evaluator = Evaluator()
        result = evaluator.evaluate("q", "a", ["ctx"])
        assert result.overall_score == 1.0


# ---------------------------------------------------------------------------
# FeedbackLearning
# ---------------------------------------------------------------------------

class TestFeedbackLearning:
    def test_learn_returns_update(self):
        mem = EpisodicMemory()
        learner = FeedbackLearning(mem)
        eval_result = EvaluationResult(
            faithfulness=0.8,
            answer_relevance=0.85,
            context_precision=0.75,
            context_recall=0.7,
            overall_score=0.78,
        )
        update = learner.learn(eval_result, {"session_id": "s1", "query": "q", "strategy_used": "vector"})
        assert isinstance(update, LearningUpdate)

    def test_poor_score_adds_notes(self):
        mem = EpisodicMemory()
        learner = FeedbackLearning(mem)
        eval_result = EvaluationResult(
            faithfulness=0.2,
            answer_relevance=0.3,
            context_precision=0.2,
            context_recall=0.25,
            overall_score=0.25,
        )
        update = learner.learn(eval_result, {"session_id": "s2", "query": "q", "strategy_used": "vector"})
        assert len(update.notes) > 0


# ---------------------------------------------------------------------------
# SystemOptimizer
# ---------------------------------------------------------------------------

class TestSystemOptimizer:
    def test_apply_strategy_override(self):
        optimizer = SystemOptimizer()
        update = LearningUpdate(preferred_strategies={"factual": "hybrid"})
        result = optimizer.apply(update)
        assert "factual" in optimizer._strategy_overrides
        assert optimizer.get_strategy_override("factual") == "hybrid"

    def test_apply_rerank_top_k(self):
        optimizer = SystemOptimizer()
        update = LearningUpdate(rerank_top_k=7)
        optimizer.apply(update)
        assert optimizer.get_rerank_top_k() == 7

    def test_cycle_count_increments(self):
        optimizer = SystemOptimizer()
        assert optimizer._cycle_count == 0
        optimizer.apply(LearningUpdate())
        optimizer.apply(LearningUpdate())
        assert optimizer._cycle_count == 2

    def test_get_status(self):
        optimizer = SystemOptimizer()
        status = optimizer.get_status()
        assert "cycle_count" in status
        assert "rerank_top_k" in status
        assert "strategy_overrides" in status

    def test_optimization_history(self):
        optimizer = SystemOptimizer()
        optimizer.apply(LearningUpdate(notes=["Test note"]))
        history = optimizer.get_optimization_history()
        assert len(history) == 1
        assert history[0].cycle_id == 1
