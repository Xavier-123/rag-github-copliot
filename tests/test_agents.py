"""
Tests for all agent modules.

These tests run in mock mode (no API key required) to verify:
- Agent instantiation and interface conformance
- AgentInput / AgentOutput data flow
- Fallback behaviour when the LLM is unavailable
"""

from __future__ import annotations

import pytest

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from agents.context_engineering_agent import ContextEngineeringAgent
from agents.multi_retriever_agent import MultiRetrieverAgent
from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.reasoning_agent import ReasoningAgent
from agents.reflection_agent import ReflectionAgent
from agents.rerank_agent import RerankAgent
from agents.retrieval_planning_agent import RetrievalPlanningAgent
from agents.task_planning_agent import TaskPlanningAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_docs():
    return [
        {"content": "The Eiffel Tower is located in Paris, France.", "source": "wiki/eiffel", "score": 0.9},
        {"content": "Paris is the capital city of France.", "source": "wiki/paris", "score": 0.85},
        {"content": "The Eiffel Tower was built in 1889.", "source": "wiki/eiffel2", "score": 0.8},
    ]


@pytest.fixture
def basic_input():
    return AgentInput(
        query="Where is the Eiffel Tower located?",
        session_id="test-session-001",
    )


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing the base class."""

    def run(self, agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(result="test", confidence=0.9)


class FailingAgent(BaseAgent):
    """Agent that always raises an exception."""

    def run(self, agent_input: AgentInput) -> AgentOutput:
        raise ValueError("Intentional test error")


class TestBaseAgent:
    def test_execute_returns_output(self):
        agent = ConcreteAgent("TestAgent")
        output = agent.execute(AgentInput(query="test"))
        assert isinstance(output, AgentOutput)
        assert output.result == "test"
        assert output.agent_name == "TestAgent"
        assert output.elapsed_ms > 0

    def test_execute_catches_exceptions(self):
        agent = FailingAgent("FailingAgent")
        output = agent.execute(AgentInput(query="test"))
        assert output.error == "Intentional test error"
        assert output.confidence == 0.0
        assert output.result is None

    def test_agent_name_defaults_to_class_name(self):
        agent = ConcreteAgent()
        assert agent.name == "ConcreteAgent"


# ---------------------------------------------------------------------------
# QueryUnderstandingAgent
# ---------------------------------------------------------------------------

class TestQueryUnderstandingAgent:
    def test_run_returns_output(self, basic_input):
        agent = QueryUnderstandingAgent()
        output = agent.execute(basic_input)
        assert isinstance(output, AgentOutput)
        assert output.result is not None
        assert output.error == ""

    def test_result_has_expected_fields(self, basic_input):
        agent = QueryUnderstandingAgent()
        output = agent.execute(basic_input)
        qu = output.result
        assert hasattr(qu, "original_query")
        assert hasattr(qu, "rewritten_query")
        assert hasattr(qu, "intent")
        assert hasattr(qu, "complexity")
        assert hasattr(qu, "sub_queries")

    def test_original_query_preserved(self, basic_input):
        agent = QueryUnderstandingAgent()
        output = agent.execute(basic_input)
        assert output.result.original_query == basic_input.query

    def test_sub_queries_non_empty(self, basic_input):
        agent = QueryUnderstandingAgent()
        output = agent.execute(basic_input)
        assert len(output.result.sub_queries) >= 1

    def test_complexity_valid_value(self, basic_input):
        agent = QueryUnderstandingAgent()
        output = agent.execute(basic_input)
        assert output.result.complexity in {"simple", "moderate", "complex"}

    def test_intent_valid_value(self, basic_input):
        agent = QueryUnderstandingAgent()
        output = agent.execute(basic_input)
        valid_intents = {"factual", "analytical", "creative", "procedural", "conversational"}
        assert output.result.intent in valid_intents


# ---------------------------------------------------------------------------
# TaskPlanningAgent
# ---------------------------------------------------------------------------

class TestTaskPlanningAgent:
    def test_run_returns_task_plan(self, basic_input):
        agent = TaskPlanningAgent()
        from agents.query_understanding_agent import QueryUnderstanding
        qu = QueryUnderstanding(
            original_query=basic_input.query,
            rewritten_query=basic_input.query,
            intent="factual",
            complexity="simple",
            sub_queries=[basic_input.query],
        )
        inp = AgentInput(
            query=basic_input.query,
            session_id=basic_input.session_id,
            metadata={"query_understanding": {
                "original_query": qu.original_query,
                "rewritten_query": qu.rewritten_query,
                "intent": qu.intent,
                "complexity": qu.complexity,
                "sub_queries": qu.sub_queries,
            }},
        )
        output = agent.execute(inp)
        assert output.result is not None
        plan = output.result
        assert len(plan.tasks) >= 2  # At least retrieve + generate_answer

    def test_complex_plan_has_more_tasks(self):
        agent = TaskPlanningAgent()
        inp = AgentInput(
            query="complex query",
            metadata={"query_understanding": {
                "original_query": "complex query",
                "rewritten_query": "complex query",
                "intent": "analytical",
                "complexity": "complex",
                "sub_queries": ["sub1", "sub2", "sub3"],
            }},
        )
        output = agent.execute(inp)
        plan = output.result
        assert len(plan.tasks) >= 3


# ---------------------------------------------------------------------------
# RetrievalPlanningAgent
# ---------------------------------------------------------------------------

class TestRetrievalPlanningAgent:
    def test_returns_strategies(self, basic_input):
        agent = RetrievalPlanningAgent()
        inp = AgentInput(
            query="Where is the Eiffel Tower?",
            metadata={
                "query_understanding": {"intent": "factual", "complexity": "simple"},
                "task_plan": {
                    "tasks": [
                        {"task_id": 1, "task_type": "retrieve", "description": "Retrieve",
                         "query": "Where is the Eiffel Tower?", "depends_on": [], "parameters": {}}
                    ],
                    "requires_multi_hop": False,
                    "requires_tool_use": False,
                },
            },
        )
        output = agent.execute(inp)
        assert isinstance(output.result, list)
        assert len(output.result) == 1

    def test_recency_keywords_prefer_web(self):
        agent = RetrievalPlanningAgent()
        inp = AgentInput(
            query="latest news today",
            metadata={
                "query_understanding": {"intent": "factual", "complexity": "simple"},
                "task_plan": {
                    "tasks": [
                        {"task_id": 1, "task_type": "retrieve", "description": "Retrieve",
                         "query": "latest news today", "depends_on": [], "parameters": {}}
                    ],
                    "requires_multi_hop": False,
                    "requires_tool_use": False,
                },
            },
        )
        output = agent.execute(inp)
        from config.settings import RetrievalMode
        assert output.result[0].mode == RetrievalMode.WEB


# ---------------------------------------------------------------------------
# MultiRetrieverAgent
# ---------------------------------------------------------------------------

class TestMultiRetrieverAgent:
    def test_returns_documents(self):
        agent = MultiRetrieverAgent()
        from config.settings import RetrievalMode
        inp = AgentInput(
            query="Paris",
            metadata={
                "retrieval_strategies": [
                    {
                        "task_id": 1,
                        "query": "Paris",
                        "mode": RetrievalMode.VECTOR,
                        "top_k": 5,
                        "filters": {},
                        "rationale": "test",
                    }
                ]
            },
        )
        output = agent.execute(inp)
        assert isinstance(output.result, list)

    def test_fallback_when_no_strategies(self):
        agent = MultiRetrieverAgent()
        inp = AgentInput(query="test query")
        output = agent.execute(inp)
        # Should still return something (mock documents)
        assert output.result is not None


# ---------------------------------------------------------------------------
# RerankAgent
# ---------------------------------------------------------------------------

class TestRerankAgent:
    def test_reranks_documents(self, sample_docs):
        agent = RerankAgent()
        inp = AgentInput(
            query="Where is the Eiffel Tower?",
            context=sample_docs,
        )
        output = agent.execute(inp)
        assert isinstance(output.result, list)
        # Should return at most rerank_top_k documents
        from config.settings import settings
        assert len(output.result) <= settings.rerank_top_k

    def test_empty_documents(self):
        agent = RerankAgent()
        inp = AgentInput(query="test", context=[])
        output = agent.execute(inp)
        assert output.result == []


# ---------------------------------------------------------------------------
# ContextEngineeringAgent
# ---------------------------------------------------------------------------

class TestContextEngineeringAgent:
    def test_returns_string(self, sample_docs):
        agent = ContextEngineeringAgent()
        inp = AgentInput(
            query="Where is the Eiffel Tower?",
            context=sample_docs,
        )
        output = agent.execute(inp)
        assert isinstance(output.result, str)
        assert len(output.result) > 0

    def test_empty_input(self):
        agent = ContextEngineeringAgent()
        inp = AgentInput(query="test", context=[])
        output = agent.execute(inp)
        assert output.result == ""


# ---------------------------------------------------------------------------
# ReasoningAgent
# ---------------------------------------------------------------------------

class TestReasoningAgent:
    def test_single_hop_reasoning(self):
        agent = ReasoningAgent()
        inp = AgentInput(
            query="What is the capital of France?",
            metadata={
                "context_str": "Paris is the capital city of France.",
                "task_plan": {"requires_multi_hop": False},
            },
        )
        output = agent.execute(inp)
        assert output.result is not None
        from agents.reasoning_agent import ReasoningOutput
        assert isinstance(output.result, ReasoningOutput)

    def test_multi_hop_reasoning(self):
        agent = ReasoningAgent()
        inp = AgentInput(
            query="Complex multi-hop query",
            metadata={
                "context_str": "Some context here.",
                "task_plan": {"requires_multi_hop": True},
            },
        )
        output = agent.execute(inp)
        assert output.result is not None


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------

class TestReflectionAgent:
    def test_evaluates_answer(self):
        agent = ReflectionAgent()
        inp = AgentInput(
            query="Where is the Eiffel Tower?",
            metadata={
                "candidate_answer": "The Eiffel Tower is in Paris, France.",
                "context_str": "The Eiffel Tower is located in Paris, France.",
            },
        )
        output = agent.execute(inp)
        from agents.reflection_agent import ReflectionResult
        assert isinstance(output.result, ReflectionResult)
        assert 0.0 <= output.result.confidence <= 1.0

    def test_empty_answer_triggers_retry(self):
        agent = ReflectionAgent()
        inp = AgentInput(
            query="test",
            metadata={"candidate_answer": "", "context_str": ""},
        )
        output = agent.execute(inp)
        assert output.result.should_retry is True
