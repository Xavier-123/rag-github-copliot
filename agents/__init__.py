"""
Agents package for the Agentic RAG system.

Each agent is an independent, decoupled module responsible for a specific
stage of the RAG pipeline.  The calling relationship is:

    QueryUnderstandingAgent
        └─▶ TaskPlanningAgent
                └─▶ RetrievalPlanningAgent
                        └─▶ MultiRetrieverAgent (parallel retrievers)
                                └─▶ RerankAgent
                                        └─▶ ContextEngineeringAgent
                                                └─▶ ReasoningAgent
                                                        └─▶ ReflectionAgent
"""

from .base_agent import AgentInput, AgentOutput, BaseAgent
from .context_engineering_agent import ContextEngineeringAgent
from .multi_retriever_agent import MultiRetrieverAgent
from .query_understanding_agent import QueryUnderstandingAgent
from .reasoning_agent import ReasoningAgent
from .reflection_agent import ReflectionAgent
from .rerank_agent import RerankAgent
from .retrieval_planning_agent import RetrievalPlanningAgent
from .task_planning_agent import TaskPlanningAgent

__all__ = [
    "AgentInput",
    "AgentOutput",
    "BaseAgent",
    "ContextEngineeringAgent",
    "MultiRetrieverAgent",
    "QueryUnderstandingAgent",
    "ReasoningAgent",
    "ReflectionAgent",
    "RerankAgent",
    "RetrievalPlanningAgent",
    "TaskPlanningAgent",
]
