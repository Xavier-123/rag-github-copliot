"""
Agents 模块
===========
包含 RAG 系统中所有 Agent 的实现。
使用懒加载避免在未安装 langchain/langchain_openai 时导入失败。
"""

from rag_system.agents.base import BaseAgent, AgentState

__all__ = [
    "BaseAgent",
    "AgentState",
    "QueryUnderstandingAgent",
    "TaskPlanningAgent",
    "RetrievalStrategyAgent",
    "AdaptiveRetrievalController",
    "MultiRetrieverAgents",
    "RerankAgent",
    "ContextEngineeringAgent",
    "ReasoningAgent",
    "ReflectionAgent",
]


def __getattr__(name):
    """懒加载各 Agent 类，仅在实际引用时才触发导入。"""
    _module_map = {
        "QueryUnderstandingAgent": "rag_system.agents.query_understanding_agent",
        "TaskPlanningAgent": "rag_system.agents.task_planning_agent",
        "RetrievalStrategyAgent": "rag_system.agents.retrieval_strategy_agent",
        "AdaptiveRetrievalController": "rag_system.agents.adaptive_retrieval_controller",
        "MultiRetrieverAgents": "rag_system.agents.multi_retriever_agents",
        "RerankAgent": "rag_system.agents.rerank_agent",
        "ContextEngineeringAgent": "rag_system.agents.context_engineering_agent",
        "ReasoningAgent": "rag_system.agents.reasoning_agent",
        "ReflectionAgent": "rag_system.agents.reflection_agent",
    }
    if name in _module_map:
        import importlib
        module = importlib.import_module(_module_map[name])
        return getattr(module, name)
    raise AttributeError(f"module 'rag_system.agents' has no attribute {name!r}")
