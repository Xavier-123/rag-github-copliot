"""
Enterprise RAG Dialogue System
================================
基于 LangGraph + LlamaIndex + GraphRAG + RAGatouille + DSPy + RAGAS
构建的企业级 RAG 对话系统。

模块结构：
- config/       : 全局配置管理
- agents/       : 各 Agent 模块（查询理解、任务规划、检索策略、上下文工程、推理、反思）
- memory/       : 记忆检索与更新模块
- retrieval/    : 多路检索器（向量、知识图谱、Web）
- evaluation/   : 评估与反馈模块（DSPy + RAGAS）
- workflow/     : LangGraph 工作流编排
- data/         : LlamaIndex 数据与索引层

使用懒加载避免在测试环境中因缺少可选依赖（langchain、langgraph等）而导致导入失败。
"""

__all__ = ["RAGWorkflow", "Settings"]
__version__ = "1.0.0"


def __getattr__(name):
    """懒加载顶级导出，仅在实际使用时才导入相关依赖。"""
    if name == "RAGWorkflow":
        from rag_system.workflow.graph import RAGWorkflow
        return RAGWorkflow
    if name == "Settings":
        from rag_system.config.settings import Settings
        return Settings
    raise AttributeError(f"module 'rag_system' has no attribute {name!r}")
