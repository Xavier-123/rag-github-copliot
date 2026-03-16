"""
RAG Pipeline（高层接口）
========================
对外暴露简洁的 Python API，封装底层 RAGWorkflow 与 IndexManager，
并确保在初始化前正确加载 .env 文件中的配置。

问题背景：
  pydantic-settings 在读取 env_file 时，只会在当前 Settings 实例内部使用
  读取到的值，不会写入 os.environ。因此，通过 default_factory 独立实例化的
  子配置类（LLMSettings、EmbeddingSettings 等）无法从 .env 文件获取到值。
  本模块在初始化时调用 load_dotenv() 将 .env 的内容写入 os.environ，从而
  保证所有子配置类都能正确读取到配置。

使用示例：
    from pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()

    # 添加文档
    pipeline.add_documents([
        {"content": "RAG is Retrieval Augmented Generation.", "source": "intro.txt"}
    ])

    # 对话
    response = pipeline.chat("What is RAG?", session_id="user-001")
    print(response.answer)
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Sources: {response.sources}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from loguru import logger


# ============================================================
# 响应数据模型
# ============================================================

@dataclass
class ChatResponse:
    """
    RAGPipeline.chat() 的返回结果。

    Attributes:
        answer           : 生成的答案文本
        confidence       : 置信度（0.0 ~ 1.0），由评估分数或反思结果推导
        sources          : 参考文档的来源列表（去重后）
        reasoning_steps  : Chain-of-Thought 推理步骤（仅分析型查询非空）
        evaluation_scores: RAGAS 评估指标字典
    """

    answer: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    evaluation_scores: Dict[str, float] = field(default_factory=dict)


# ============================================================
# RAGPipeline
# ============================================================

class RAGPipeline:
    """
    企业级 RAG 对话系统的高层接口。
    ==================================
    封装 RAGWorkflow（LangGraph 工作流）与 IndexManager（文档索引），
    提供简洁的文档添加和对话 API。

    初始化顺序：
    1. 调用 load_dotenv() 将 .env 写入 os.environ
       （确保 LLMSettings 等子配置类能读取到 API Key 等配置）
    2. 重置配置单例，使新加载的环境变量生效
    3. 初始化 RAGWorkflow 和 IndexManager

    Args:
        env_file: .env 文件路径，默认自动向上查找（find_dotenv）
    """

    def __init__(self, env_file: str = ".env") -> None:
        # Step 1: 将 .env 内容加载到 os.environ，使所有子配置类可读取
        _env_path = find_dotenv(filename=env_file, usecwd=True) or env_file
        load_dotenv(_env_path, override=False)

        # Step 2: 重置配置单例，确保本次初始化后创建的 Settings 实例
        #         能读取到最新的 os.environ
        from rag_system.config.settings import reset_settings
        reset_settings()

        # Step 3: 延迟导入，保证在配置已就绪后再初始化各组件
        from rag_system.workflow.graph import RAGWorkflow
        from rag_system.data.index_manager import IndexManager

        self._workflow = RAGWorkflow()
        self._index_manager = IndexManager()
        self._logger = logger.bind(module="RAGPipeline")
        self._logger.info("RAGPipeline 初始化完成")

    # ──────────────────────────────────────────────────────────
    # 文档管理
    # ──────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        向知识库添加文档。

        Args:
            documents: 文档列表，每个元素为包含 'content' 键的字典，
                       其余键值对作为 metadata 存储（例如 'source'）。

        Returns:
            成功添加的文档数量
        """
        added = 0
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            metadata = {k: v for k, v in doc.items() if k != "content"}
            if self._index_manager.add_text(content, metadata=metadata):
                added += 1
        self._logger.info(f"添加文档: {added}/{len(documents)} 成功")
        return added

    # ──────────────────────────────────────────────────────────
    # 对话
    # ──────────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        发送查询并返回结构化响应。

        Args:
            query     : 用户问题
            session_id: 会话 ID，用于多轮对话记忆隔离；不指定则自动生成

        Returns:
            ChatResponse 对象，包含 answer、confidence、sources 等字段
        """
        result = self._workflow.run(query, session_id=session_id)

        answer = result.get("answer", "")
        evaluation_scores: Dict[str, float] = result.get("evaluation_scores") or {}

        # 置信度：优先使用评估分数的均值，否则由反思结果推导
        if evaluation_scores:
            confidence = sum(evaluation_scores.values()) / len(evaluation_scores)
        else:
            reflection = result.get("reflection_result", "passed")
            confidence = 0.8 if reflection == "passed" else 0.5

        # 去重后的参考来源列表
        docs = result.get("reranked_docs") or result.get("retrieved_docs") or []
        seen: dict = {}
        for doc in docs:
            src = (
                getattr(doc, "metadata", {}).get("source")
                or getattr(doc, "source", None)
            )
            if src and src not in seen:
                seen[src] = True
        sources: List[str] = list(seen.keys())

        return ChatResponse(
            answer=answer,
            confidence=round(confidence, 4),
            sources=sources,
            reasoning_steps=result.get("reasoning_steps") or [],
            evaluation_scores=evaluation_scores,
        )
