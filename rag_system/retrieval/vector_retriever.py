"""
Vector Retriever（向量数据库检索器）
=====================================
基于 LlamaIndex 实现的语义向量检索。
使用向量相似度（余弦相似度）在向量数据库中检索与查询最相关的文档片段。

技术栈：
  - LlamaIndex VectorStoreIndex    : 向量索引构建与查询
  - ChromaDB                       : 向量存储后端（可替换）
  - OpenAI text-embedding-3-small  : 嵌入模型（可替换）

检索流程：
  query → 向量化(embedding) → 向量数据库 ANN 搜索 → 返回 top_k 相似文档
"""

from __future__ import annotations

import os
from typing import List, Optional

from loguru import logger

from rag_system.agents.base import RetrievedDocument
from rag_system.config.settings import get_settings


class VectorRetriever:
    """
    向量检索器
    ==========
    封装 LlamaIndex 向量索引查询接口。
    支持 ChromaDB 持久化存储，避免每次重建索引。

    索引初始化：
    - 若本地已有持久化索引，直接加载
    - 否则创建空索引（通过 IndexManager 添加文档）

    [预留接口]：
    - 通过 IndexManager.add_documents() 向索引添加文档
    - 支持切换到 Pinecone / Weaviate 等云端向量数据库
    """

    def __init__(self, settings=None) -> None:
        self._cfg = settings or get_settings()
        self._logger = logger.bind(module="VectorRetriever")
        self._index = None
        self._retriever = None
        self._initialized = False

    def _initialize(self) -> None:
        """懒加载初始化 LlamaIndex 向量索引"""
        if self._initialized:
            return
        try:
            self._setup_index()
        except Exception as e:
            self._logger.warning(f"向量索引初始化失败，将使用 Mock 检索: {e}")
        finally:
            self._initialized = True

    def _setup_index(self) -> None:
        """
        配置并加载 LlamaIndex 向量存储索引。

        使用 ChromaDB 作为持久化向量存储后端：
        - 支持增量更新，无需重建全量索引
        - 本地文件系统持久化，适合中小规模部署
        """
        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Settings as LlamaSettings

        # 配置嵌入模型
        LlamaSettings.embed_model = OpenAIEmbedding(
            model=self._cfg.embedding.model_name,
            api_key=self._cfg.llm.api_key,
        )

        # 初始化 ChromaDB
        os.makedirs(self._cfg.vector_store.persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(
            path=self._cfg.vector_store.persist_dir
        )
        collection = chroma_client.get_or_create_collection(
            self._cfg.vector_store.collection_name
        )

        # 构建 LlamaIndex 向量存储
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
        self._retriever = self._index.as_retriever(
            similarity_top_k=self._cfg.vector_store.top_k
        )
        self._logger.info(
            f"向量索引加载成功: collection={self._cfg.vector_store.collection_name}"
        )

    def retrieve(self, query: str) -> List[RetrievedDocument]:
        """
        执行向量相似度检索。

        Args:
            query: 查询文本

        Returns:
            RetrievedDocument 列表，按相关性降序排列
        """
        self._initialize()

        if self._retriever is None:
            return self._mock_retrieve(query)

        try:
            nodes = self._retriever.retrieve(query)
            docs = []
            for node in nodes:
                docs.append(
                    RetrievedDocument(
                        doc_id=node.node.node_id,
                        content=node.node.get_content(),
                        source="vector",
                        score=float(node.score or 0.0),
                        metadata=dict(node.node.metadata or {}),
                    )
                )
            return docs
        except Exception as e:
            self._logger.error(f"向量检索失败: {e}")
            return self._mock_retrieve(query)

    def _mock_retrieve(self, query: str) -> List[RetrievedDocument]:
        """
        Mock 检索（当向量索引不可用时的降级方案）。
        返回包含查询信息的占位文档，用于开发调试。
        """
        self._logger.warning("使用 Mock 向量检索（索引不可用）")
        return [
            RetrievedDocument(
                doc_id="mock-vector-1",
                content=f"[Mock 向量检索] 查询: '{query}' — 这是一段来自向量数据库的占位内容。"
                        "生产环境中，此处将返回真实的相关文档片段。",
                source="vector",
                score=0.75,
                metadata={"is_mock": True},
            )
        ]

    def get_index(self):
        """获取 LlamaIndex 索引实例（供 IndexManager 使用）"""
        self._initialize()
        return self._index
