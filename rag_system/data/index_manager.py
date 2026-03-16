"""
Index Manager（LlamaIndex 数据与索引层）
=========================================
使用 LlamaIndex 管理文档的加载、切分、嵌入和索引构建，
是整个 RAG 系统的数据基础设施层。

核心功能：
  1. 文档加载（Document Loading）  : 支持 PDF / TXT / MD / DOCX / URL 等多种格式
  2. 文档切分（Text Splitting）    : 使用 SentenceWindowNodeParser 保留上下文窗口
  3. 向量嵌入（Embedding）         : 使用 OpenAI Embedding 生成向量表示
  4. 索引构建（Index Building）    : 构建 LlamaIndex VectorStoreIndex
  5. 增量更新（Incremental Update）: 支持增量添加文档，无需重建全量索引

技术选型：
  - LlamaIndex SentenceWindowNodeParser : 切分时保留前后 N 句上下文，
    检索后通过 MetadataReplacementPostProcessor 还原完整上下文
  - ChromaDB 持久化存储，避免重启后重建索引
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

from rag_system.config.settings import get_settings


class IndexManager:
    """
    LlamaIndex 索引管理器
    =====================
    提供文档索引的完整生命周期管理。

    使用示例：
    ```python
    manager = IndexManager()
    # 从目录批量加载文档
    manager.load_documents_from_directory("./docs")
    # 从单个文件加载
    manager.load_documents_from_file("./data/manual.pdf")
    # 添加原始文本
    manager.add_text("这是一段需要索引的文本内容")
    ```

    高级特性（SentenceWindow）：
    - 切分粒度：以句子为基本单元
    - 上下文窗口：每个节点包含前后 3 句上下文
    - 检索精度：精确匹配句子级内容
    - 上下文还原：返回给 LLM 的是完整上下文窗口，而非单句
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._logger = logger.bind(module="IndexManager")
        self._index = None
        self._initialized = False

    def _initialize(self) -> None:
        """懒加载初始化 LlamaIndex 环境配置"""
        if self._initialized:
            return
        try:
            self._setup_llama_index()
        except Exception as e:
            self._logger.error(f"LlamaIndex 初始化失败: {e}")
        finally:
            self._initialized = True

    def _setup_llama_index(self) -> None:
        """
        配置 LlamaIndex 全局设置：
        - LLM : 用于摘要和 Query Engine
        - 嵌入模型 : 用于向量化文档和查询
        """
        from llama_index.core import Settings as LlamaSettings
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI as LlamaOpenAI

        LlamaSettings.llm = LlamaOpenAI(
            model=self._cfg.llm.model_name,
            temperature=self._cfg.llm.temperature,
            api_key=self._cfg.llm.api_key,
        )
        LlamaSettings.embed_model = OpenAIEmbedding(
            model=self._cfg.embedding.model_name,
            api_key=self._cfg.llm.api_key,
        )
        self._logger.info("LlamaIndex 全局配置初始化完成")

    def load_documents_from_directory(self, directory: str) -> int:
        """
        从目录批量加载文档并建立索引。
        支持 .txt / .md / .pdf / .docx 格式。

        Args:
            directory: 文档目录路径

        Returns:
            成功加载的文档数量
        """
        self._initialize()
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
            from llama_index.core.node_parser import SentenceWindowNodeParser

            reader = SimpleDirectoryReader(
                input_dir=directory,
                recursive=True,
                filename_as_id=True,
            )
            documents = reader.load_data()
            self._logger.info(f"从 {directory} 加载了 {len(documents)} 个文档")

            # 使用 SentenceWindowNodeParser 切分文档
            # window_size=3 表示每个节点保留前后 3 句上下文
            parser = SentenceWindowNodeParser.from_defaults(window_size=3)
            nodes = parser.get_nodes_from_documents(documents)
            self._logger.info(f"文档切分为 {len(nodes)} 个节点")

            self._index = self._build_or_update_index(nodes)
            return len(documents)

        except Exception as e:
            self._logger.error(f"文档加载失败: {e}")
            return 0

    def load_documents_from_file(self, file_path: str) -> bool:
        """
        加载单个文件并添加到索引。

        Args:
            file_path: 文件路径（支持 .txt / .md / .pdf / .docx）

        Returns:
            是否加载成功
        """
        self._initialize()
        try:
            from llama_index.core import SimpleDirectoryReader
            from llama_index.core.node_parser import SentenceWindowNodeParser

            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            parser = SentenceWindowNodeParser.from_defaults(window_size=3)
            nodes = parser.get_nodes_from_documents(documents)
            self._index = self._build_or_update_index(nodes)
            self._logger.info(f"文件 {file_path} 加载成功，生成 {len(nodes)} 个节点")
            return True
        except Exception as e:
            self._logger.error(f"文件加载失败 {file_path}: {e}")
            return False

    def add_text(self, text: str, metadata: Optional[dict] = None) -> bool:
        """
        直接添加文本内容到索引。

        Args:
            text    : 文本内容
            metadata: 可选元数据（如来源、时间戳等）

        Returns:
            是否添加成功
        """
        self._initialize()
        try:
            from llama_index.core import Document, VectorStoreIndex
            from llama_index.core.node_parser import SentenceWindowNodeParser

            doc = Document(text=text, metadata=metadata or {})
            parser = SentenceWindowNodeParser.from_defaults(window_size=3)
            nodes = parser.get_nodes_from_documents([doc])
            self._index = self._build_or_update_index(nodes)
            self._logger.debug(f"文本添加成功，生成 {len(nodes)} 个节点")
            return True
        except Exception as e:
            self._logger.error(f"文本添加失败: {e}")
            return False

    def _build_or_update_index(self, nodes):
        """
        构建或更新向量索引。
        使用 ChromaDB 作为持久化后端。
        """
        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore

        os.makedirs(self._cfg.vector_store.persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(
            path=self._cfg.vector_store.persist_dir
        )
        collection = chroma_client.get_or_create_collection(
            self._cfg.vector_store.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if self._index is None:
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            # 增量添加节点（不重建全量索引）
            for node in nodes:
                self._index.insert_nodes([node])
            index = self._index

        return index

    def get_index(self):
        """获取 LlamaIndex 索引实例"""
        self._initialize()
        return self._index

    def get_stats(self) -> dict:
        """获取索引统计信息"""
        if self._index is None:
            return {"status": "not_initialized", "doc_count": 0}
        try:
            # 获取 ChromaDB 集合统计
            return {
                "status": "ready",
                "collection": self._cfg.vector_store.collection_name,
                "persist_dir": self._cfg.vector_store.persist_dir,
            }
        except Exception:
            return {"status": "ready", "doc_count": "unknown"}
