"""
配置管理模块
=============
集中管理系统全局配置，支持通过环境变量或 .env 文件覆盖默认值。
使用 Pydantic BaseSettings 实现类型安全的配置。
"""

from __future__ import annotations

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """大语言模型配置"""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: str = Field(default="openai", description="LLM 提供商: openai / azure / local")
    model_name: str = Field(default="gpt-4o-mini", description="模型名称")
    temperature: float = Field(default=0.0, description="生成温度，0 为确定性输出")
    max_tokens: int = Field(default=2048, description="最大生成 token 数")
    api_key: Optional[str] = Field(default=None, description="API Key（优先从环境变量读取）")
    api_base: Optional[str] = Field(default=None, description="自定义 API Base URL")


class EmbeddingSettings(BaseSettings):
    """向量嵌入配置"""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    provider: str = Field(default="openai", description="嵌入模型提供商")
    model_name: str = Field(default="text-embedding-3-small", description="嵌入模型名称")
    dimension: int = Field(default=1536, description="向量维度")
    batch_size: int = Field(default=32, description="批量嵌入大小")


class VectorStoreSettings(BaseSettings):
    """向量数据库配置"""

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", extra="ignore")

    provider: str = Field(default="chroma", description="向量数据库: chroma / pinecone / weaviate")
    persist_dir: str = Field(default="./data/chroma_db", description="本地持久化目录")
    collection_name: str = Field(default="rag_documents", description="集合名称")
    top_k: int = Field(default=10, description="检索返回的最大文档数")
    similarity_threshold: float = Field(default=0.7, description="相似度阈值")


class GraphRAGSettings(BaseSettings):
    """GraphRAG 配置（知识图谱检索）"""

    model_config = SettingsConfigDict(env_prefix="GRAPH_RAG_", extra="ignore")

    enabled: bool = Field(default=False, description="是否启用 GraphRAG")
    workspace_dir: str = Field(default="./data/graphrag", description="GraphRAG 工作目录")
    community_level: int = Field(default=2, description="图社区层级")
    top_k: int = Field(default=5, description="图检索返回节点数")


class WebSearchSettings(BaseSettings):
    """Web 搜索配置"""

    model_config = SettingsConfigDict(env_prefix="WEB_SEARCH_", extra="ignore")

    enabled: bool = Field(default=True, description="是否启用 Web 搜索")
    engine: str = Field(default="duckduckgo", description="搜索引擎: duckduckgo / google / bing")
    max_results: int = Field(default=5, description="最大搜索结果数")
    timeout: int = Field(default=10, description="搜索超时秒数")


class RerankSettings(BaseSettings):
    """重排序模型配置"""

    model_config = SettingsConfigDict(env_prefix="RERANK_", extra="ignore")

    method: str = Field(default="cross_encoder", description="重排方法: cross_encoder / colbert")
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross Encoder 模型名称",
    )
    colbert_model: str = Field(
        default="colbert-ir/colbertv2.0",
        description="ColBERT 模型名称",
    )
    top_k: int = Field(default=5, description="重排后保留的文档数")


class MemorySettings(BaseSettings):
    """记忆系统配置"""

    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")

    enabled: bool = Field(default=True, description="是否启用记忆系统")
    max_history: int = Field(default=20, description="最大对话历史轮数")
    summary_threshold: int = Field(default=10, description="触发历史摘要的轮数阈值")
    persist_dir: str = Field(default="./data/memory", description="记忆持久化目录")


class EvaluationSettings(BaseSettings):
    """评估与优化配置"""

    model_config = SettingsConfigDict(env_prefix="EVAL_", extra="ignore")

    enabled: bool = Field(default=True, description="是否启用评估模块")
    ragas_metrics: list = Field(
        default=["faithfulness", "answer_relevancy", "context_precision"],
        description="RAGAS 评估指标列表",
    )
    feedback_threshold: float = Field(default=0.7, description="触发优化的评估分数阈值")


class Settings(BaseSettings):
    """
    全局系统配置
    ===========
    支持通过 .env 文件或环境变量覆盖默认值。

    示例 .env 文件：
        LLM_MODEL_NAME=gpt-4o
        LLM_API_KEY=sk-xxx
        VECTOR_STORE_TOP_K=15
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 子配置 ---
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    graph_rag: GraphRAGSettings = Field(default_factory=GraphRAGSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    # --- 系统级配置 ---
    log_level: str = Field(default="INFO", description="日志级别")
    max_retries: int = Field(default=3, description="最大重试次数")
    enable_tracing: bool = Field(default=False, description="是否启用 LangSmith 追踪")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API Key")


# 全局单例配置实例（延迟初始化，避免导入时加载 .env）
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
