"""
Global configuration settings for the Agentic RAG + Self-Improving RAG system.

This module centralizes all configuration parameters including:
- LLM provider settings (OpenAI / Azure OpenAI)
- Retrieval backend connection strings
- Memory backend settings
- Evaluation and optimization parameters
- Logging configuration

Usage:
    from config.settings import settings
    api_key = settings.openai_api_key
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Absolute path to the project root (two levels up from this file: config/ → root)
_ROOT_DIR = Path(__file__).resolve().parent.parent


class LLMProvider(str, Enum):
    """Supported LLM service providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class RetrievalMode(str, Enum):
    """Available retrieval strategy modes."""

    VECTOR = "vector"
    GRAPH = "graph"
    WEB = "web"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """
    Central settings object loaded from environment variables or a .env file.

    Priority order for each variable:
      1. Actual environment variable
      2. Value defined in the .env file
      3. Default value specified here
    """

    # -------------------------------------------------------------------------
    # LLM Provider
    # -------------------------------------------------------------------------
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM backend to use: 'openai' or 'azure_openai'.",
    )

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key.")
    openai_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Custom base URL for OpenAI-compatible services "
            "(e.g. 'http://localhost:11434/v1' for Ollama, or a third-party "
            "OpenAI-compatible endpoint). Leave unset to use the default "
            "OpenAI API URL."
        ),
    )
    openai_model: str = Field(
        default="gpt-4o", description="Default OpenAI chat model."
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model for vector retrieval.",
    )

    # Azure OpenAI
    azure_openai_api_key: Optional[str] = Field(
        default=None, description="Azure OpenAI API key."
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="Azure OpenAI resource endpoint URL."
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01", description="Azure OpenAI API version."
    )
    azure_openai_deployment: Optional[str] = Field(
        default=None, description="Azure OpenAI deployment name (chat model)."
    )
    azure_openai_embedding_deployment: Optional[str] = Field(
        default=None, description="Azure OpenAI deployment name (embedding model)."
    )

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    default_retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Default retrieval strategy when not dynamically decided.",
    )
    retrieval_top_k: int = Field(
        default=10, description="Number of candidate documents to retrieve."
    )
    rerank_top_k: int = Field(
        default=5, description="Number of documents to keep after re-ranking."
    )

    # Vector store (Chroma / FAISS / Milvus / Weaviate)
    vector_store_type: str = Field(
        default="chroma",
        description="Vector store backend: 'chroma', 'faiss', 'milvus', or 'weaviate'.",
    )
    chroma_host: str = Field(default="localhost", description="ChromaDB host.")
    chroma_port: int = Field(default=8000, description="ChromaDB HTTP port.")
    chroma_collection: str = Field(
        default="rag_knowledge", description="ChromaDB collection name."
    )
    faiss_index_path: str = Field(
        default="./data/faiss_index", description="Local path to persist FAISS index."
    )
    milvus_uri: str = Field(
        default="http://localhost:19530", description="Milvus server URI."
    )
    weaviate_url: str = Field(
        default="http://localhost:8080", description="Weaviate server URL."
    )

    # Graph store (Neo4j)
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI.")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username.")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password.")

    # Web search
    web_search_provider: str = Field(
        default="serper",
        description="Web search provider: 'serper', 'tavily', or 'bing'.",
    )
    serper_api_key: Optional[str] = Field(default=None, description="Serper API key.")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key.")
    bing_search_api_key: Optional[str] = Field(
        default=None, description="Bing Search API key."
    )

    # -------------------------------------------------------------------------
    # Memory
    # -------------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL."
    )
    conversation_window: int = Field(
        default=10, description="Maximum number of turns to keep in conversation memory."
    )
    episodic_memory_max_entries: int = Field(
        default=1000,
        description="Maximum number of episodic memory entries to maintain.",
    )

    # -------------------------------------------------------------------------
    # Evaluation & Optimization
    # -------------------------------------------------------------------------
    evaluation_enabled: bool = Field(
        default=True, description="Whether to run automatic quality evaluation."
    )
    evaluation_framework: str = Field(
        default="llm_judge",
        description="Evaluation framework: 'ragas', 'llm_judge', or 'dspy'.",
    )
    feedback_learning_enabled: bool = Field(
        default=True,
        description="Whether to apply feedback-based learning after each evaluation.",
    )
    optimization_cycle_turns: int = Field(
        default=50,
        description="Number of conversation turns between system optimization cycles.",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level.")
    log_format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Python logging format string.",
    )

    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    max_reasoning_iterations: int = Field(
        default=5,
        description="Maximum reasoning iterations for the Reasoning Agent.",
    )
    confidence_threshold: float = Field(
        default=0.7,
        description=(
            "Minimum confidence score; below this the Reflection Agent triggers a retry. "
            "0.7 balances quality (avoiding low-confidence answers) with cost "
            "(avoiding excessive LLM retries). Tune down to 0.5 for speed, up to 0.85 "
            "for higher quality requirements."
        ),
    )
    feedback_poor_score_threshold: float = Field(
        default=0.5,
        description=(
            "Evaluation score below which a retrieval strategy is flagged as a poor "
            "performer by FeedbackLearning. Strategies below this threshold with enough "
            "episodes are added to the discouraged list."
        ),
    )

    @field_validator("openai_api_key", "azure_openai_api_key", mode="before")
    @classmethod
    def _strip_quotes(cls, v: Optional[str]) -> Optional[str]:
        """Remove accidental surrounding quotes from API key values."""
        if isinstance(v, str):
            return v.strip("\"'")
        return v

    model_config = {
        "env_prefix": "RAG_",
        "env_file": str(_ROOT_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Module-level singleton – import this everywhere instead of instantiating
# Settings() repeatedly.
# ---------------------------------------------------------------------------
settings = Settings()
