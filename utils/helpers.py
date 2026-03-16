"""
Shared utility functions used across the RAG system.

This module provides:
- LLM / embedding client factories (OpenAI & Azure OpenAI)
- Text utilities (truncation, deduplication)
- Vector math helpers
- Centralised logging factory
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

from config.settings import LLMProvider, settings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger pre-configured with the global log format.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(settings.log_format))
        logger.addHandler(handler)
    logger.setLevel(settings.log_level.upper())
    return logger


# ---------------------------------------------------------------------------
# LLM & Embedding client factories
# ---------------------------------------------------------------------------

def is_llm_available() -> bool:
    """
    Return ``True`` when a real LLM can be reached; ``False`` in mock/offline mode.

    A real LLM is considered available when **any** of the following is true:

    - An OpenAI API key is configured (``settings.openai_api_key``).
    - An Azure OpenAI API key is configured (``settings.azure_openai_api_key``).
    - A custom ``base_url`` is configured for the OpenAI provider
      (``settings.openai_base_url``), which covers OpenAI-compatible services
      such as Ollama, DeepSeek, or other self-hosted endpoints that may not
      require a traditional API key.

    Returns:
        ``True`` if an LLM endpoint is reachable, ``False`` otherwise.
    """
    if settings.openai_api_key or settings.azure_openai_api_key:
        return True
    if settings.llm_provider == LLMProvider.OPENAI and settings.openai_base_url:
        return True
    return False


def build_llm_client() -> Any:
    """
    Build and return the appropriate LLM client based on ``settings.llm_provider``.

    Supported providers
    -------------------
    - ``LLMProvider.OPENAI``       – uses ``openai.OpenAI``
    - ``LLMProvider.AZURE_OPENAI`` – uses ``openai.AzureOpenAI``

    When ``settings.openai_base_url`` is set, it is forwarded as the ``base_url``
    parameter, enabling any OpenAI-compatible service (e.g. Ollama, DeepSeek,
    SiliconFlow) to be used as a drop-in replacement.

    Returns:
        An OpenAI-compatible chat client instance.

    Raises:
        ValueError: If the configured provider is not supported.
        ImportError: If the ``openai`` package is not installed.
    """
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    if settings.llm_provider == LLMProvider.OPENAI:
        kwargs: Dict[str, Any] = {}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        # When a custom base_url is used without an explicit API key (e.g. Ollama),
        # provide a placeholder so the openai library does not raise an
        # AuthenticationError before the request reaches the endpoint.
        kwargs["api_key"] = settings.openai_api_key or (
            "ollama" if settings.openai_base_url else None
        )
        return openai.OpenAI(**kwargs)

    if settings.llm_provider == LLMProvider.AZURE_OPENAI:
        return openai.AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def build_embedding_client() -> Any:
    """
    Build and return an embedding client compatible with the configured LLM provider.

    Returns:
        An OpenAI-compatible client whose ``.embeddings`` attribute can be called.
    """
    # Both OpenAI and Azure OpenAI expose the same embeddings API surface.
    return build_llm_client()


def get_model_name() -> str:
    """Return the effective chat model name based on the configured provider."""
    if settings.llm_provider == LLMProvider.AZURE_OPENAI:
        return settings.azure_openai_deployment or "gpt-4o"
    return settings.openai_model


def get_embedding_model_name() -> str:
    """Return the effective embedding model name based on the configured provider."""
    if settings.llm_provider == LLMProvider.AZURE_OPENAI:
        return settings.azure_openai_embedding_deployment or "text-embedding-3-small"
    return settings.openai_embedding_model


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_tokens: int = 4000, chars_per_token: int = 4) -> str:
    """
    Truncate *text* to approximately *max_tokens* tokens.

    This is a heuristic approximation (``chars_per_token`` characters ≈ 1 token)
    suitable for preventing context-window overflows before a precise tokeniser
    is available.

    Args:
        text:            Input text.
        max_tokens:      Maximum allowed token count (approximate).
        chars_per_token: Average characters per token used for estimation.

    Returns:
        Possibly truncated text string.
    """
    limit = max_tokens * chars_per_token
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def deduplicate_documents(
    documents: List[Dict[str, Any]],
    content_key: str = "content",
) -> List[Dict[str, Any]]:
    """
    Remove duplicate documents by content hash.

    Documents are considered duplicates when their text content (after
    whitespace normalisation) produces the same MD5 digest.

    Args:
        documents:   List of document dictionaries.
        content_key: Key used to access the text content in each document.

    Returns:
        Deduplicated list preserving original order (first occurrence wins).
    """
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for doc in documents:
        content = doc.get(content_key, "")
        digest = hashlib.md5(content.strip().encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(doc)
    return unique


def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """
    Flatten a one-level nested list.

    Args:
        nested: A list whose elements are themselves lists.

    Returns:
        A single flat list containing all elements.
    """
    return [item for sublist in nested for item in sublist]


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two dense vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity in the range ``[-1, 1]``.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def format_documents_for_context(
    documents: List[Dict[str, Any]],
    content_key: str = "content",
    source_key: str = "source",
    max_chars: Optional[int] = None,
) -> str:
    """
    Format a list of retrieved documents into a single context string.

    Each document is prefixed with its index and optional source metadata,
    making it easy to include in a prompt.

    Args:
        documents:   Retrieved document list.
        content_key: Key for document text content.
        source_key:  Key for source/metadata info.
        max_chars:   If set, truncate the combined result to this many characters.

    Returns:
        Formatted multi-document context string.
    """
    parts: List[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.get(source_key, "unknown")
        content = doc.get(content_key, "")
        parts.append(f"[Document {i}] (source: {source})\n{content}")
    combined = "\n\n---\n\n".join(parts)
    if max_chars and len(combined) > max_chars:
        combined = combined[:max_chars] + "\n...[truncated]"
    return combined
