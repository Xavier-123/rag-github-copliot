"""Utility helpers for the RAG system."""

from .helpers import (
    build_embedding_client,
    build_llm_client,
    cosine_similarity,
    deduplicate_documents,
    flatten_list,
    format_documents_for_context,
    get_embedding_model_name,
    get_logger,
    get_model_name,
    truncate_text,
)

__all__ = [
    "build_embedding_client",
    "build_llm_client",
    "cosine_similarity",
    "deduplicate_documents",
    "flatten_list",
    "format_documents_for_context",
    "get_embedding_model_name",
    "get_logger",
    "get_model_name",
    "truncate_text",
]
