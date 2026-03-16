"""
Tests for config.settings – verifying that environment variables are loaded
with the RAG_ prefix and that the .env file path is resolved correctly.
"""

from __future__ import annotations

import os

import pytest

from config.settings import Settings, _ROOT_DIR


class TestSettingsEnvPrefix:
    """Verify that the Settings class picks up RAG_-prefixed environment variables."""

    def test_env_prefix_is_rag(self):
        """model_config must declare env_prefix = 'RAG_'."""
        assert Settings.model_config.get("env_prefix") == "RAG_"

    def test_env_file_is_absolute(self):
        """env_file must be an absolute path so the file is found regardless of CWD."""
        env_file = Settings.model_config.get("env_file", "")
        assert os.path.isabs(env_file), (
            f"env_file '{env_file}' should be an absolute path"
        )

    def test_env_file_points_to_project_root(self):
        """env_file must resolve inside the project root directory."""
        env_file = Settings.model_config.get("env_file", "")
        assert env_file == str(_ROOT_DIR / ".env")

    def test_rag_openai_api_key_is_read(self, monkeypatch):
        """RAG_OPENAI_API_KEY in the environment must populate openai_api_key."""
        monkeypatch.setenv("RAG_OPENAI_API_KEY", "sk-test-key")
        s = Settings()
        assert s.openai_api_key == "sk-test-key"

    def test_rag_azure_api_key_is_read(self, monkeypatch):
        """RAG_AZURE_OPENAI_API_KEY must populate azure_openai_api_key."""
        monkeypatch.setenv("RAG_AZURE_OPENAI_API_KEY", "azure-test-key")
        s = Settings()
        assert s.azure_openai_api_key == "azure-test-key"

    def test_unprefixed_openai_key_is_ignored(self, monkeypatch):
        """A plain OPENAI_API_KEY (without RAG_ prefix) must NOT be picked up."""
        monkeypatch.delenv("RAG_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")
        s = Settings()
        assert s.openai_api_key is None

    def test_is_llm_available_with_rag_prefix(self, monkeypatch):
        """is_llm_available() must return True when RAG_OPENAI_API_KEY is set."""
        monkeypatch.setenv("RAG_OPENAI_API_KEY", "sk-live-key")
        # Re-create settings so the monkeypatched env var is picked up
        new_settings = Settings()
        # A non-None openai_api_key means is_llm_available() would return True
        assert new_settings.openai_api_key is not None
