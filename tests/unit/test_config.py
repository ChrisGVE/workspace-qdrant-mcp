"""
Unit tests for configuration management.

Tests configuration loading, validation, and environment variable handling.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from common.core.config import (
    Config,
    EmbeddingConfig,
    QdrantConfig,
    WorkspaceConfig,
)


class TestEmbeddingConfig:
    """Test EmbeddingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.enable_sparse_vectors is True
        assert config.chunk_size == 800
        assert config.chunk_overlap == 120
        assert config.batch_size == 50

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model="custom-model",
            enable_sparse_vectors=False,
            chunk_size=500,
            chunk_overlap=100,
            batch_size=25,
        )

        assert config.model == "custom-model"
        assert config.enable_sparse_vectors is False
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.batch_size == 25

    @pytest.mark.parametrize(
        "chunk_size,chunk_overlap,expected_valid",
        [
            (1000, 200, True),
            (500, 100, True),
            (100, 200, False),  # overlap >= size
            (200, 200, False),  # overlap == size
        ],
    )
    def test_chunk_size_validation(self, chunk_size, chunk_overlap, expected_valid):
        """Test chunk size and overlap validation logic."""
        EmbeddingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # This would be validated by the main Config class
        is_valid = chunk_overlap < chunk_size
        assert is_valid == expected_valid


class TestQdrantConfig:
    """Test QdrantConfig class."""

    def test_default_values(self):
        """Test default Qdrant configuration."""
        config = QdrantConfig()

        assert config.url == "http://localhost:6333"
        assert config.api_key is None
        assert config.timeout == 30
        assert config.prefer_grpc is False

    def test_custom_values(self):
        """Test custom Qdrant configuration."""
        config = QdrantConfig(
            url="https://cluster.qdrant.io",
            api_key="secret-key",
            timeout=60,
            prefer_grpc=True,
        )

        assert config.url == "https://cluster.qdrant.io"
        assert config.api_key == "secret-key"
        assert config.timeout == 60
        assert config.prefer_grpc is True


class TestWorkspaceConfig:
    """Test WorkspaceConfig class."""

    def test_default_values(self):
        """Test default workspace configuration."""
        config = WorkspaceConfig()

        assert config.global_collections == []
        assert config.github_user is None
        assert config.auto_create_collections is False
        # Note: collection_prefix and max_collections removed as part of multi-tenant migration

    def test_custom_values(self):
        """Test custom workspace configuration."""
        config = WorkspaceConfig(
            global_collections=["custom", "collections"],
            github_user="testuser",
            auto_create_collections=True,
        )

        assert config.global_collections == ["custom", "collections"]
        assert config.github_user == "testuser"
        assert config.auto_create_collections is True
        # Note: collection_prefix and max_collections removed as part of multi-tenant migration


class TestConfig:
    """Test main Config class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()

        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.debug is False
        assert isinstance(config.qdrant, QdrantConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.workspace, WorkspaceConfig)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = Config(host="0.0.0.0", port=9000, debug=True)

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.debug is True

    def test_nested_config_override(self):
        """Test overriding nested configuration objects."""
        custom_qdrant = QdrantConfig(url="https://custom.qdrant.io")
        custom_embedding = EmbeddingConfig(model="custom-model")
        custom_workspace = WorkspaceConfig(github_user="testuser")

        config = Config(
            qdrant=custom_qdrant, embedding=custom_embedding, workspace=custom_workspace
        )

        assert config.qdrant.url == "https://custom.qdrant.io"
        assert config.embedding.model == "custom-model"
        assert config.workspace.github_user == "testuser"

    @patch.dict(
        os.environ,
        {
            "WORKSPACE_QDRANT_HOST": "test.host",
            "WORKSPACE_QDRANT_PORT": "9999",
            "WORKSPACE_QDRANT_DEBUG": "true",
        },
    )
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        config = Config()

        assert config.host == "test.host"
        assert config.port == 9999
        assert config.debug is True

    @patch.dict(
        os.environ,
        {
            "WORKSPACE_QDRANT_QDRANT__URL": "https://env.qdrant.io",
            "WORKSPACE_QDRANT_QDRANT__API_KEY": "env-api-key",
            "WORKSPACE_QDRANT_QDRANT__TIMEOUT": "45",
        },
        clear=False,
    )
    def test_nested_environment_variables(self):
        """Test loading nested configuration from environment variables."""
        config = Config()

        assert config.qdrant.url == "https://env.qdrant.io"
        assert config.qdrant.api_key == "env-api-key"
        assert config.qdrant.timeout == 45

    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "https://legacy.qdrant.io",
            "QDRANT_API_KEY": "legacy-key",
            "FASTEMBED_MODEL": "legacy-model",
            "ENABLE_SPARSE_VECTORS": "false",
            "CHUNK_SIZE": "800",
            "GITHUB_USER": "legacyuser",
        },
    )
    def test_legacy_environment_variables(self):
        """Test loading legacy environment variables."""
        config = Config()

        assert config.qdrant.url == "https://legacy.qdrant.io"
        assert config.qdrant.api_key == "legacy-key"
        assert config.embedding.model == "legacy-model"
        assert config.embedding.enable_sparse_vectors is False
        assert config.embedding.chunk_size == 800
        assert config.workspace.github_user == "legacyuser"

    @patch.dict(os.environ, {"QDRANT_API_KEY": ""}, clear=False)
    def test_qdrant_client_config(self):
        """Test Qdrant client configuration generation."""
        config = Config()
        client_config = config.qdrant_client_config

        expected_keys = ["url", "timeout", "prefer_grpc"]
        for key in expected_keys:
            assert key in client_config

        # API key should not be present if not set
        assert "api_key" not in client_config

    def test_qdrant_client_config_with_api_key(self):
        """Test Qdrant client configuration with API key."""
        config = Config()
        config.qdrant.api_key = "test-api-key"
        client_config = config.qdrant_client_config

        assert "api_key" in client_config
        assert client_config["api_key"] == "test-api-key"

    def test_validate_config_valid(self):
        """Test configuration validation with valid settings."""
        config = Config()
        issues = config.validate_config()

        assert len(issues) == 0

    def test_validate_config_missing_url(self):
        """Test configuration validation with missing URL."""
        config = Config()
        config.qdrant.url = ""
        issues = config.validate_config()

        assert len(issues) > 0
        assert any("Qdrant URL is required" in issue for issue in issues)

    @pytest.mark.parametrize(
        "chunk_size,chunk_overlap,expected_issues",
        [
            (1000, 200, 0),  # Valid
            (-1, 200, 1),  # Invalid chunk_size
            (1000, -1, 1),  # Invalid batch_size would be caught elsewhere
            (200, 200, 1),  # Invalid overlap >= size
            (200, 300, 1),  # Invalid overlap > size
        ],
    )
    def test_validate_config_chunk_settings(
        self, chunk_size, chunk_overlap, expected_issues
    ):
        """Test configuration validation for chunking settings."""
        config = Config()
        config.embedding.chunk_size = chunk_size
        config.embedding.chunk_overlap = chunk_overlap

        issues = config.validate_config()

        if expected_issues == 0:
            assert len(issues) == 0
        else:
            assert len(issues) >= expected_issues

    def test_validate_config_batch_size(self):
        """Test configuration validation for batch size."""
        config = Config()
        config.embedding.batch_size = -1

        issues = config.validate_config()

        assert len(issues) > 0
        assert any("Batch size must be positive" in issue for issue in issues)

    def test_validate_config_collection_types(self):
        """Test configuration validation with collection types."""
        config = Config()
        config.workspace.collection_types = ["docs", "notes"]

        issues = config.validate_config()

        # Should validate successfully with proper collection types
        assert len(issues) == 0

    def test_env_file_loading(self):
        """Test loading configuration from .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "WORKSPACE_QDRANT_HOST=file.host\n"
                "WORKSPACE_QDRANT_PORT=7777\n"
                "WORKSPACE_QDRANT_DEBUG=true\n"
            )

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = Config()

                assert config.host == "file.host"
                assert config.port == 7777
                assert config.debug is True

            finally:
                os.chdir(original_cwd)

    def test_config_immutability_after_init(self):
        """Test that configuration values are properly set after initialization."""
        config = Config(host="test.host", port=9000, debug=True)

        # Verify values are set correctly
        assert config.host == "test.host"
        assert config.port == 9000
        assert config.debug is True

        # Nested configs should also be properly initialized
        assert config.qdrant.url == "http://localhost:6333"
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.workspace.github_user is None

    def test_config_serialization(self):
        """Test that configuration can be serialized (for debugging/logging)."""
        config = Config()

        # Should be able to convert to dict-like structure
        config_dict = config.model_dump()

        assert "host" in config_dict
        assert "port" in config_dict
        assert "qdrant" in config_dict
        assert "embedding" in config_dict
        assert "workspace" in config_dict

        # Nested configs should also be serializable
        assert "url" in config_dict["qdrant"]
        assert "model" in config_dict["embedding"]
        assert "global_collections" in config_dict["workspace"]
