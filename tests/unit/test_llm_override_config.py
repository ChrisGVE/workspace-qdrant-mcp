"""
Unit tests for LLM tool override configuration system.

Tests the manual override configuration that allows users to force
a specific LLM tool regardless of auto-detection results.
"""

import json
import os
from pathlib import Path

import pytest

from src.python.common.core.context_injection import (
    LLMOverrideConfig,
    LLMOverrideManager,
    LLMToolType,
    set_llm_override_cli,
    clear_llm_override_cli,
    show_llm_override_cli,
)


class TestLLMOverrideConfig:
    """Test LLMOverrideConfig dataclass."""

    def test_create_config(self):
        """Test creating an override configuration."""
        config = LLMOverrideConfig(
            enabled=True,
            tool_type=LLMToolType.CLAUDE_CODE,
            reason="Testing",
            set_at="2025-01-20T15:30:00",
            set_by="user",
        )

        assert config.enabled is True
        assert config.tool_type == LLMToolType.CLAUDE_CODE
        assert config.reason == "Testing"
        assert config.set_at == "2025-01-20T15:30:00"
        assert config.set_by == "user"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = LLMOverrideConfig(
            enabled=True,
            tool_type=LLMToolType.CLAUDE_CODE,
            reason="Testing",
            set_at="2025-01-20T15:30:00",
            set_by="user",
        )

        data = config.to_dict()

        assert data["enabled"] is True
        assert data["tool_type"] == "claude_code"
        assert data["reason"] == "Testing"
        assert data["set_at"] == "2025-01-20T15:30:00"
        assert data["set_by"] == "user"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "enabled": True,
            "tool_type": "claude_code",
            "reason": "Testing",
            "set_at": "2025-01-20T15:30:00",
            "set_by": "user",
        }

        config = LLMOverrideConfig.from_dict(data)

        assert config.enabled is True
        assert config.tool_type == LLMToolType.CLAUDE_CODE
        assert config.reason == "Testing"
        assert config.set_at == "2025-01-20T15:30:00"
        assert config.set_by == "user"

    def test_from_dict_invalid_tool_type(self):
        """Test creating config from dictionary with invalid tool type."""
        data = {
            "enabled": True,
            "tool_type": "invalid_tool",
            "reason": "Testing",
        }

        with pytest.raises(ValueError, match="Invalid tool_type"):
            LLMOverrideConfig.from_dict(data)

    def test_from_dict_none_tool_type(self):
        """Test creating config with None tool type."""
        data = {
            "enabled": False,
            "tool_type": None,
        }

        config = LLMOverrideConfig.from_dict(data)

        assert config.enabled is False
        assert config.tool_type is None

    def test_round_trip_serialization(self):
        """Test serialization round trip (to_dict -> from_dict)."""
        original = LLMOverrideConfig(
            enabled=True,
            tool_type=LLMToolType.GITHUB_COPILOT,
            reason="Testing Copilot formatter",
            set_at="2025-01-20T15:30:00",
            set_by="cli",
        )

        data = original.to_dict()
        restored = LLMOverrideConfig.from_dict(data)

        assert restored.enabled == original.enabled
        assert restored.tool_type == original.tool_type
        assert restored.reason == original.reason
        assert restored.set_at == original.set_at
        assert restored.set_by == original.set_by


class TestLLMOverrideManager:
    """Test LLMOverrideManager class."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory for tests."""
        config_dir = tmp_path / ".wqm"
        config_dir.mkdir(parents=True)
        return config_dir

    @pytest.fixture
    def mock_config_dir(self, temp_config_dir, monkeypatch):
        """Mock the default config directory."""
        monkeypatch.setattr(
            LLMOverrideManager, "DEFAULT_CONFIG_DIR", temp_config_dir
        )
        return temp_config_dir

    @pytest.fixture
    def clear_env_var(self, monkeypatch):
        """Clear the LLM_TOOL_OVERRIDE environment variable."""
        monkeypatch.delenv(LLMOverrideManager.ENV_VAR_NAME, raising=False)

    def test_set_override(self, mock_config_dir, clear_env_var):
        """Test setting an override configuration."""
        LLMOverrideManager.set_override(
            LLMToolType.CLAUDE_CODE,
            reason="Testing formatter",
            set_by="cli",
        )

        # Verify file was created
        config_path = mock_config_dir / LLMOverrideManager.CONFIG_FILE_NAME
        assert config_path.exists()

        # Verify content
        with open(config_path, "r") as f:
            data = json.load(f)

        assert data["enabled"] is True
        assert data["tool_type"] == "claude_code"
        assert data["reason"] == "Testing formatter"
        assert data["set_by"] == "cli"
        assert "set_at" in data

    def test_set_override_invalid_type(self, mock_config_dir, clear_env_var):
        """Test setting override with invalid tool type."""
        with pytest.raises(ValueError, match="tool_type must be LLMToolType"):
            LLMOverrideManager.set_override("invalid_tool")  # type: ignore

    def test_get_override_from_file(self, mock_config_dir, clear_env_var):
        """Test getting override from config file."""
        # Set override
        LLMOverrideManager.set_override(
            LLMToolType.CLAUDE_CODE,
            reason="Testing",
            set_by="user",
        )

        # Get override
        config = LLMOverrideManager.get_override()

        assert config is not None
        assert config.enabled is True
        assert config.tool_type == LLMToolType.CLAUDE_CODE
        assert config.reason == "Testing"
        assert config.set_by == "user"

    def test_get_override_from_env_var(self, mock_config_dir, monkeypatch):
        """Test getting override from environment variable."""
        # Set environment variable
        monkeypatch.setenv(LLMOverrideManager.ENV_VAR_NAME, "claude_code")

        # Get override (should come from env var, not file)
        config = LLMOverrideManager.get_override()

        assert config is not None
        assert config.enabled is True
        assert config.tool_type == LLMToolType.CLAUDE_CODE
        assert config.set_by == "environment"

    def test_get_override_env_var_priority(self, mock_config_dir, monkeypatch):
        """Test that environment variable takes priority over config file."""
        # Set override in file
        LLMOverrideManager.set_override(
            LLMToolType.GITHUB_COPILOT,
            reason="From file",
            set_by="user",
        )

        # Set different override in env var
        monkeypatch.setenv(LLMOverrideManager.ENV_VAR_NAME, "claude_code")

        # Get override - should be from env var
        config = LLMOverrideManager.get_override()

        assert config is not None
        assert config.tool_type == LLMToolType.CLAUDE_CODE
        assert config.set_by == "environment"

    def test_get_override_invalid_env_var(self, mock_config_dir, monkeypatch):
        """Test getting override with invalid environment variable."""
        # Set invalid environment variable
        monkeypatch.setenv(LLMOverrideManager.ENV_VAR_NAME, "invalid_tool")

        # Should return None (invalid env var is ignored)
        config = LLMOverrideManager.get_override()
        assert config is None

    def test_get_override_no_config(self, mock_config_dir, clear_env_var):
        """Test getting override when no configuration exists."""
        config = LLMOverrideManager.get_override()
        assert config is None

    def test_get_override_disabled_config(self, mock_config_dir, clear_env_var):
        """Test getting override when configuration is disabled."""
        # Write disabled config
        config_path = mock_config_dir / LLMOverrideManager.CONFIG_FILE_NAME
        with open(config_path, "w") as f:
            json.dump({"enabled": False, "tool_type": "claude_code"}, f)

        # Should return None (disabled config)
        config = LLMOverrideManager.get_override()
        assert config is None

    def test_is_override_active(self, mock_config_dir, clear_env_var):
        """Test checking if override is active."""
        # No override
        assert LLMOverrideManager.is_override_active() is False

        # Set override
        LLMOverrideManager.set_override(LLMToolType.CLAUDE_CODE)

        # Override active
        assert LLMOverrideManager.is_override_active() is True

    def test_clear_override(self, mock_config_dir, clear_env_var):
        """Test clearing override configuration."""
        # Set override
        LLMOverrideManager.set_override(LLMToolType.CLAUDE_CODE)
        assert LLMOverrideManager.is_override_active() is True

        # Clear override
        LLMOverrideManager.clear_override()

        # Override should be disabled
        config = LLMOverrideManager.get_override()
        assert config is None or config.enabled is False

    def test_read_corrupted_config_file(self, mock_config_dir, clear_env_var):
        """Test reading a corrupted config file."""
        # Write invalid JSON
        config_path = mock_config_dir / LLMOverrideManager.CONFIG_FILE_NAME
        with open(config_path, "w") as f:
            f.write("{ invalid json }")

        # Should return None (corrupted file is ignored)
        config = LLMOverrideManager.get_override()
        assert config is None

    def test_all_tool_types_serializable(self, mock_config_dir, clear_env_var):
        """Test that all LLMToolType values can be serialized/deserialized."""
        for tool_type in LLMToolType:
            # Set override
            LLMOverrideManager.set_override(tool_type)

            # Get override
            config = LLMOverrideManager.get_override()

            assert config is not None
            assert config.tool_type == tool_type

            # Clear for next iteration
            LLMOverrideManager.clear_override()


class TestIntegrationWithLLMToolDetector:
    """Test integration with LLMToolDetector."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory for tests."""
        config_dir = tmp_path / ".wqm"
        config_dir.mkdir(parents=True)
        return config_dir

    @pytest.fixture
    def mock_config_dir(self, temp_config_dir, monkeypatch):
        """Mock the default config directory."""
        monkeypatch.setattr(
            LLMOverrideManager, "DEFAULT_CONFIG_DIR", temp_config_dir
        )
        return temp_config_dir

    @pytest.fixture
    def clear_env_var(self, monkeypatch):
        """Clear the LLM_TOOL_OVERRIDE environment variable."""
        monkeypatch.delenv(LLMOverrideManager.ENV_VAR_NAME, raising=False)

    def test_override_takes_precedence(self, mock_config_dir, clear_env_var):
        """Test that override takes precedence over auto-detection."""
        from src.python.common.core.context_injection import LLMToolDetector

        # Set override
        LLMOverrideManager.set_override(
            LLMToolType.GITHUB_COPILOT,
            reason="Testing override precedence",
        )

        # Detect (should return override, not auto-detection)
        session = LLMToolDetector.detect()

        assert session.is_active is True
        assert session.tool_type == LLMToolType.GITHUB_COPILOT
        assert session.detection_method == "manual_override"
        assert "override_active" in session.metadata
        assert session.metadata["override_active"] is True
        assert session.metadata["override_reason"] == "Testing override precedence"

    def test_override_metadata_included(self, mock_config_dir, clear_env_var):
        """Test that override metadata is included in session."""
        from src.python.common.core.context_injection import LLMToolDetector

        # Set override
        LLMOverrideManager.set_override(
            LLMToolType.CURSOR,
            reason="Testing metadata",
            set_by="cli",
        )

        # Detect
        session = LLMToolDetector.detect()

        # Check metadata
        assert session.metadata["override_active"] is True
        assert session.metadata["override_reason"] == "Testing metadata"
        assert session.metadata["override_set_by"] == "cli"
        assert "override_set_at" in session.metadata

    def test_no_override_uses_auto_detection(self, mock_config_dir, clear_env_var):
        """Test that auto-detection works when no override is set."""
        from src.python.common.core.context_injection import LLMToolDetector

        # Don't set override
        session = LLMToolDetector.detect()

        # Should use auto-detection (may detect UNKNOWN if no LLM is active)
        assert session.detection_method != "manual_override"
        assert "override_active" not in session.metadata

    def test_clear_override_restores_auto_detection(
        self, mock_config_dir, clear_env_var
    ):
        """Test that clearing override restores auto-detection."""
        from src.python.common.core.context_injection import LLMToolDetector

        # Set override
        LLMOverrideManager.set_override(LLMToolType.CLAUDE_CODE)
        session1 = LLMToolDetector.detect()
        assert session1.detection_method == "manual_override"

        # Clear override
        LLMOverrideManager.clear_override()
        session2 = LLMToolDetector.detect()

        # Should use auto-detection now
        assert session2.detection_method != "manual_override"
        assert "override_active" not in session2.metadata

    def test_get_formatter_with_override(self, mock_config_dir, clear_env_var):
        """Test that get_formatter works with override."""
        from src.python.common.core.context_injection import LLMToolDetector

        # Set override to Claude Code
        LLMOverrideManager.set_override(LLMToolType.CLAUDE_CODE)

        # Get formatter
        formatter = LLMToolDetector.get_formatter()

        # Should return Claude adapter
        assert formatter is not None

    def test_override_with_all_tool_types(self, mock_config_dir, clear_env_var):
        """Test override works with all tool types."""
        from src.python.common.core.context_injection import LLMToolDetector

        for tool_type in LLMToolType:
            if tool_type == LLMToolType.UNKNOWN:
                continue  # Skip UNKNOWN

            # Set override
            LLMOverrideManager.set_override(tool_type)

            # Detect
            session = LLMToolDetector.detect()

            assert session.tool_type == tool_type
            assert session.detection_method == "manual_override"

            # Clear for next iteration
            LLMOverrideManager.clear_override()
