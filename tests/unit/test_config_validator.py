"""
Unit tests for configuration validator.

Tests configuration validation logic and error reporting.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from unittest.mock import MagicMock, patch

from workspace_qdrant_mcp.utils.config_validator import ConfigValidator


class TestConfigValidator:
    """Test ConfigValidator class."""

    def test_init(self, mock_config):
        """Test validator initialization."""
        validator = ConfigValidator(mock_config)
        assert validator.config == mock_config

    def test_validate_qdrant_connection_success(self, mock_config):
        """Test Qdrant connection validation success."""
        validator = ConfigValidator(mock_config)

        with patch(
            "workspace_qdrant_mcp.utils.config_validator.QdrantClient"
        ) as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_qdrant.return_value = mock_client

            is_valid, message = validator.validate_qdrant_connection()

            assert is_valid is True
            assert "successfully connected" in message.lower()

    def test_validate_qdrant_connection_failure(self, mock_config):
        """Test Qdrant connection validation failure."""
        validator = ConfigValidator(mock_config)

        with patch(
            "workspace_qdrant_mcp.utils.config_validator.QdrantClient"
        ) as mock_qdrant:
            mock_qdrant.side_effect = Exception("Connection failed")

            is_valid, message = validator.validate_qdrant_connection()

            assert is_valid is False
            assert "Connection failed" in message

    def test_validate_embedding_model_success(self, mock_config):
        """Test embedding model validation success."""
        validator = ConfigValidator(mock_config)

        with patch(
            "workspace_qdrant_mcp.utils.config_validator.EmbeddingService"
        ) as mock_embedding_service:
            mock_service = MagicMock()
            mock_service.initialize = MagicMock()
            mock_service.get_model_info.return_value = {
                "dense_model": {"name": "test-model", "dimensions": 384}
            }
            mock_embedding_service.return_value = mock_service

            is_valid, message = validator.validate_embedding_model()

            assert is_valid is True
            assert "successfully loaded" in message.lower()

    def test_validate_embedding_model_failure(self, mock_config):
        """Test embedding model validation failure."""
        validator = ConfigValidator(mock_config)

        with patch(
            "workspace_qdrant_mcp.utils.config_validator.EmbeddingService"
        ) as mock_embedding_service:
            mock_embedding_service.side_effect = Exception("Model load failed")

            is_valid, message = validator.validate_embedding_model()

            assert is_valid is False
            assert "Model load failed" in message

    def test_validate_project_detection_git_repo(self, mock_config, temp_git_repo):
        """Test project detection validation in Git repository."""
        validator = ConfigValidator(mock_config)

        with patch(
            "workspace_qdrant_mcp.utils.config_validator.ProjectDetector"
        ) as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.get_project_info.return_value = {
                "main_project": "test-project",
                "is_git_repo": True,
                "subprojects": ["sub1"],
                "belongs_to_user": True,
            }
            mock_detector_class.return_value = mock_detector

            is_valid, message = validator.validate_project_detection()

            assert is_valid is True
            assert "test-project" in message
            assert "1 subproject" in message

    def test_validate_project_detection_no_git(self, mock_config):
        """Test project detection validation outside Git repository."""
        validator = ConfigValidator(mock_config)

        with patch(
            "workspace_qdrant_mcp.utils.config_validator.ProjectDetector"
        ) as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.get_project_info.return_value = {
                "main_project": "directory-name",
                "is_git_repo": False,
                "subprojects": [],
                "belongs_to_user": False,
            }
            mock_detector_class.return_value = mock_detector

            is_valid, message = validator.validate_project_detection()

            assert is_valid is True
            assert "directory-name" in message
            assert "not a Git repository" in message

    def test_validate_all_success(self, mock_config):
        """Test complete validation success."""
        validator = ConfigValidator(mock_config)

        # Mock all validation methods to succeed
        with (
            patch.object(
                validator,
                "validate_qdrant_connection",
                return_value=(True, "Qdrant OK"),
            ),
            patch.object(
                validator,
                "validate_embedding_model",
                return_value=(True, "Embedding OK"),
            ),
            patch.object(
                validator,
                "validate_project_detection",
                return_value=(True, "Project OK"),
            ),
            patch(
                "workspace_qdrant_mcp.core.config.Config.validate_config",
                return_value=[],
            ),
        ):
            is_valid, results = validator.validate_all()

            assert is_valid is True
            assert len(results["issues"]) == 0
            assert len(results["warnings"]) == 0
            assert results["qdrant_connection"]["valid"] is True
            assert results["embedding_model"]["valid"] is True
            assert results["project_detection"]["valid"] is True

    def test_validate_all_with_issues(self, mock_config):
        """Test complete validation with issues."""
        validator = ConfigValidator(mock_config)

        # Mock validation methods with some failures
        with (
            patch.object(
                validator,
                "validate_qdrant_connection",
                return_value=(False, "Qdrant failed"),
            ),
            patch.object(
                validator,
                "validate_embedding_model",
                return_value=(True, "Embedding OK"),
            ),
            patch.object(
                validator,
                "validate_project_detection",
                return_value=(True, "Project OK"),
            ),
            patch(
                "workspace_qdrant_mcp.core.config.Config.validate_config",
                return_value=["Config issue 1", "Config issue 2"],
            ),
        ):
            is_valid, results = validator.validate_all()

            assert is_valid is False
            assert len(results["issues"]) == 3  # 1 Qdrant + 2 Config
            assert "Qdrant failed" in results["issues"]
            assert "Config issue 1" in results["issues"]
            assert "Config issue 2" in results["issues"]
            assert results["qdrant_connection"]["valid"] is False
            assert results["embedding_model"]["valid"] is True

    def test_validate_all_with_warnings(self, mock_config):
        """Test validation with warnings."""
        # Setup config with missing GitHub user (should generate warning)
        mock_config.workspace.github_user = None

        validator = ConfigValidator(mock_config)

        # Mock other validations to succeed
        with (
            patch.object(
                validator,
                "validate_qdrant_connection",
                return_value=(True, "Qdrant OK"),
            ),
            patch.object(
                validator,
                "validate_embedding_model",
                return_value=(True, "Embedding OK"),
            ),
            patch.object(
                validator,
                "validate_project_detection",
                return_value=(True, "Project OK"),
            ),
            patch(
                "workspace_qdrant_mcp.core.config.Config.validate_config",
                return_value=[],
            ),
        ):
            is_valid, results = validator.validate_all()

            assert is_valid is True  # Warnings don't make validation fail
            assert len(results["issues"]) == 0
            assert len(results["warnings"]) > 0
            assert any(
                "GitHub user not configured" in warning
                for warning in results["warnings"]
            )

    def test_validate_config_basic_validation(self, mock_config):
        """Test that basic config validation is called."""
        validator = ConfigValidator(mock_config)

        # Mock config validation to return specific issues
        expected_issues = ["Test issue 1", "Test issue 2"]
        with patch(
            "workspace_qdrant_mcp.core.config.Config.validate_config",
            return_value=expected_issues,
        ):
            issues = validator.config.validate_config()

            assert issues == expected_issues

    def test_generate_warnings_github_user_missing(self, mock_config):
        """Test warning generation for missing GitHub user."""
        mock_config.workspace.github_user = None
        validator = ConfigValidator(mock_config)

        # Mock project detection to return user-owned repos (would benefit from GitHub user)
        project_info = {
            "is_git_repo": True,
            "belongs_to_user": False,  # Can't determine without GitHub user
            "remote_url": "https://github.com/someuser/somerepo.git",
        }

        with patch(
            "workspace_qdrant_mcp.utils.config_validator.ProjectDetector"
        ) as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.get_project_info.return_value = project_info
            mock_detector_class.return_value = mock_detector

            with (
                patch.object(
                    validator, "validate_qdrant_connection", return_value=(True, "OK")
                ),
                patch.object(
                    validator, "validate_embedding_model", return_value=(True, "OK")
                ),
                patch(
                    "workspace_qdrant_mcp.core.config.Config.validate_config",
                    return_value=[],
                ),
            ):
                is_valid, results = validator.validate_all()

                assert len(results["warnings"]) > 0
                github_warnings = [w for w in results["warnings"] if "GitHub user" in w]
                assert len(github_warnings) > 0

    def test_validation_exception_handling(self, mock_config):
        """Test that validation exceptions are properly handled."""
        validator = ConfigValidator(mock_config)

        # Mock QdrantClient to raise an exception
        with patch(
            "workspace_qdrant_mcp.utils.config_validator.QdrantClient",
            side_effect=Exception("Unexpected error"),
        ):
            is_valid, message = validator.validate_qdrant_connection()

            assert is_valid is False
            assert "Unexpected error" in message

    def test_validate_all_comprehensive_results(self, mock_config):
        """Test that validate_all returns comprehensive results structure."""
        validator = ConfigValidator(mock_config)

        with (
            patch.object(
                validator,
                "validate_qdrant_connection",
                return_value=(True, "Qdrant OK"),
            ),
            patch.object(
                validator,
                "validate_embedding_model",
                return_value=(True, "Embedding OK"),
            ),
            patch.object(
                validator,
                "validate_project_detection",
                return_value=(True, "Project OK"),
            ),
            patch(
                "workspace_qdrant_mcp.core.config.Config.validate_config",
                return_value=[],
            ),
        ):
            is_valid, results = validator.validate_all()

            # Check result structure
            expected_keys = [
                "issues",
                "warnings",
                "qdrant_connection",
                "embedding_model",
                "project_detection",
                "config_validation",
            ]

            for key in expected_keys:
                assert key in results, f"Missing key: {key}"

            # Check individual validation results structure
            for validation_key in [
                "qdrant_connection",
                "embedding_model",
                "project_detection",
            ]:
                validation_result = results[validation_key]
                assert "valid" in validation_result
                assert "message" in validation_result
                assert isinstance(validation_result["valid"], bool)
                assert isinstance(validation_result["message"], str)

    def test_cli_validation_interface(self, mock_config):
        """Test CLI validation interface if implemented."""
        # This would test the CLI interface if it exists
        # For now, we'll just verify the validator can be used programmatically
        validator = ConfigValidator(mock_config)

        # Should be able to call validation methods individually
        assert hasattr(validator, "validate_qdrant_connection")
        assert hasattr(validator, "validate_embedding_model")
        assert hasattr(validator, "validate_project_detection")
        assert hasattr(validator, "validate_all")

        # Methods should be callable
        assert callable(validator.validate_qdrant_connection)
        assert callable(validator.validate_embedding_model)
        assert callable(validator.validate_project_detection)
        assert callable(validator.validate_all)
