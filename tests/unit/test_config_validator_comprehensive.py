"""
Comprehensive unit tests for configuration validation utilities.

This module provides 100% test coverage for the config validation system,
including all validation methods, error conditions, and edge cases.

Test coverage:
- ConfigValidator: validation methods, connection testing, configuration analysis
- CLI commands and setup guidance functionality
- Error handling and edge cases for all methods
- Async patterns and comprehensive mocking
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock, call
import pytest
from typer.testing import CliRunner

# Ensure proper imports from the project structure
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

from common.utils.config_validator import ConfigValidator, validate_config_cmd, validate_config_cli
from common.core.config import Config
from common.core.embeddings import EmbeddingService
from common.utils.project_detection import ProjectDetector


class TestConfigValidator:
    """Comprehensive tests for ConfigValidator class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_config = Mock(spec=Config)
        self.mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        self.mock_config.qdrant.url = "http://localhost:6333"
        self.mock_config.qdrant.timeout = 30
        self.mock_config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        self.mock_config.embedding.chunk_size = 1000
        self.mock_config.embedding.chunk_overlap = 100
        self.mock_config.embedding.batch_size = 32
        self.mock_config.workspace.github_user = "testuser"
        self.mock_config.workspace.global_collections = ["docs", "reference"]
        self.mock_config.host = "127.0.0.1"
        self.mock_config.port = 8000
        self.mock_config.debug = False
        self.mock_config.validate_config.return_value = []

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_basic(self):
        """Test basic initialization of ConfigValidator."""
        validator = ConfigValidator(self.mock_config)

        assert validator.config == self.mock_config
        assert validator.issues == []
        assert validator.warnings == []
        assert validator.suggestions == []

    @patch('common.utils.config_validator.Config')
    def test_init_no_config(self, mock_config_class):
        """Test initialization without providing config."""
        mock_config_instance = Mock()
        mock_config_class.return_value = mock_config_instance

        validator = ConfigValidator()

        assert validator.config == mock_config_instance
        mock_config_class.assert_called_once_with()

    @patch('common.utils.config_validator.QdrantClient')
    def test_validate_qdrant_connection_success(self, mock_qdrant_client):
        """Test successful Qdrant connection validation."""
        mock_client = Mock()
        mock_client.get_collections.return_value = []
        mock_qdrant_client.return_value = mock_client

        validator = ConfigValidator(self.mock_config)

        with patch('common.utils.config_validator.suppress_qdrant_ssl_warnings'):
            is_valid, message = validator.validate_qdrant_connection()

            assert is_valid == True
            assert "successfully connected" in message
            mock_client.get_collections.assert_called_once()
            mock_client.close.assert_called_once()

    @patch('common.utils.config_validator.QdrantClient')
    def test_validate_qdrant_connection_failure(self, mock_qdrant_client):
        """Test Qdrant connection validation failure."""
        mock_qdrant_client.side_effect = Exception("Connection failed")

        validator = ConfigValidator(self.mock_config)

        with patch('common.utils.config_validator.suppress_qdrant_ssl_warnings'):
            is_valid, message = validator.validate_qdrant_connection()

            assert is_valid == False
            assert "Connection failed" in message

    @patch('common.utils.config_validator.EmbeddingService')
    def test_validate_embedding_model_success(self, mock_embedding_service):
        """Test successful embedding model validation."""
        mock_service = Mock()
        mock_service.get_model_info.return_value = {
            "dense_model": {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384
            }
        }
        mock_embedding_service.return_value = mock_service

        validator = ConfigValidator(self.mock_config)
        is_valid, message = validator.validate_embedding_model()

        assert is_valid == True
        assert "all-MiniLM-L6-v2" in message
        assert "384D" in message

    @patch('common.utils.config_validator.EmbeddingService')
    def test_validate_embedding_model_failure(self, mock_embedding_service):
        """Test embedding model validation failure."""
        mock_embedding_service.side_effect = Exception("Model not found")

        validator = ConfigValidator(self.mock_config)
        is_valid, message = validator.validate_embedding_model()

        assert is_valid == False
        assert "Model not found" in message

    @patch('common.utils.config_validator.ProjectDetector')
    def test_validate_project_detection_success(self, mock_project_detector):
        """Test successful project detection validation."""
        mock_detector = Mock()
        mock_detector.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": ["sub1", "sub2"],
            "is_git_repo": True
        }
        mock_project_detector.return_value = mock_detector

        validator = ConfigValidator(self.mock_config)
        is_valid, message = validator.validate_project_detection()

        assert is_valid == True
        assert "test-project" in message
        assert "2 subprojects" in message

    @patch('common.utils.config_validator.ProjectDetector')
    def test_validate_project_detection_no_git(self, mock_project_detector):
        """Test project detection validation for non-Git directory."""
        mock_detector = Mock()
        mock_detector.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": [],
            "is_git_repo": False
        }
        mock_project_detector.return_value = mock_detector

        validator = ConfigValidator(self.mock_config)
        is_valid, message = validator.validate_project_detection()

        assert is_valid == True
        assert "test-project" in message
        assert "not a Git repository" in message

    @patch('common.utils.config_validator.ProjectDetector')
    def test_validate_project_detection_single_subproject(self, mock_project_detector):
        """Test project detection validation with single subproject."""
        mock_detector = Mock()
        mock_detector.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": ["sub1"],
            "is_git_repo": True
        }
        mock_project_detector.return_value = mock_detector

        validator = ConfigValidator(self.mock_config)
        is_valid, message = validator.validate_project_detection()

        assert is_valid == True
        assert "1 subproject" in message  # Singular form

    @patch('common.utils.config_validator.ProjectDetector')
    def test_validate_project_detection_failure(self, mock_project_detector):
        """Test project detection validation failure."""
        mock_project_detector.side_effect = Exception("Detection failed")

        validator = ConfigValidator(self.mock_config)
        is_valid, message = validator.validate_project_detection()

        assert is_valid == False
        assert "Detection failed" in message

    def test_validate_all_success(self):
        """Test comprehensive validation with all checks passing."""
        validator = ConfigValidator(self.mock_config)

        with patch.object(validator, 'validate_qdrant_connection', return_value=(True, "Qdrant OK")), \
             patch.object(validator, 'validate_embedding_model', return_value=(True, "Model OK")), \
             patch.object(validator, 'validate_project_detection', return_value=(True, "Project OK")), \
             patch.object(validator, '_generate_warnings', return_value=[]):

            is_valid, results = validator.validate_all()

            assert is_valid == True
            assert len(results['issues']) == 0
            assert results['qdrant_connection']['valid'] == True
            assert results['embedding_model']['valid'] == True
            assert results['project_detection']['valid'] == True
            assert results['config_validation']['valid'] == True

    def test_validate_all_with_issues(self):
        """Test comprehensive validation with issues."""
        self.mock_config.validate_config.return_value = ["Config issue"]
        validator = ConfigValidator(self.mock_config)

        with patch.object(validator, 'validate_qdrant_connection', return_value=(False, "Qdrant failed")), \
             patch.object(validator, 'validate_embedding_model', return_value=(True, "Model OK")), \
             patch.object(validator, 'validate_project_detection', return_value=(True, "Project OK")), \
             patch.object(validator, '_generate_warnings', return_value=["Warning"]):

            is_valid, results = validator.validate_all()

            assert is_valid == False
            assert len(results['issues']) == 2  # Qdrant + Config issue
            assert "Qdrant failed" in results['issues']
            assert "Config issue" in results['issues']
            assert len(results['warnings']) == 1

    def test_generate_warnings_no_github_user(self):
        """Test warning generation when GitHub user not configured."""
        self.mock_config.workspace.github_user = None
        validator = ConfigValidator(self.mock_config)

        mock_detector = Mock()
        mock_detector.get_project_info.return_value = {
            "is_git_repo": True,
            "remote_url": "https://github.com/user/repo.git"
        }

        with patch('common.utils.config_validator.ProjectDetector', return_value=mock_detector):
            warnings = validator._generate_warnings()

            assert len(warnings) == 1
            assert "GitHub user not configured" in warnings[0]

    def test_generate_warnings_github_user_configured(self):
        """Test warning generation when GitHub user is configured."""
        validator = ConfigValidator(self.mock_config)

        mock_detector = Mock()
        mock_detector.get_project_info.return_value = {
            "is_git_repo": True,
            "remote_url": "https://github.com/user/repo.git"
        }

        with patch('common.utils.config_validator.ProjectDetector', return_value=mock_detector):
            warnings = validator._generate_warnings()

            assert len(warnings) == 0

    def test_generate_warnings_no_git_repo(self):
        """Test warning generation for non-Git repository."""
        self.mock_config.workspace.github_user = None
        validator = ConfigValidator(self.mock_config)

        mock_detector = Mock()
        mock_detector.get_project_info.return_value = {
            "is_git_repo": False,
            "remote_url": None
        }

        with patch('common.utils.config_validator.ProjectDetector', return_value=mock_detector):
            warnings = validator._generate_warnings()

            assert len(warnings) == 0

    def test_generate_warnings_detection_exception(self):
        """Test warning generation with project detection exception."""
        self.mock_config.workspace.github_user = None
        validator = ConfigValidator(self.mock_config)

        with patch('common.utils.config_validator.ProjectDetector', side_effect=Exception("Detection error")):
            warnings = validator._generate_warnings()

            # Should not raise exception, just ignore the error
            assert isinstance(warnings, list)

    def test_validate_qdrant_config_valid_url(self):
        """Test Qdrant configuration validation with valid URL."""
        validator = ConfigValidator(self.mock_config)

        with patch.object(validator, '_test_qdrant_connection', return_value=True):
            validator._validate_qdrant_config()

            assert len(validator.issues) == 0

    def test_validate_qdrant_config_invalid_url_format(self):
        """Test Qdrant configuration validation with invalid URL format."""
        self.mock_config.qdrant.url = "invalid-url"
        validator = ConfigValidator(self.mock_config)

        validator._validate_qdrant_config()

        assert any("Invalid Qdrant URL format" in issue for issue in validator.issues)

    def test_validate_qdrant_config_missing_scheme(self):
        """Test Qdrant configuration validation with missing scheme."""
        self.mock_config.qdrant.url = "localhost:6333"
        validator = ConfigValidator(self.mock_config)

        validator._validate_qdrant_config()

        assert any("must include scheme and hostname" in issue for issue in validator.issues)

    def test_validate_qdrant_config_invalid_scheme(self):
        """Test Qdrant configuration validation with invalid scheme."""
        self.mock_config.qdrant.url = "ftp://localhost:6333"
        validator = ConfigValidator(self.mock_config)

        validator._validate_qdrant_config()

        assert any("scheme must be http, https, or grpc" in issue for issue in validator.issues)

    def test_validate_qdrant_config_negative_timeout(self):
        """Test Qdrant configuration validation with negative timeout."""
        self.mock_config.qdrant.timeout = -5
        validator = ConfigValidator(self.mock_config)

        validator._validate_qdrant_config()

        assert any("timeout must be positive" in issue for issue in validator.issues)

    def test_validate_qdrant_config_short_timeout_warning(self):
        """Test Qdrant configuration validation with short timeout."""
        self.mock_config.qdrant.timeout = 3
        validator = ConfigValidator(self.mock_config)

        with patch.object(validator, '_test_qdrant_connection', return_value=True):
            validator._validate_qdrant_config()

            assert any("timeout is very short" in warning for warning in validator.warnings)

    def test_validate_qdrant_config_connection_failure(self):
        """Test Qdrant configuration validation with connection failure."""
        validator = ConfigValidator(self.mock_config)

        with patch.object(validator, '_test_qdrant_connection', return_value=False):
            validator._validate_qdrant_config()

            assert any("Cannot connect to Qdrant instance" in issue for issue in validator.issues)
            assert any("Ensure Qdrant is running" in suggestion for suggestion in validator.suggestions)

    def test_validate_embedding_config_supported_model(self):
        """Test embedding configuration validation with supported model."""
        validator = ConfigValidator(self.mock_config)
        validator._validate_embedding_config()

        # Should not add warnings for supported model
        model_warnings = [w for w in validator.warnings if "may not be optimized" in w]
        assert len(model_warnings) == 0

    def test_validate_embedding_config_unsupported_model(self):
        """Test embedding configuration validation with unsupported model."""
        self.mock_config.embedding.model = "custom-model"
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("may not be optimized" in warning for warning in validator.warnings)

    def test_validate_embedding_config_negative_chunk_size(self):
        """Test embedding configuration validation with negative chunk size."""
        self.mock_config.embedding.chunk_size = -100
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("Chunk size must be positive" in issue for issue in validator.issues)

    def test_validate_embedding_config_large_chunk_size(self):
        """Test embedding configuration validation with large chunk size."""
        self.mock_config.embedding.chunk_size = 3000
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("Large chunk size" in warning for warning in validator.warnings)

    def test_validate_embedding_config_negative_overlap(self):
        """Test embedding configuration validation with negative overlap."""
        self.mock_config.embedding.chunk_overlap = -50
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("Chunk overlap cannot be negative" in issue for issue in validator.issues)

    def test_validate_embedding_config_overlap_too_large(self):
        """Test embedding configuration validation with overlap >= chunk size."""
        self.mock_config.embedding.chunk_overlap = 1000  # Same as chunk_size
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("overlap must be less than chunk size" in issue for issue in validator.issues)

    def test_validate_embedding_config_high_overlap_warning(self):
        """Test embedding configuration validation with high overlap."""
        self.mock_config.embedding.chunk_overlap = 600  # > 50% of 1000
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("High chunk overlap" in warning for warning in validator.warnings)

    def test_validate_embedding_config_negative_batch_size(self):
        """Test embedding configuration validation with negative batch size."""
        self.mock_config.embedding.batch_size = -10
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("Batch size must be positive" in issue for issue in validator.issues)

    def test_validate_embedding_config_large_batch_size(self):
        """Test embedding configuration validation with large batch size."""
        self.mock_config.embedding.batch_size = 150
        validator = ConfigValidator(self.mock_config)

        validator._validate_embedding_config()

        assert any("Large batch size" in warning for warning in validator.warnings)

    def test_validate_workspace_config_no_global_collections(self):
        """Test workspace configuration validation with no global collections."""
        self.mock_config.workspace.global_collections = []
        validator = ConfigValidator(self.mock_config)

        validator._validate_workspace_config()

        assert any("No global collections configured" in warning for warning in validator.warnings)

    def test_validate_workspace_config_invalid_collection_name(self):
        """Test workspace configuration validation with invalid collection names."""
        self.mock_config.workspace.global_collections = ["invalid@name", "valid-name"]
        validator = ConfigValidator(self.mock_config)

        validator._validate_workspace_config()

        assert any("Invalid collection name 'invalid@name'" in issue for issue in validator.issues)

    def test_validate_workspace_config_valid_collection_names(self):
        """Test workspace configuration validation with valid collection names."""
        self.mock_config.workspace.global_collections = ["docs", "reference-data", "test_collection"]
        validator = ConfigValidator(self.mock_config)

        validator._validate_workspace_config()

        # Should not add any issues for valid names
        collection_issues = [i for i in validator.issues if "Invalid collection name" in i]
        assert len(collection_issues) == 0

    def test_validate_workspace_config_invalid_github_user(self):
        """Test workspace configuration validation with invalid GitHub user."""
        self.mock_config.workspace.github_user = "invalid@user"
        validator = ConfigValidator(self.mock_config)

        validator._validate_workspace_config()

        assert any("GitHub user must contain only alphanumeric" in issue for issue in validator.issues)

    def test_validate_workspace_config_valid_github_user(self):
        """Test workspace configuration validation with valid GitHub user."""
        self.mock_config.workspace.github_user = "valid-user123"
        validator = ConfigValidator(self.mock_config)

        validator._validate_workspace_config()

        # Should not add any issues for valid GitHub user
        github_issues = [i for i in validator.issues if "GitHub user must contain" in i]
        assert len(github_issues) == 0

    def test_validate_workspace_config_no_github_user(self):
        """Test workspace configuration validation with no GitHub user."""
        self.mock_config.workspace.github_user = None
        validator = ConfigValidator(self.mock_config)

        validator._validate_workspace_config()

        assert any("Consider setting GITHUB_USER" in suggestion for suggestion in validator.suggestions)

    def test_validate_server_config_empty_host(self):
        """Test server configuration validation with empty host."""
        self.mock_config.host = ""
        validator = ConfigValidator(self.mock_config)

        validator._validate_server_config()

        assert any("Server host cannot be empty" in issue for issue in validator.issues)

    def test_validate_server_config_invalid_port_low(self):
        """Test server configuration validation with port too low."""
        self.mock_config.port = 0
        validator = ConfigValidator(self.mock_config)

        validator._validate_server_config()

        assert any("port must be between 1 and 65535" in issue for issue in validator.issues)

    def test_validate_server_config_invalid_port_high(self):
        """Test server configuration validation with port too high."""
        self.mock_config.port = 70000
        validator = ConfigValidator(self.mock_config)

        validator._validate_server_config()

        assert any("port must be between 1 and 65535" in issue for issue in validator.issues)

    def test_validate_server_config_privileged_port_localhost(self):
        """Test server configuration validation with privileged port on localhost."""
        self.mock_config.port = 80
        self.mock_config.host = "127.0.0.1"
        validator = ConfigValidator(self.mock_config)

        validator._validate_server_config()

        assert any("privileged port" in warning for warning in validator.warnings)

    def test_validate_server_config_privileged_port_external(self):
        """Test server configuration validation with privileged port on external interface."""
        self.mock_config.port = 80
        self.mock_config.host = "192.168.1.100"
        validator = ConfigValidator(self.mock_config)

        validator._validate_server_config()

        # Should not warn about privileged ports on external interfaces
        privileged_warnings = [w for w in validator.warnings if "privileged port" in w]
        assert len(privileged_warnings) == 0

    def test_validate_environment_missing_required_vars(self):
        """Test environment validation with missing required variables."""
        validator = ConfigValidator(self.mock_config)

        with patch.dict(os.environ, {}, clear=True):
            validator._validate_environment()

            assert any("Consider setting QDRANT_URL" in suggestion for suggestion in validator.suggestions)

    def test_validate_environment_env_file_missing(self):
        """Test environment validation with missing .env file."""
        validator = ConfigValidator(self.mock_config)

        with patch('pathlib.Path.exists', return_value=False):
            validator._validate_environment()

            suggestions = [s for s in validator.suggestions if ".env" in s]
            assert len(suggestions) > 0

    def test_validate_environment_with_example_file(self):
        """Test environment validation with .env.example present."""
        validator = ConfigValidator(self.mock_config)

        def mock_exists(path):
            return str(path).endswith('.env.example')

        with patch('pathlib.Path.exists', side_effect=mock_exists):
            validator._validate_environment()

            assert any("Copy .env.example to .env" in suggestion for suggestion in validator.suggestions)

    def test_validate_environment_debug_mode_conflict(self):
        """Test environment validation with debug mode conflict."""
        validator = ConfigValidator(self.mock_config)

        with patch.dict(os.environ, {"WORKSPACE_QDRANT_DEBUG": "true"}):
            validator._validate_environment()

            assert any("Debug mode set in environment" in warning for warning in validator.warnings)

    @patch('common.utils.config_validator.QdrantClient')
    def test_test_qdrant_connection_success(self, mock_qdrant_client):
        """Test successful Qdrant connection test."""
        mock_client = Mock()
        mock_client.get_collections.return_value = []
        mock_qdrant_client.return_value = mock_client

        validator = ConfigValidator(self.mock_config)

        with patch('common.utils.config_validator.suppress_qdrant_ssl_warnings'):
            result = validator._test_qdrant_connection()

            assert result == True

    @patch('common.utils.config_validator.QdrantClient')
    def test_test_qdrant_connection_failure(self, mock_qdrant_client):
        """Test failed Qdrant connection test."""
        mock_qdrant_client.side_effect = Exception("Connection error")

        validator = ConfigValidator(self.mock_config)

        with patch('common.utils.config_validator.suppress_qdrant_ssl_warnings'):
            result = validator._test_qdrant_connection()

            assert result == False

    def test_get_setup_guide(self):
        """Test setup guide generation."""
        validator = ConfigValidator(self.mock_config)
        guide = validator.get_setup_guide()

        assert "quick_start" in guide
        assert "qdrant_setup" in guide
        assert "environment_variables" in guide
        assert "troubleshooting" in guide

        # Check that each section has content
        for section_name, content in guide.items():
            assert isinstance(content, list)
            assert len(content) > 0

    def test_print_validation_results_with_issues(self, capsys):
        """Test printing validation results with issues."""
        validator = ConfigValidator(self.mock_config)

        results = {
            "issues": ["Critical error"],
            "warnings": ["Warning message"],
            "suggestions": ["Suggestion"]
        }

        with patch('typer.echo') as mock_echo:
            validator.print_validation_results(results)

            # Check that typer.echo was called for issues, warnings, and suggestions
            echo_calls = [call[0][0] for call in mock_echo.call_args_list]
            assert any("Configuration Issues:" in call for call in echo_calls)
            assert any("Configuration Warnings:" in call for call in echo_calls)
            assert any("💡 Suggestions:" in call for call in echo_calls)

    def test_print_validation_results_valid_config(self, capsys):
        """Test printing validation results for valid configuration."""
        validator = ConfigValidator(self.mock_config)

        results = {
            "issues": [],
            "warnings": [],
            "suggestions": []
        }

        with patch('typer.echo') as mock_echo:
            validator.print_validation_results(results)

            # Check that success message was displayed
            echo_calls = [call[0][0] for call in mock_echo.call_args_list]
            assert any("Configuration is valid!" in call for call in echo_calls)


class TestValidateConfigCmd:
    """Test the validate_config_cmd CLI function."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.qdrant.url = "http://localhost:6333"
        self.mock_config.embedding.model = "test-model"
        self.mock_config.workspace.github_user = "testuser"
        self.mock_config.workspace.global_collections = ["docs"]

    @patch('common.utils.config_validator.Config')
    @patch('common.utils.config_validator.ConfigValidator')
    def test_validate_config_cmd_basic(self, mock_validator_class, mock_config_class):
        """Test basic validation command."""
        mock_config_class.return_value = self.mock_config
        mock_validator = Mock()
        mock_validator.validate_all.return_value = (True, {
            "issues": [],
            "warnings": [],
            "qdrant_connection": {"valid": True, "message": "OK"},
            "embedding_model": {"valid": True, "message": "OK"},
            "project_detection": {"valid": True, "message": "OK"},
            "config_validation": {"valid": True, "issues": []}
        })
        mock_validator.print_validation_results = Mock()
        mock_validator_class.return_value = mock_validator

        with pytest.raises(SystemExit) as exc_info:
            validate_config_cmd(verbose=False, config_file=None, setup_guide=False)

        assert exc_info.value.code == 0

    @patch('common.utils.config_validator.Config')
    @patch('common.utils.config_validator.ConfigValidator')
    def test_validate_config_cmd_verbose(self, mock_validator_class, mock_config_class):
        """Test validation command with verbose output."""
        mock_config_class.return_value = self.mock_config
        mock_validator = Mock()
        mock_validator.validate_all.return_value = (True, {
            "issues": [],
            "warnings": [],
            "qdrant_connection": {"valid": True, "message": "OK"},
            "embedding_model": {"valid": True, "message": "OK"},
            "project_detection": {"valid": True, "message": "OK"},
            "config_validation": {"valid": True, "issues": []}
        })
        mock_validator.print_validation_results = Mock()
        mock_validator_class.return_value = mock_validator

        with patch('typer.echo') as mock_echo:
            with pytest.raises(SystemExit) as exc_info:
                validate_config_cmd(verbose=True, config_file=None, setup_guide=False)

            # Check that configuration summary was printed
            echo_calls = [call[0][0] for call in mock_echo.call_args_list]
            assert any("Configuration Summary:" in call for call in echo_calls)

        assert exc_info.value.code == 0

    @patch('common.utils.config_validator.Config')
    @patch('common.utils.config_validator.ConfigValidator')
    def test_validate_config_cmd_with_config_file(self, mock_validator_class, mock_config_class):
        """Test validation command with custom config file."""
        mock_config_class.return_value = self.mock_config
        mock_validator = Mock()
        mock_validator.validate_all.return_value = (True, {
            "issues": [],
            "warnings": [],
            "qdrant_connection": {"valid": True, "message": "OK"},
            "embedding_model": {"valid": True, "message": "OK"},
            "project_detection": {"valid": True, "message": "OK"},
            "config_validation": {"valid": True, "issues": []}
        })
        mock_validator.print_validation_results = Mock()
        mock_validator_class.return_value = mock_validator

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                validate_config_cmd(verbose=False, config_file="/path/to/config.toml", setup_guide=False)

            assert os.environ.get("CONFIG_FILE") == "/path/to/config.toml"

        assert exc_info.value.code == 0

    @patch('common.utils.config_validator.Config')
    @patch('common.utils.config_validator.ConfigValidator')
    def test_validate_config_cmd_setup_guide(self, mock_validator_class, mock_config_class):
        """Test validation command with setup guide."""
        mock_config_class.return_value = self.mock_config
        mock_validator = Mock()
        mock_validator.get_setup_guide.return_value = {
            "quick_start": ["Step 1", "Step 2"],
            "qdrant_setup": ["Install Qdrant"]
        }
        mock_validator_class.return_value = mock_validator

        with patch('typer.echo') as mock_echo:
            with pytest.raises(SystemExit) as exc_info:
                validate_config_cmd(verbose=False, config_file=None, setup_guide=True)

            # Check that setup guide was printed
            echo_calls = [call[0][0] for call in mock_echo.call_args_list]
            assert any("Setup Guide" in call for call in echo_calls)

        assert exc_info.value.code == 0

    @patch('common.utils.config_validator.Config')
    @patch('common.utils.config_validator.ConfigValidator')
    def test_validate_config_cmd_validation_failure(self, mock_validator_class, mock_config_class):
        """Test validation command with validation failure."""
        mock_config_class.return_value = self.mock_config
        mock_validator = Mock()
        mock_validator.validate_all.return_value = (False, {
            "issues": ["Validation error"],
            "warnings": [],
            "qdrant_connection": {"valid": False, "message": "Failed"},
            "embedding_model": {"valid": True, "message": "OK"},
            "project_detection": {"valid": True, "message": "OK"},
            "config_validation": {"valid": True, "issues": []}
        })
        mock_validator.print_validation_results = Mock()
        mock_validator_class.return_value = mock_validator

        with pytest.raises(SystemExit) as exc_info:
            validate_config_cmd(verbose=False, config_file=None, setup_guide=False)

        assert exc_info.value.code == 1

    @patch('common.utils.config_validator.Config')
    def test_validate_config_cmd_exception(self, mock_config_class):
        """Test validation command with exception."""
        mock_config_class.side_effect = Exception("Config error")

        with patch('typer.echo') as mock_echo:
            with pytest.raises(SystemExit) as exc_info:
                validate_config_cmd(verbose=False, config_file=None, setup_guide=False)

            # Check that error was printed
            echo_calls = [call[0][0] for call in mock_echo.call_args_list]
            assert any("Configuration error:" in call for call in echo_calls)

        assert exc_info.value.code == 1


class TestValidateConfigCli:
    """Test the validate_config_cli function."""

    @patch('typer.run')
    def test_validate_config_cli(self, mock_typer_run):
        """Test CLI entry point function."""
        validate_config_cli()

        mock_typer_run.assert_called_once_with(validate_config_cmd)


if __name__ == "__main__":
    pytest.main([__file__])