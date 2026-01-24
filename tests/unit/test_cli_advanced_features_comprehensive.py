"""
Comprehensive Unit Tests for Advanced CLI Features

Tests configuration wizard, smart defaults, command suggestions,
and edge cases for user interaction patterns.

Task 251: Comprehensive testing for advanced CLI features.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest

# Add src paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

# Set CLI mode before any imports
os.environ["WQM_CLI_MODE"] = "true"
os.environ["WQM_LOG_INIT"] = "false"

try:
    from wqm_cli.cli.advanced_features import (
        CommandSuggestionSystem,
        ConfigurationWizard,
        SmartDefaults,
        advanced_features_app,
        command_suggestions,
        configuration_wizard,
        create_advanced_features_app,
        smart_defaults,
    )
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    print(f"Warning: advanced_features module not available: {e}")


@pytest.mark.skipif(not ADVANCED_FEATURES_AVAILABLE, reason="Advanced features module not available")
class TestConfigurationWizard:
    """Test configuration wizard functionality."""

    @pytest.fixture
    def wizard(self):
        """Create fresh configuration wizard."""
        return ConfigurationWizard()

    def test_wizard_initialization(self, wizard):
        """Test wizard initializes correctly."""
        assert wizard is not None
        assert wizard.config_data == {}

    @patch('wqm_cli.cli.advanced_features.Prompt.ask')
    @patch('wqm_cli.cli.advanced_features.Confirm.ask')
    @patch('wqm_cli.cli.advanced_features.IntPrompt.ask')
    def test_configure_qdrant_manual(self, mock_int, mock_confirm, mock_prompt, wizard):
        """Test manual Qdrant configuration."""
        # Mock user inputs
        mock_prompt.side_effect = ["http://localhost:6333", "test-key"]
        mock_confirm.return_value = True
        mock_int.return_value = 30

        with patch.object(wizard, '_detect_qdrant_setups', return_value={}):
            wizard._configure_qdrant()

        expected_config = {
            'url': 'http://localhost:6333',
            'api_key': 'test-key',
            'prefer_grpc': True,
            'timeout': 30
        }

        assert 'qdrant' in wizard.config_data
        assert wizard.config_data['qdrant'] == expected_config

    @patch('wqm_cli.cli.advanced_features.Prompt.ask')
    def test_configure_qdrant_detected_setup(self, mock_prompt, wizard):
        """Test using detected Qdrant setup."""
        detected_setups = {
            'Docker (detected)': {
                'url': 'http://localhost:6333',
                'prefer_grpc': True
            }
        }

        mock_prompt.return_value = '1'  # Select first detected setup

        with patch.object(wizard, '_detect_qdrant_setups', return_value=detected_setups):
            wizard._configure_qdrant()

        assert wizard.config_data['qdrant'] == detected_setups['Docker (detected)']

    @patch('wqm_cli.cli.advanced_features.Prompt.ask')
    @patch('wqm_cli.cli.advanced_features.Confirm.ask')
    @patch('wqm_cli.cli.advanced_features.IntPrompt.ask')
    def test_configure_embedding(self, mock_int, mock_confirm, mock_prompt, wizard):
        """Test embedding configuration."""
        mock_prompt.return_value = '1'  # Select first popular model
        mock_confirm.return_value = True
        mock_int.side_effect = [1000, 200]

        wizard._configure_embedding()

        assert 'embedding' in wizard.config_data
        config = wizard.config_data['embedding']
        assert 'model' in config
        assert config['enable_sparse_vectors'] is True
        assert config['chunk_size'] == 1000
        assert config['chunk_overlap'] == 200

    @patch('wqm_cli.cli.advanced_features.Prompt.ask')
    @patch('wqm_cli.cli.advanced_features.Confirm.ask')
    def test_configure_workspace(self, mock_confirm, mock_prompt, wizard):
        """Test workspace configuration."""
        mock_prompt.side_effect = ["testuser", "project,docs"]
        mock_confirm.return_value = True

        with patch.object(wizard, '_detect_github_user', return_value=None):
            wizard._configure_workspace()

        assert 'workspace' in wizard.config_data
        config = wizard.config_data['workspace']
        assert config['github_user'] == 'testuser'
        assert config['collection_suffixes'] == ['project', 'docs']
        assert config['auto_create_collections'] is True

    @patch('wqm_cli.cli.advanced_features.IntPrompt.ask')
    @patch('wqm_cli.cli.advanced_features.Confirm.ask')
    def test_configure_performance(self, mock_confirm, mock_int, wizard):
        """Test performance configuration."""
        mock_int.side_effect = [32, 8]
        mock_confirm.return_value = True

        with patch.object(wizard, '_estimate_system_memory', return_value=16):
            with patch.object(wizard, '_get_cpu_count', return_value=8):
                wizard._configure_performance()

        assert 'performance' in wizard.config_data
        config = wizard.config_data['performance']
        assert config['batch_size'] == 32
        assert config['max_concurrent_operations'] == 8
        assert config['enable_caching'] is True

    @patch('subprocess.run')
    def test_detect_qdrant_setups(self, mock_run, wizard):
        """Test Qdrant setup detection."""
        # Mock successful docker ps output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="qdrant-container    0.0.0.0:6333->6333/tcp"
        )

        with patch.dict(os.environ, {'QDRANT_URL': 'http://test.com'}):
            setups = wizard._detect_qdrant_setups()

        assert len(setups) >= 2  # Docker detected + environment + default
        assert 'Docker (detected)' in setups
        assert 'Environment Variable' in setups

    def test_detect_qdrant_setups_no_docker(self, wizard):
        """Test setup detection when Docker is not available."""
        with patch('subprocess.run', side_effect=Exception("Docker not available")):
            setups = wizard._detect_qdrant_setups()

        assert 'Local Default' in setups
        assert len(setups) >= 1

    @patch('subprocess.run')
    def test_detect_github_user(self, mock_run, wizard):
        """Test GitHub user detection."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="testuser\n"
        )

        user = wizard._detect_github_user()
        assert user == "testuser"

    @patch('subprocess.run', side_effect=Exception("Git not available"))
    def test_detect_github_user_no_git(self, mock_run, wizard):
        """Test GitHub user detection when git is not available."""
        user = wizard._detect_github_user()
        assert user is None

    @patch('psutil.virtual_memory')
    def test_estimate_system_memory(self, mock_memory, wizard):
        """Test system memory estimation."""
        mock_memory.return_value = Mock(total=16 * 1024**3)  # 16GB

        memory = wizard._estimate_system_memory()
        assert memory == 16

    def test_estimate_system_memory_no_psutil(self, wizard):
        """Test memory estimation fallback when psutil unavailable."""
        with patch('psutil.virtual_memory', side_effect=ImportError):
            memory = wizard._estimate_system_memory()
        assert memory == 8  # Default

    def test_get_cpu_count(self, wizard):
        """Test CPU count detection."""
        with patch('os.cpu_count', return_value=8):
            count = wizard._get_cpu_count()
        assert count == 8

    def test_get_cpu_count_fallback(self, wizard):
        """Test CPU count fallback."""
        with patch('os.cpu_count', return_value=None):
            count = wizard._get_cpu_count()
        assert count == 4  # Default

    @patch('wqm_cli.cli.advanced_features.Confirm.ask')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    def test_review_and_save_confirmed(self, mock_yaml_dump, mock_file, mock_confirm, wizard):
        """Test configuration review and save when confirmed."""
        wizard.config_data = {'test': {'key': 'value'}}
        mock_confirm.return_value = True

        with patch('wqm_cli.cli.advanced_features.console.print'):
            wizard._review_and_save()

        mock_file.assert_called_once()
        mock_yaml_dump.assert_called_once()

    @patch('wqm_cli.cli.advanced_features.Confirm.ask')
    def test_review_and_save_declined(self, mock_confirm, wizard):
        """Test configuration review when save is declined."""
        wizard.config_data = {'test': {'key': 'value'}}
        mock_confirm.return_value = False

        with patch('wqm_cli.cli.advanced_features.console.print'):
            with patch('builtins.open') as mock_file:
                wizard._review_and_save()

        mock_file.assert_not_called()


@pytest.mark.skipif(not ADVANCED_FEATURES_AVAILABLE, reason="Advanced features module not available")
class TestSmartDefaults:
    """Test smart defaults system functionality."""

    @pytest.fixture
    def smart_defaults_instance(self):
        """Create SmartDefaults instance with temp file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = Path(f.name)

        instance = SmartDefaults()
        instance.usage_history_file = temp_file
        instance.usage_history = {
            'command_frequency': {},
            'flag_preferences': {},
            'collection_names': [],
            'file_paths': []
        }

        yield instance

        # Cleanup
        if temp_file.exists():
            temp_file.unlink()

    def test_smart_defaults_initialization(self, smart_defaults_instance):
        """Test SmartDefaults initializes correctly."""
        assert smart_defaults_instance is not None
        assert isinstance(smart_defaults_instance.usage_history, dict)
        assert 'command_frequency' in smart_defaults_instance.usage_history

    def test_load_usage_history_existing_file(self):
        """Test loading existing usage history file."""
        test_data = {'command_frequency': {'test': 1}}

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_data, f)
            temp_file = Path(f.name)

        try:
            instance = SmartDefaults()
            instance.usage_history_file = temp_file
            history = instance._load_usage_history()

            assert history['command_frequency'] == {'test': 1}
        finally:
            temp_file.unlink()

    def test_load_usage_history_nonexistent_file(self):
        """Test loading when file doesn't exist."""
        instance = SmartDefaults()
        instance.usage_history_file = Path('/nonexistent/file.json')

        history = instance._load_usage_history()
        assert 'command_frequency' in history
        assert history['command_frequency'] == {}

    def test_record_command_usage(self, smart_defaults_instance):
        """Test recording command usage."""
        smart_defaults_instance.record_command_usage(
            'memory',
            'add',
            {'verbose': True, 'force': False}
        )

        freq = smart_defaults_instance.usage_history['command_frequency']
        assert 'memory add' in freq
        assert freq['memory add'] == 1

        flag_prefs = smart_defaults_instance.usage_history['flag_preferences']
        assert 'verbose' in flag_prefs
        assert flag_prefs['verbose']['True'] == 1

    def test_record_command_usage_multiple_times(self, smart_defaults_instance):
        """Test recording the same command multiple times."""
        for _ in range(3):
            smart_defaults_instance.record_command_usage('search', 'project')

        freq = smart_defaults_instance.usage_history['command_frequency']
        assert freq['search project'] == 3

    def test_get_suggested_collection_name_from_history(self, smart_defaults_instance):
        """Test collection name suggestion from history."""
        smart_defaults_instance.usage_history['collection_names'] = ['docs', 'project']

        suggestion = smart_defaults_instance.get_suggested_collection_name()
        assert suggestion == 'project'  # Most recent

    @patch('subprocess.run')
    def test_get_suggested_collection_name_from_git(self, mock_run, smart_defaults_instance):
        """Test collection name suggestion from git remote."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="git@github.com:user/test-repo.git\n"
        )

        suggestion = smart_defaults_instance.get_suggested_collection_name('project')
        assert suggestion == 'test-repo'

    def test_get_suggested_collection_name_fallback(self, smart_defaults_instance):
        """Test collection name suggestion fallback."""
        with patch('subprocess.run', side_effect=Exception("No git")):
            suggestion = smart_defaults_instance.get_suggested_collection_name('project')

        assert suggestion == 'default'

    def test_get_suggested_search_limit(self, smart_defaults_instance):
        """Test search limit suggestions."""
        assert smart_defaults_instance.get_suggested_search_limit('project') == 10
        assert smart_defaults_instance.get_suggested_search_limit('global') == 20
        assert smart_defaults_instance.get_suggested_search_limit('unknown') == 10

    def test_get_preferred_format_from_history(self, smart_defaults_instance):
        """Test format preference from history."""
        smart_defaults_instance.usage_history['flag_preferences'] = {
            'format': {'json': 5, 'yaml': 2}
        }

        preferred = smart_defaults_instance.get_preferred_format('search')
        assert preferred == 'json'  # Most used

    def test_get_preferred_format_defaults(self, smart_defaults_instance):
        """Test format preference defaults."""
        assert smart_defaults_instance.get_preferred_format('config') == 'yaml'
        assert smart_defaults_instance.get_preferred_format('admin') == 'table'
        assert smart_defaults_instance.get_preferred_format('unknown') == 'yaml'

    def test_save_usage_history_creates_directory(self, smart_defaults_instance):
        """Test that save creates parent directories."""
        temp_dir = Path(tempfile.mkdtemp())
        smart_defaults_instance.usage_history_file = temp_dir / 'nested' / 'usage.json'

        smart_defaults_instance._save_usage_history()

        assert smart_defaults_instance.usage_history_file.exists()

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


@pytest.mark.skipif(not ADVANCED_FEATURES_AVAILABLE, reason="Advanced features module not available")
class TestCommandSuggestionSystem:
    """Test command suggestion system functionality."""

    @pytest.fixture
    def suggestion_system(self):
        """Create command suggestion system."""
        return CommandSuggestionSystem()

    def test_suggestion_system_initialization(self, suggestion_system):
        """Test suggestion system initializes correctly."""
        assert suggestion_system is not None
        assert hasattr(suggestion_system, 'command_relationships')
        assert hasattr(suggestion_system, 'context_patterns')
        assert len(suggestion_system.command_relationships) > 0
        assert len(suggestion_system.context_patterns) > 0

    def test_suggest_next_commands_config(self, suggestion_system):
        """Test suggestions after config command."""
        suggestions = suggestion_system.suggest_next_commands("config set qdrant.url")

        assert len(suggestions) > 0
        assert any("admin status" in s for s in suggestions)

    def test_suggest_next_commands_ingest(self, suggestion_system):
        """Test suggestions after ingest command."""
        suggestions = suggestion_system.suggest_next_commands("ingest file document.pdf")

        assert len(suggestions) > 0
        assert any("search" in s for s in suggestions)

    def test_suggest_next_commands_error_context(self, suggestion_system):
        """Test suggestions for error context."""
        suggestions = suggestion_system.suggest_next_commands("command failed with error")

        assert len(suggestions) > 0
        assert any("diagnostics" in s or "logs" in s for s in suggestions)

    def test_suggest_next_commands_unknown(self, suggestion_system):
        """Test suggestions for unknown command."""
        suggestions = suggestion_system.suggest_next_commands("unknown command xyz")

        # Should return some suggestions or empty list
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5

    def test_suggest_for_context_first_time(self, suggestion_system):
        """Test context suggestions for first time user."""
        suggestions = suggestion_system.suggest_for_context("first_time_user")

        assert len(suggestions) > 0
        assert any("help" in s.lower() for s in suggestions)

    def test_suggest_for_context_debugging(self, suggestion_system):
        """Test context suggestions for debugging."""
        suggestions = suggestion_system.suggest_for_context("debugging_connection")

        assert len(suggestions) > 0
        assert any("status" in s for s in suggestions)

    def test_suggest_for_context_unknown(self, suggestion_system):
        """Test context suggestions for unknown context."""
        suggestions = suggestion_system.suggest_for_context("unknown_context")

        assert suggestions == []

    def test_suggest_next_commands_deduplication(self, suggestion_system):
        """Test that suggestions are deduplicated."""
        # Modify relationships to create duplicates for testing
        suggestion_system.command_relationships["test_pattern"] = ["admin status", "admin status"]

        suggestions = suggestion_system.suggest_next_commands("test_pattern")

        # Should not have duplicates
        assert len(suggestions) == len(set(suggestions))

    def test_suggest_next_commands_limit(self, suggestion_system):
        """Test that suggestions are limited to 5."""
        # Create many suggestions
        suggestion_system.command_relationships["test_many"] = [
            f"command_{i}" for i in range(10)
        ]

        suggestions = suggestion_system.suggest_next_commands("test_many")

        assert len(suggestions) <= 5


@pytest.mark.skipif(not ADVANCED_FEATURES_AVAILABLE, reason="Advanced features module not available")
class TestAdvancedFeaturesEdgeCases:
    """Test edge cases and error conditions."""

    def test_configuration_wizard_with_keyboard_interrupt(self):
        """Test wizard handles KeyboardInterrupt gracefully."""
        wizard = ConfigurationWizard()

        with patch('wqm_cli.cli.advanced_features.Prompt.ask', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                wizard._configure_qdrant()

    def test_smart_defaults_corrupted_file(self):
        """Test SmartDefaults handles corrupted usage history file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('invalid json content {{{')
            temp_file = Path(f.name)

        try:
            instance = SmartDefaults()
            instance.usage_history_file = temp_file
            history = instance._load_usage_history()

            # Should fall back to default structure
            assert 'command_frequency' in history
            assert history['command_frequency'] == {}
        finally:
            temp_file.unlink()

    def test_smart_defaults_permission_denied(self):
        """Test SmartDefaults handles permission errors gracefully."""
        instance = SmartDefaults()
        instance.usage_history_file = Path('/root/denied.json')  # Likely permission denied

        # Should not raise exception
        instance._save_usage_history()

    def test_configuration_wizard_yaml_import_error(self):
        """Test wizard handles missing YAML library."""
        wizard = ConfigurationWizard()
        # config_data must have nested dict structure for _review_and_save iteration
        wizard.config_data = {'test': {'key': 'value'}}

        with patch('wqm_cli.cli.advanced_features.Confirm.ask', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.dump', side_effect=ImportError("No YAML")):
                    with pytest.raises(ImportError):
                        wizard._review_and_save()

    def test_suggestion_system_empty_relationships(self):
        """Test suggestion system with empty relationships."""
        system = CommandSuggestionSystem()
        system.command_relationships = {}
        system.context_patterns = {}

        suggestions = system.suggest_next_commands("any command")
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

        context_suggestions = system.suggest_for_context("any context")
        assert context_suggestions == []

    def test_smart_defaults_unicode_in_commands(self):
        """Test SmartDefaults handles unicode in command names."""
        instance = SmartDefaults()
        instance.usage_history_file = Path(tempfile.mkdtemp()) / 'unicode.json'

        # Record command with unicode
        instance.record_command_usage('测试', '命令', {'标志': '值'})

        # Should handle unicode gracefully
        freq = instance.usage_history['command_frequency']
        assert '测试 命令' in freq

        # Cleanup
        if instance.usage_history_file.exists():
            instance.usage_history_file.unlink()
            instance.usage_history_file.parent.rmdir()

    @pytest.mark.xfail(reason="Test design flaw: directly calls patched methods that raise exceptions instead of testing higher-level behavior")
    def test_configuration_wizard_system_detection_errors(self):
        """Test wizard handles system detection errors gracefully."""
        wizard = ConfigurationWizard()

        # Mock all detection methods to raise exceptions
        with patch.object(wizard, '_estimate_system_memory', side_effect=Exception("Error")):
            with patch.object(wizard, '_get_cpu_count', side_effect=Exception("Error")):
                with patch.object(wizard, '_detect_github_user', side_effect=Exception("Error")):
                    # Should fall back to defaults without crashing
                    memory = wizard._estimate_system_memory()
                    cpu = wizard._get_cpu_count()
                    user = wizard._detect_github_user()

                    assert isinstance(memory, int)
                    assert isinstance(cpu, int)
                    assert user is None


@pytest.mark.skipif(not ADVANCED_FEATURES_AVAILABLE, reason="Advanced features module not available")
class TestGlobalInstances:
    """Test global instances and app creation."""

    def test_global_instances_exist(self):
        """Test that global instances are created."""
        assert configuration_wizard is not None
        assert smart_defaults is not None
        assert command_suggestions is not None
        assert advanced_features_app is not None

    def test_create_advanced_features_app(self):
        """Test advanced features app creation."""
        app = create_advanced_features_app()
        assert app is not None
        assert hasattr(app, 'registered_commands') or hasattr(app, 'commands')

    def test_global_configuration_wizard_type(self):
        """Test global configuration wizard type."""
        assert isinstance(configuration_wizard, ConfigurationWizard)

    def test_global_smart_defaults_type(self):
        """Test global smart defaults type."""
        assert isinstance(smart_defaults, SmartDefaults)

    def test_global_command_suggestions_type(self):
        """Test global command suggestions type."""
        assert isinstance(command_suggestions, CommandSuggestionSystem)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
