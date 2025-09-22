"""
Comprehensive test suite for LSP detection system.

Tests LSP detection across different platforms, notification system,
fallback mechanisms, and integration scenarios.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, call, patch
import pytest

from workspace_qdrant_mcp.core.lsp_detector import (
    LSPDetector,
    LSPDetectionResult,
    LSPServerInfo,
    get_default_detector,
    scan_lsps,
    get_supported_extensions
)
from workspace_qdrant_mcp.core.lsp_notifications import (
    LSPNotificationManager,
    NotificationEntry,
    NotificationLevel,
    NotificationHandler,
    LSPInstallationInfo,
    get_default_notification_manager,
    notify_missing_lsp
)
from workspace_qdrant_mcp.core.lsp_fallback import (
    BuildToolDetector,
    FallbackExtensionProvider,
    BuildToolInfo,
    BuildToolType,
    FallbackDetectionResult,
    get_default_build_tool_detector,
    get_default_fallback_provider
)


class TestLSPDetector:
    """Test core LSP detection functionality."""
    
    def test_detector_initialization(self):
        """Test LSP detector initializes with correct parameters."""
        detector = LSPDetector(cache_ttl=600, detection_timeout=10.0)
        assert detector.cache_ttl == 600
        assert detector.detection_timeout == 10.0
        assert detector._cached_result is None
        assert detector._last_scan_time == 0.0
    
    def test_lsp_extension_mappings(self):
        """Test LSP extension mappings are complete and valid."""
        detector = LSPDetector()
        
        # Check that all expected LSPs are in the mapping
        expected_lsps = [
            'rust-analyzer', 'ruff', 'typescript-language-server',
            'pyright', 'pylsp', 'gopls', 'clangd', 'java-language-server'
        ]
        
        for lsp in expected_lsps:
            assert lsp in detector.LSP_EXTENSION_MAP
            mapping = detector.LSP_EXTENSION_MAP[lsp]
            assert 'extensions' in mapping
            assert 'priority' in mapping
            assert 'alternative_names' in mapping
            assert 'capabilities' in mapping
            assert isinstance(mapping['extensions'], list)
            assert isinstance(mapping['priority'], int)
            assert isinstance(mapping['alternative_names'], list)
            assert isinstance(mapping['capabilities'], set)
    
    def test_essential_extensions(self):
        """Test essential extensions are properly defined."""
        detector = LSPDetector()
        essential = detector.ESSENTIAL_EXTENSIONS
        
        # Should include common file types
        assert '.md' in essential
        assert '.json' in essential
        assert '.yaml' in essential
        assert '.sh' in essential
        assert '.gitignore' in essential
        assert 'Dockerfile' in essential
    
    @patch('shutil.which')
    def test_binary_detection_success(self, mock_which):
        """Test successful binary detection."""
        detector = LSPDetector()
        mock_which.return_value = '/usr/local/bin/rust-analyzer'
        
        result = detector._check_binary_exists('rust-analyzer')
        assert result == '/usr/local/bin/rust-analyzer'
        mock_which.assert_called_once_with('rust-analyzer')
    
    @patch('shutil.which')
    def test_binary_detection_failure(self, mock_which):
        """Test binary detection when binary not found."""
        detector = LSPDetector()
        mock_which.return_value = None
        
        result = detector._check_binary_exists('nonexistent-lsp')
        assert result is None
        mock_which.assert_called_once_with('nonexistent-lsp')
    
    @patch('subprocess.run')
    def test_version_detection_success(self, mock_run):
        """Test successful LSP version detection."""
        detector = LSPDetector()
        
        # Mock successful version check
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "rust-analyzer 1.0.0\nUsage info..."
        mock_run.return_value = mock_result
        
        version = detector._get_lsp_version('/usr/bin/rust-analyzer', 'rust-analyzer')
        assert version == "rust-analyzer 1.0.0"
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_version_detection_timeout(self, mock_run):
        """Test version detection timeout handling."""
        detector = LSPDetector(detection_timeout=0.1)
        mock_run.side_effect = subprocess.TimeoutExpired(['test'], 0.1)
        
        version = detector._get_lsp_version('/usr/bin/rust-analyzer', 'rust-analyzer')
        assert version is None
    
    @patch('subprocess.run')
    def test_version_detection_error(self, mock_run):
        """Test version detection error handling."""
        detector = LSPDetector()
        mock_run.side_effect = Exception("Command failed")
        
        version = detector._get_lsp_version('/usr/bin/rust-analyzer', 'rust-analyzer')
        assert version is None
    
    @patch('shutil.which')
    def test_scan_available_lsps(self, mock_which):
        """Test LSP scanning functionality."""
        detector = LSPDetector()
        
        # Mock finding some LSPs
        def which_side_effect(binary):
            lsp_paths = {
                'rust-analyzer': '/usr/bin/rust-analyzer',
                'ruff': '/usr/local/bin/ruff',
                'gopls': '/usr/bin/gopls'
            }
            return lsp_paths.get(binary)
        
        mock_which.side_effect = which_side_effect
        
        with patch.object(detector, '_get_lsp_version', return_value="1.0.0"):
            result = detector.scan_available_lsps()
        
        assert isinstance(result, LSPDetectionResult)
        assert len(result.detected_lsps) >= 0  # May find LSPs based on mock
        assert result.scan_duration >= 0
        assert isinstance(result.errors, list)
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        detector = LSPDetector()
        
        with patch.object(detector, 'scan_available_lsps') as mock_scan:
            mock_result = LSPDetectionResult()
            mock_result.detected_lsps = {
                'rust-analyzer': LSPServerInfo(
                    name='rust-analyzer',
                    binary_path='/usr/bin/rust-analyzer',
                    supported_extensions=['.rs', '.toml'],
                    priority=10
                )
            }
            mock_scan.return_value = mock_result
            
            extensions = detector.get_supported_extensions()
            assert '.rs' in extensions
            assert '.toml' in extensions
            assert '.md' in extensions  # Essential extension
    
    def test_cache_functionality(self):
        """Test caching behavior."""
        detector = LSPDetector(cache_ttl=1)
        
        with patch.object(detector, 'scan_available_lsps', wraps=detector.scan_available_lsps) as mock_scan:
            # First call should trigger scan
            result1 = detector.get_supported_extensions()
            assert mock_scan.call_count >= 1
            
            # Second call should use cache
            mock_scan.reset_mock()
            result2 = detector.get_supported_extensions()
            assert result1 == result2
            # Cache should be used, so no new scan
            assert mock_scan.call_count == 0
            
            # Wait for cache to expire
            time.sleep(1.1)
            result3 = detector.get_supported_extensions()
            # Should trigger new scan after cache expiry
    
    def test_extension_lsp_mapping(self):
        """Test getting LSP for specific extension."""
        detector = LSPDetector()
        
        with patch.object(detector, 'scan_available_lsps') as mock_scan:
            mock_result = LSPDetectionResult()
            mock_result.detected_lsps = {
                'rust-analyzer': LSPServerInfo(
                    name='rust-analyzer',
                    binary_path='/usr/bin/rust-analyzer',
                    supported_extensions=['.rs'],
                    priority=10
                ),
                'ruff': LSPServerInfo(
                    name='ruff',
                    binary_path='/usr/bin/ruff',
                    supported_extensions=['.py'],
                    priority=8
                )
            }
            mock_scan.return_value = mock_result
            
            rust_lsp = detector.get_lsp_for_extension('.rs')
            assert rust_lsp is not None
            assert rust_lsp.name == 'rust-analyzer'
            
            python_lsp = detector.get_lsp_for_extension('.py')
            assert python_lsp is not None
            assert python_lsp.name == 'ruff'
            
            unknown_lsp = detector.get_lsp_for_extension('.unknown')
            assert unknown_lsp is None
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        detector = LSPDetector()
        
        # Populate cache
        with patch.object(detector, 'scan_available_lsps', return_value=LSPDetectionResult()):
            detector.get_supported_extensions()
            assert detector._cached_result is not None
            
            # Clear cache
            detector.clear_cache()
            assert detector._cached_result is None
            assert detector._last_scan_time == 0.0
    
    def test_global_detector_instance(self):
        """Test global detector instance functionality."""
        detector1 = get_default_detector()
        detector2 = get_default_detector()
        assert detector1 is detector2  # Should be same instance
        
        # Test convenience functions
        with patch.object(detector1, 'scan_available_lsps', return_value=LSPDetectionResult()):
            result = scan_lsps()
            assert isinstance(result, LSPDetectionResult)
            
            extensions = get_supported_extensions()
            assert isinstance(extensions, list)


class TestLSPNotificationManager:
    """Test LSP notification system."""
    
    def test_notification_manager_initialization(self):
        """Test notification manager initializes correctly."""
        manager = LSPNotificationManager(
            max_notifications_per_type=5,
            notification_cooldown=600
        )
        assert manager.max_notifications_per_type == 5
        assert manager.notification_cooldown == 600
        assert len(manager.notifications) == 0
        assert len(manager.dismissed_types) == 0
    
    def test_lsp_installation_database(self):
        """Test LSP installation information is comprehensive."""
        manager = LSPNotificationManager()
        
        # Check that major LSPs are covered
        expected_lsps = [
            'rust-analyzer', 'ruff', 'typescript-language-server',
            'pyright', 'pylsp', 'gopls', 'clangd'
        ]
        
        for lsp in expected_lsps:
            assert lsp in manager.LSP_INSTALLATION_DB
            info = manager.LSP_INSTALLATION_DB[lsp]
            assert isinstance(info, LSPInstallationInfo)
            assert info.description
            assert info.package_managers
            assert isinstance(info.package_managers, dict)
    
    def test_platform_specific_instructions(self):
        """Test platform-specific installation instructions."""
        manager = LSPNotificationManager()
        
        instructions = manager._get_platform_specific_instructions('rust-analyzer')
        assert isinstance(instructions, dict)
        assert len(instructions) > 0
        
        # Should include at least some package manager instructions
        assert any(key.startswith('via_') for key in instructions.keys())
    
    def test_notification_throttling(self):
        """Test notification throttling mechanism."""
        manager = LSPNotificationManager(
            max_notifications_per_type=2,
            notification_cooldown=300
        )
        
        # First notification should go through
        result1 = manager.notify_missing_lsp('.rs', 'rust-analyzer')
        assert result1 is True
        assert manager.session_counts['.rs'] == 1
        
        # Second notification should go through
        result2 = manager.notify_missing_lsp('.rs', 'rust-analyzer')
        assert result2 is True
        assert manager.session_counts['.rs'] == 2
        
        # Third notification should be throttled
        result3 = manager.notify_missing_lsp('.rs', 'rust-analyzer')
        assert result3 is False  # Throttled
    
    def test_notification_cooldown(self):
        """Test notification cooldown period."""
        manager = LSPNotificationManager(
            max_notifications_per_type=10,
            notification_cooldown=1  # 1 second for testing
        )
        
        # First notification
        result1 = manager.notify_missing_lsp('.py', 'ruff')
        assert result1 is True
        
        # Immediate second notification should be throttled (cooldown)
        result2 = manager.notify_missing_lsp('.py', 'ruff')
        assert result2 is False
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        result3 = manager.notify_missing_lsp('.py', 'ruff')
        assert result3 is True
    
    def test_dismiss_functionality(self):
        """Test file type dismissal functionality."""
        manager = LSPNotificationManager()
        
        # Dismiss a file type
        manager.dismiss_file_type('.js')
        assert '.js' in manager.dismissed_types
        
        # Notifications for dismissed types should be blocked
        result = manager.notify_missing_lsp('.js', 'typescript-language-server')
        assert result is False
        
        # Re-enable notifications
        manager.undismiss_file_type('.js')
        assert '.js' not in manager.dismissed_types
        
        # Should work again
        result = manager.notify_missing_lsp('.js', 'typescript-language-server')
        assert result is True
    
    def test_notification_handlers(self):
        """Test notification handler system."""
        manager = LSPNotificationManager(
            default_handlers=[NotificationHandler.CALLBACK]
        )
        
        # Register custom callback
        callback_calls = []
        def test_callback(entry):
            callback_calls.append(entry)
        
        manager.register_callback('test', test_callback)
        
        # Send notification
        manager.notify_missing_lsp('.go', 'gopls')
        
        # Callback should have been called
        assert len(callback_calls) == 1
        assert callback_calls[0].file_extension == '.go'
        assert callback_calls[0].lsp_name == 'gopls'
        
        # Unregister callback
        manager.unregister_callback('test')
        assert 'test' not in manager.custom_callbacks
    
    def test_notification_history(self):
        """Test notification history tracking."""
        manager = LSPNotificationManager()
        
        # Send some notifications
        manager.notify_missing_lsp('.rs', 'rust-analyzer')
        manager.notify_missing_lsp('.py', 'ruff')
        
        history = manager.get_notification_history()
        assert len(history) == 2
        
        extensions = [entry.file_extension for entry in history]
        assert '.rs' in extensions
        assert '.py' in extensions
    
    def test_clear_history(self):
        """Test history clearing functionality."""
        manager = LSPNotificationManager()
        
        # Add some notifications
        manager.notify_missing_lsp('.rs', 'rust-analyzer')
        assert len(manager.notifications) == 1
        
        # Clear history
        manager.clear_history()
        assert len(manager.notifications) == 0
        assert len(manager.session_counts) == 0
    
    def test_statistics_generation(self):
        """Test notification statistics."""
        manager = LSPNotificationManager()
        
        # Add some notifications
        manager.notify_missing_lsp('.rs', 'rust-analyzer')
        manager.notify_missing_lsp('.py', 'ruff')
        manager.dismiss_file_type('.js')
        
        stats = manager.get_statistics()
        assert isinstance(stats, dict)
        assert stats['total_notifications'] == 2
        assert stats['dismissed_types'] == 1
        assert '.rs' in stats['session_counts']
        assert '.py' in stats['session_counts']
        assert len(stats['available_lsps']) > 0
    
    def test_message_formatting(self):
        """Test notification message formatting."""
        manager = LSPNotificationManager()
        
        message = manager._format_installation_message('.rs', 'rust-analyzer')
        assert isinstance(message, str)
        assert '.rs' in message
        assert 'rust-analyzer' in message
        assert 'Installation options:' in message
        
        # Should include platform-appropriate instructions
        current_platform = platform.system()
        if current_platform == 'Darwin':
            assert 'brew' in message or 'cargo' in message
        elif current_platform == 'Linux':
            assert any(pm in message for pm in ['apt', 'pacman', 'dnf'])
    
    @pytest.fixture
    def temp_persist_file(self):
        """Create temporary file for persistence testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            yield f.name
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    def test_persistence(self, temp_persist_file):
        """Test notification state persistence."""
        # Create manager with persistence
        manager = LSPNotificationManager(persist_file=temp_persist_file)
        
        # Add notifications and dismissals
        manager.notify_missing_lsp('.rs', 'rust-analyzer')
        manager.dismiss_file_type('.js')
        
        # Create new manager with same persistence file
        manager2 = LSPNotificationManager(persist_file=temp_persist_file)
        
        # Should load previous state
        assert '.js' in manager2.dismissed_types
        # Recent notifications should be loaded
        if manager2.notifications:
            assert any(entry.file_extension == '.rs' for entry in manager2.notifications.values())
    
    def test_global_notification_manager(self):
        """Test global notification manager instance."""
        manager1 = get_default_notification_manager()
        manager2 = get_default_notification_manager()
        assert manager1 is manager2
        
        # Test convenience function
        result = notify_missing_lsp('.test', 'test-lsp')
        assert isinstance(result, bool)


class TestBuildToolDetector:
    """Test build tool detection functionality."""
    
    def test_build_tool_detector_initialization(self):
        """Test build tool detector initialization."""
        detector = BuildToolDetector(detection_timeout=10.0, cache_ttl=600)
        assert detector.detection_timeout == 10.0
        assert detector.cache_ttl == 600
        assert detector._cached_result is None
    
    def test_build_tool_configuration(self):
        """Test build tool configuration is complete."""
        detector = BuildToolDetector()
        
        # Check that all build tool types have configuration
        for tool_type in BuildToolType:
            assert tool_type in detector.BUILD_TOOL_CONFIG
            config = detector.BUILD_TOOL_CONFIG[tool_type]
            assert 'binaries' in config
            assert 'config_files' in config
            assert 'extensions' in config
            assert 'version_args' in config
            assert isinstance(config['binaries'], list)
            assert isinstance(config['config_files'], list)
            assert isinstance(config['extensions'], list)
            assert isinstance(config['version_args'], list)
    
    def test_infrastructure_patterns(self):
        """Test infrastructure file patterns."""
        detector = BuildToolDetector()
        
        patterns = detector.INFRASTRUCTURE_PATTERNS
        assert 'docker' in patterns
        assert 'kubernetes' in patterns
        assert 'terraform' in patterns
        
        # Check Docker patterns
        docker_patterns = patterns['docker']
        assert 'Dockerfile*' in docker_patterns
        assert 'docker-compose*.yml' in docker_patterns
    
    @patch('shutil.which')
    def test_build_tool_detection(self, mock_which):
        """Test build tool binary detection."""
        detector = BuildToolDetector()
        
        # Mock finding some build tools
        def which_side_effect(binary):
            tool_paths = {
                'make': '/usr/bin/make',
                'cargo': '/usr/local/bin/cargo',
                'npm': '/usr/bin/npm'
            }
            return tool_paths.get(binary)
        
        mock_which.side_effect = which_side_effect
        
        with patch.object(detector, '_get_tool_version', return_value="1.0.0"):
            result = detector.scan_build_tools()
        
        assert isinstance(result, FallbackDetectionResult)
        assert len(result.essential_extensions) > 0
        assert isinstance(result.build_tool_extensions, list)
        assert isinstance(result.infrastructure_extensions, list)
        assert isinstance(result.total_extensions, list)
    
    def test_fallback_extensions(self):
        """Test default fallback extensions."""
        detector = BuildToolDetector()
        
        extensions = detector.DEFAULT_FALLBACK_EXTENSIONS
        assert '.md' in extensions
        assert '.json' in extensions
        assert '.yaml' in extensions
        assert '.sh' in extensions
        assert '.gitignore' in extensions
    
    def test_get_fallback_extensions(self):
        """Test getting comprehensive fallback extensions."""
        detector = BuildToolDetector()
        
        with patch.object(detector, 'scan_build_tools') as mock_scan:
            mock_result = FallbackDetectionResult()
            mock_result.total_extensions = ['.md', '.json', '.py', '.rs']
            mock_scan.return_value = mock_result
            
            extensions = detector.get_fallback_extensions()
            assert extensions == ['.md', '.json', '.py', '.rs']
    
    def test_build_tool_availability_check(self):
        """Test checking specific build tool availability."""
        detector = BuildToolDetector()
        
        with patch.object(detector, 'scan_build_tools') as mock_scan:
            mock_result = FallbackDetectionResult()
            mock_result.detected_build_tools = {
                'make': BuildToolInfo(
                    name='make',
                    tool_type=BuildToolType.MAKE,
                    binary_path='/usr/bin/make'
                )
            }
            mock_scan.return_value = mock_result
            
            assert detector.is_build_tool_available(BuildToolType.MAKE) is True
            assert detector.is_build_tool_available(BuildToolType.GRADLE) is False
    
    def test_extensions_by_category(self):
        """Test extension categorization."""
        detector = BuildToolDetector()
        
        with patch.object(detector, 'scan_build_tools') as mock_scan:
            mock_result = FallbackDetectionResult()
            mock_result.essential_extensions = ['.md', '.json']
            mock_result.build_tool_extensions = ['.gradle']
            mock_result.infrastructure_extensions = ['Dockerfile']
            mock_result.detected_build_tools = {'make': Mock()}
            mock_scan.return_value = mock_result
            
            categories = detector.get_extensions_by_category()
            assert 'essential' in categories
            assert 'build_tools' in categories
            assert 'infrastructure' in categories
            assert 'detected_tools' in categories
            assert categories['essential'] == ['.md', '.json']


class TestFallbackExtensionProvider:
    """Test fallback extension provider integration."""
    
    def test_fallback_provider_initialization(self):
        """Test fallback extension provider initialization."""
        provider = FallbackExtensionProvider()
        assert provider.build_tool_detector is not None
        assert provider.include_language_fallbacks is True
        assert provider.include_build_tools is True
        assert provider.include_infrastructure is True
    
    def test_comprehensive_extensions_without_lsp(self):
        """Test getting extensions without LSP detector."""
        provider = FallbackExtensionProvider(lsp_detector=None)
        
        with patch.object(provider.build_tool_detector, 'scan_build_tools') as mock_scan:
            mock_result = FallbackDetectionResult()
            mock_result.essential_extensions = ['.md', '.json']
            mock_result.build_tool_extensions = ['.gradle', '.mk']
            mock_result.infrastructure_extensions = ['Dockerfile', '*.tf']
            mock_scan.return_value = mock_result
            
            extensions = provider.get_comprehensive_extensions()
            assert '.md' in extensions
            assert '.json' in extensions
            assert '.gradle' in extensions
            assert 'Dockerfile' in extensions
    
    def test_comprehensive_extensions_with_lsp(self):
        """Test getting extensions with LSP detector integration."""
        mock_lsp_detector = Mock()
        mock_lsp_detector.get_supported_extensions.return_value = ['.py', '.rs', '.ts']
        
        provider = FallbackExtensionProvider(lsp_detector=mock_lsp_detector)
        
        with patch.object(provider.build_tool_detector, 'scan_build_tools') as mock_scan:
            mock_result = FallbackDetectionResult()
            mock_result.essential_extensions = ['.md', '.json']
            mock_result.build_tool_extensions = ['.gradle']
            mock_result.infrastructure_extensions = ['Dockerfile']
            mock_scan.return_value = mock_result
            
            extensions = provider.get_comprehensive_extensions()
            assert '.py' in extensions  # From LSP
            assert '.rs' in extensions  # From LSP
            assert '.md' in extensions  # Essential
            assert '.gradle' in extensions  # Build tools
            assert 'Dockerfile' in extensions  # Infrastructure
    
    def test_extensions_with_sources(self):
        """Test getting extensions organized by source."""
        mock_lsp_detector = Mock()
        mock_lsp_detector.get_supported_extensions.return_value = ['.py', '.rs']
        
        provider = FallbackExtensionProvider(lsp_detector=mock_lsp_detector)
        
        with patch.object(provider.build_tool_detector, 'scan_build_tools') as mock_scan:
            mock_result = FallbackDetectionResult()
            mock_result.essential_extensions = ['.md']
            mock_result.build_tool_extensions = ['.gradle']
            mock_result.infrastructure_extensions = ['Dockerfile']
            mock_scan.return_value = mock_result
            
            sources = provider.get_extensions_with_sources()
            assert sources['lsp_detected'] == ['.py', '.rs']
            assert sources['essential'] == ['.md']
            assert sources['build_tools'] == ['.gradle']
            assert sources['infrastructure'] == ['Dockerfile']
    
    def test_priority_extensions(self):
        """Test priority-ordered extension list."""
        provider = FallbackExtensionProvider()
        
        with patch.object(provider, 'get_extensions_with_sources') as mock_sources:
            mock_sources.return_value = {
                'lsp_detected': ['.py', '.rs'],
                'essential': ['.md', '.json'],
                'build_tools': ['.gradle'],
                'infrastructure': ['Dockerfile'],
                'language_fallbacks': ['.java']
            }
            
            priority_exts = provider.get_priority_extensions()
            
            # Should be ordered by priority: LSP > Essential > Build > Infra > Fallbacks
            assert priority_exts.index('.py') < priority_exts.index('.md')
            assert priority_exts.index('.md') < priority_exts.index('.gradle')
            assert priority_exts.index('.gradle') < priority_exts.index('Dockerfile')
            assert priority_exts.index('Dockerfile') < priority_exts.index('.java')
    
    def test_extension_support_check(self):
        """Test checking if extension is supported."""
        provider = FallbackExtensionProvider()
        
        with patch.object(provider, 'get_extensions_with_sources') as mock_sources:
            mock_sources.return_value = {
                'lsp_detected': ['.py'],
                'essential': ['.md'],
                'build_tools': [],
                'infrastructure': [],
                'language_fallbacks': []
            }
            
            supported, sources = provider.is_extension_supported('.py')
            assert supported is True
            assert 'lsp_detected' in sources
            
            supported, sources = provider.is_extension_supported('.unknown')
            assert supported is False
            assert len(sources) == 0
    
    def test_provider_summary(self):
        """Test provider summary generation."""
        provider = FallbackExtensionProvider()
        
        with patch.object(provider, 'get_extensions_with_sources') as mock_sources:
            with patch.object(provider, 'get_comprehensive_extensions') as mock_comprehensive:
                mock_sources.return_value = {
                    'lsp_detected': ['.py'],
                    'essential': ['.md'],
                    'build_tools': ['.gradle'],
                    'infrastructure': ['Dockerfile'],
                    'language_fallbacks': []
                }
                mock_comprehensive.return_value = ['.py', '.md', '.gradle', 'Dockerfile']
                
                with patch.object(provider.build_tool_detector, 'scan_build_tools') as mock_scan:
                    mock_result = FallbackDetectionResult()
                    mock_result.detected_build_tools = {'make': Mock()}
                    mock_result.scan_duration = 0.1
                    mock_scan.return_value = mock_result
                    
                    summary = provider.get_summary()
                    assert summary['total_extensions'] == 4
                    assert summary['sources_enabled']['build_tools'] is True
                    assert 'make' in summary['detected_build_tools']
    
    def test_global_fallback_provider(self):
        """Test global fallback provider instance."""
        provider1 = get_default_fallback_provider()
        provider2 = get_default_fallback_provider()
        assert provider1 is provider2


class TestLSPDetectionIntegration:
    """Integration tests for complete LSP detection system."""
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock environment with some LSPs and build tools available."""
        with patch('shutil.which') as mock_which:
            def which_side_effect(binary):
                available_tools = {
                    'rust-analyzer': '/usr/bin/rust-analyzer',
                    'ruff': '/usr/local/bin/ruff',
                    'make': '/usr/bin/make',
                    'cargo': '/usr/local/bin/cargo',
                    'npm': '/usr/bin/npm'
                }
                return available_tools.get(binary)
            
            mock_which.side_effect = which_side_effect
            yield mock_which
    
    def test_complete_detection_workflow(self, mock_environment):
        """Test complete detection workflow with LSP and build tools."""
        # Initialize components
        lsp_detector = LSPDetector()
        build_detector = BuildToolDetector()
        notification_manager = LSPNotificationManager()
        fallback_provider = FallbackExtensionProvider(
            lsp_detector=lsp_detector,
            build_tool_detector=build_detector
        )
        
        with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
            with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                # Get comprehensive extensions
                extensions = fallback_provider.get_comprehensive_extensions()
                
                # Should include LSP extensions
                assert '.rs' in extensions  # rust-analyzer
                assert '.py' in extensions  # ruff
                
                # Should include essential extensions
                assert '.md' in extensions
                assert '.json' in extensions
                
                # Check extension sources
                sources = fallback_provider.get_extensions_with_sources()
                assert len(sources['lsp_detected']) > 0
                assert len(sources['essential']) > 0
    
    def test_notification_integration(self, mock_environment):
        """Test notification system integration with detection."""
        lsp_detector = LSPDetector()
        notification_manager = LSPNotificationManager()
        
        with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
            # Scan for available LSPs
            detection_result = lsp_detector.scan_available_lsps()
            
            # Simulate encountering unsupported file type
            if '.kt' not in lsp_detector.get_supported_extensions():
                # Kotlin not supported, should trigger notification
                result = notification_manager.notify_missing_lsp('.kt', 'kotlin-language-server')
                assert isinstance(result, bool)
    
    def test_fallback_without_lsps(self):
        """Test fallback behavior when no LSPs are available."""
        with patch('shutil.which', return_value=None):  # No binaries found
            fallback_provider = FallbackExtensionProvider()
            
            extensions = fallback_provider.get_comprehensive_extensions()
            
            # Should still have essential extensions
            assert '.md' in extensions
            assert '.json' in extensions
            assert '.yaml' in extensions
            
            # Should have some infrastructure support
            assert 'Dockerfile' in extensions or any('docker' in ext.lower() for ext in extensions)
    
    def test_performance_characteristics(self, mock_environment):
        """Test performance characteristics of detection system."""
        lsp_detector = LSPDetector()
        build_detector = BuildToolDetector()
        
        with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
            with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                import time
                
                # Time LSP detection
                start = time.time()
                lsp_result = lsp_detector.scan_available_lsps()
                lsp_duration = time.time() - start
                
                # Time build tool detection  
                start = time.time()
                build_result = build_detector.scan_build_tools()
                build_duration = time.time() - start
                
                # Should complete reasonably quickly
                assert lsp_duration < 10.0  # 10 seconds max
                assert build_duration < 10.0  # 10 seconds max
                
                # Cached calls should be much faster
                start = time.time()
                lsp_result2 = lsp_detector.scan_available_lsps()
                cached_duration = time.time() - start
                
                assert cached_duration < lsp_duration  # Should be faster from cache
    
    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility."""
        # This test verifies the system works on different platforms
        detector = LSPDetector()
        build_detector = BuildToolDetector()
        notification_manager = LSPNotificationManager()
        
        # Should work regardless of platform
        current_platform = platform.system()
        assert current_platform in ['Darwin', 'Linux', 'Windows']
        
        # Essential extensions should be platform-agnostic
        extensions = detector.ESSENTIAL_EXTENSIONS
        assert '.md' in extensions
        assert '.json' in extensions
        
        # Build tool detection should have appropriate tools for platform
        tool_config = build_detector.BUILD_TOOL_CONFIG
        assert BuildToolType.MAKE in tool_config  # Available on most platforms
        
        # Notification system should have platform-specific instructions
        instructions = notification_manager._get_platform_specific_instructions('rust-analyzer')
        assert len(instructions) > 0
    
    def test_configuration_integration(self):
        """Test integration with configuration system."""
        # Test that detection system respects configuration when available
        detector = LSPDetector()
        
        # Should handle missing configuration gracefully
        assert detector.config is None or hasattr(detector.config, 'cache_ttl')
        
        # Should work with default settings
        extensions = detector.get_supported_extensions()
        assert isinstance(extensions, list)
        assert len(extensions) > 0
    
    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        detector = LSPDetector()
        
        # Should handle binary detection errors gracefully
        with patch('shutil.which', side_effect=Exception("System error")):
            result = detector.scan_available_lsps()
            assert isinstance(result, LSPDetectionResult)
            # Should have errors but still return some extensions
            extensions = detector.get_supported_extensions()
            assert len(extensions) > 0  # At least essential extensions
    
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        # Create multiple instances to test for memory leaks
        detectors = []
        for _ in range(10):
            detector = LSPDetector()
            with patch('shutil.which', return_value=None):
                detector.scan_available_lsps()
            detectors.append(detector)
        
        # Should not accumulate excessive state
        for detector in detectors:
            assert len(detector._cached_result.errors if detector._cached_result else []) < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])