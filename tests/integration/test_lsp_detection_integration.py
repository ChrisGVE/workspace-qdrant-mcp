"""
End-to-End Integration Tests for LSP Detection System

Tests complete workflow: LSP detection -> extension mapping -> watch configuration 
-> file monitoring -> notification system integration with performance validation.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import pytest

# Import the components we're testing
from workspace_qdrant_mcp.core.lsp_detector import LSPDetector, LSPDetectionResult, LSPServerInfo
from workspace_qdrant_mcp.core.lsp_notifications import LSPNotificationManager, NotificationLevel
from workspace_qdrant_mcp.core.lsp_fallback import BuildToolDetector, FallbackExtensionProvider
from workspace_qdrant_mcp.core.watch_config import WatchConfigurationPersistent as WatchConfig
from workspace_qdrant_mcp.core.config import WorkspaceConfig


class MockProjectStructure:
    """Creates mock project structures for testing different scenarios."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.files_created = []
    
    def create_rust_project(self):
        """Create a Rust project structure."""
        cargo_toml = self.base_path / "Cargo.toml"
        cargo_toml.write_text("""
[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
""")
        
        src_dir = self.base_path / "src"
        src_dir.mkdir()
        
        main_rs = src_dir / "main.rs"
        main_rs.write_text("""
fn main() {
    println!("Hello, world!");
}
""")
        
        lib_rs = src_dir / "lib.rs"
        lib_rs.write_text("""
pub fn add(left: usize, right: usize) -> usize {
    left + right
}
""")
        
        self.files_created.extend([cargo_toml, main_rs, lib_rs])
        return ['.rs', '.toml']
    
    def create_python_project(self):
        """Create a Python project structure."""
        setup_py = self.base_path / "setup.py"
        setup_py.write_text("""
from setuptools import setup, find_packages

setup(
    name="test-project",
    version="0.1.0",
    packages=find_packages(),
)
""")
        
        requirements = self.base_path / "requirements.txt"
        requirements.write_text("""
requests>=2.25.0
pytest>=6.0.0
""")
        
        src_dir = self.base_path / "src"
        src_dir.mkdir()
        
        main_py = src_dir / "main.py"
        main_py.write_text("""
def main():
    print("Hello, Python!")

if __name__ == "__main__":
    main()
""")
        
        test_py = src_dir / "test_main.py"
        test_py.write_text("""
import pytest
from main import main

def test_main():
    assert main() is None
""")
        
        self.files_created.extend([setup_py, requirements, main_py, test_py])
        return ['.py', '.txt']
    
    def create_typescript_project(self):
        """Create a TypeScript/Node.js project structure."""
        package_json = self.base_path / "package.json"
        package_json.write_text("""
{
  "name": "test-project",
  "version": "1.0.0",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js"
  },
  "devDependencies": {
    "typescript": "^4.9.0",
    "@types/node": "^18.0.0"
  }
}
""")
        
        tsconfig = self.base_path / "tsconfig.json"
        tsconfig.write_text("""
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
""")
        
        src_dir = self.base_path / "src"
        src_dir.mkdir()
        
        index_ts = src_dir / "index.ts"
        index_ts.write_text("""
interface User {
    name: string;
    age: number;
}

function greetUser(user: User): string {
    return `Hello, ${user.name}! You are ${user.age} years old.`;
}

console.log(greetUser({ name: "Test", age: 25 }));
""")
        
        self.files_created.extend([package_json, tsconfig, index_ts])
        return ['.ts', '.js', '.json']
    
    def create_mixed_project(self):
        """Create a mixed project with multiple languages and tools."""
        # Docker setup
        dockerfile = self.base_path / "Dockerfile"
        dockerfile.write_text("""
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
""")
        
        docker_compose = self.base_path / "docker-compose.yml"
        docker_compose.write_text("""
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: testdb
""")
        
        # Kubernetes
        k8s_dir = self.base_path / "k8s"
        k8s_dir.mkdir()
        
        deployment = k8s_dir / "deployment.yaml"
        deployment.write_text("""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: test-app
  template:
    metadata:
      labels:
        app: test-app
    spec:
      containers:
      - name: app
        image: test-app:latest
        ports:
        - containerPort: 3000
""")
        
        # Terraform
        main_tf = self.base_path / "main.tf"
        main_tf.write_text("""
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1d0"
  instance_type = "t2.micro"
  
  tags = {
    Name = "HelloWorld"
  }
}
""")
        
        # Build tools
        makefile = self.base_path / "Makefile"
        makefile.write_text("""
.PHONY: build test clean

build:
	docker build -t test-app .

test:
	pytest tests/

clean:
	docker rmi test-app
""")
        
        self.files_created.extend([dockerfile, docker_compose, deployment, main_tf, makefile])
        return ['.yml', '.yaml', '.tf', '.mk']
    
    def cleanup(self):
        """Clean up created files."""
        for file_path in self.files_created:
            try:
                if file_path.exists():
                    if file_path.is_file():
                        file_path.unlink()
                    else:
                        shutil.rmtree(file_path)
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory with cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir)
        yield workspace_path


@pytest.fixture
def mock_lsp_environment():
    """Mock environment with various LSPs available."""
    with patch('shutil.which') as mock_which:
        def which_side_effect(binary):
            available_lsps = {
                'rust-analyzer': '/usr/bin/rust-analyzer',
                'ruff': '/usr/local/bin/ruff',
                'typescript-language-server': '/usr/bin/typescript-language-server',
                'gopls': '/usr/bin/gopls',
                'clangd': '/usr/bin/clangd',
                'make': '/usr/bin/make',
                'cargo': '/usr/local/bin/cargo',
                'npm': '/usr/bin/npm',
                'docker': '/usr/bin/docker'
            }
            return available_lsps.get(binary)
        
        mock_which.side_effect = which_side_effect
        yield mock_which


@pytest.fixture
def mock_limited_environment():
    """Mock environment with limited LSPs available."""
    with patch('shutil.which') as mock_which:
        def which_side_effect(binary):
            available_tools = {
                'make': '/usr/bin/make',
                'docker': '/usr/bin/docker'
            }
            return available_tools.get(binary)
        
        mock_which.side_effect = which_side_effect
        yield mock_which


class TestLSPDetectionWorkflow:
    """Test complete LSP detection to file monitoring workflow."""
    
    def test_rust_project_detection_workflow(self, temp_workspace, mock_lsp_environment):
        """Test complete workflow for Rust project detection and configuration."""
        # Create project structure
        project = MockProjectStructure(temp_workspace)
        expected_extensions = project.create_rust_project()
        
        try:
            # Initialize components
            lsp_detector = LSPDetector()
            build_detector = BuildToolDetector()
            notification_manager = LSPNotificationManager()
            fallback_provider = FallbackExtensionProvider(
                lsp_detector=lsp_detector,
                build_tool_detector=build_detector
            )
            
            # Mock version detection
            with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
                with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                    # Step 1: Detect available LSPs
                    detection_result = lsp_detector.scan_available_lsps()
                    assert isinstance(detection_result, LSPDetectionResult)
                    assert 'rust-analyzer' in detection_result.detected_lsps
                    
                    # Step 2: Get comprehensive extension list
                    extensions = fallback_provider.get_comprehensive_extensions()
                    assert '.rs' in extensions
                    assert '.toml' in extensions
                    
                    # Step 3: Verify extension sources
                    sources = fallback_provider.get_extensions_with_sources()
                    assert '.rs' in sources['lsp_detected']  # From rust-analyzer
                    
                    # Step 4: Check priority ordering
                    priority_extensions = fallback_provider.get_priority_extensions()
                    rust_index = priority_extensions.index('.rs')
                    essential_index = priority_extensions.index('.md')
                    assert rust_index < essential_index  # LSP should have higher priority
                    
                    # Step 5: Test watch configuration integration
                    watch_config = WatchConfiguration(
                        watch_path=str(temp_workspace),
                        include_patterns=extensions[:50],  # Limit for performance
                        exclude_patterns=['.git', 'target', 'node_modules']
                    )
                    
                    effective_patterns = watch_config.get_effective_patterns()
                    assert any('.rs' in pattern for pattern in effective_patterns)
                    
                    # Step 6: Verify no notifications for supported types
                    rust_lsp = lsp_detector.get_lsp_for_extension('.rs')
                    assert rust_lsp is not None
                    assert rust_lsp.name == 'rust-analyzer'
                    
                    # But should notify for unsupported types
                    notification_result = notification_manager.notify_missing_lsp('.kt', 'kotlin-language-server')
                    assert notification_result is True  # Should send notification
                    
        finally:
            project.cleanup()
    
    def test_python_project_detection_workflow(self, temp_workspace, mock_lsp_environment):
        """Test complete workflow for Python project detection."""
        project = MockProjectStructure(temp_workspace)
        expected_extensions = project.create_python_project()
        
        try:
            # Initialize components
            lsp_detector = LSPDetector()
            fallback_provider = FallbackExtensionProvider(lsp_detector=lsp_detector)
            
            with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
                # Detect LSPs
                detection_result = lsp_detector.scan_available_lsps()
                assert 'ruff' in detection_result.detected_lsps
                
                # Get extensions
                extensions = fallback_provider.get_comprehensive_extensions()
                assert '.py' in extensions
                
                # Check Python LSP priority (ruff vs pyright vs pylsp)
                python_lsp = lsp_detector.get_lsp_for_extension('.py')
                assert python_lsp is not None
                assert python_lsp.name == 'ruff'  # Should prefer ruff based on our mocking
                
                # Verify build tool detection for Python
                build_detector = BuildToolDetector()
                with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                    build_result = build_detector.scan_build_tools()
                    # Should detect pip-related files
                    assert 'requirements.txt' in build_result.build_tool_extensions or \
                           any('requirements' in ext for ext in build_result.build_tool_extensions)
                    
        finally:
            project.cleanup()
    
    def test_mixed_project_detection_workflow(self, temp_workspace, mock_lsp_environment):
        """Test workflow for mixed project with multiple languages and tools."""
        project = MockProjectStructure(temp_workspace)
        expected_extensions = project.create_mixed_project()
        
        try:
            # Initialize comprehensive detection
            lsp_detector = LSPDetector()
            build_detector = BuildToolDetector()
            fallback_provider = FallbackExtensionProvider(
                lsp_detector=lsp_detector,
                build_tool_detector=build_detector,
                include_infrastructure=True
            )
            
            with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
                with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                    # Get comprehensive coverage
                    extensions = fallback_provider.get_comprehensive_extensions()
                    
                    # Should include infrastructure patterns
                    assert any('docker' in ext.lower() for ext in extensions)
                    assert '.tf' in extensions  # Terraform
                    assert '.yaml' in extensions or '.yml' in extensions  # Kubernetes
                    
                    # Check categorization
                    sources = fallback_provider.get_extensions_with_sources()
                    assert len(sources['infrastructure']) > 0
                    assert len(sources['build_tools']) > 0
                    
                    # Verify Docker patterns specifically
                    infra_extensions = build_detector.get_infrastructure_extensions()
                    assert any('dockerfile' in ext.lower() for ext in infra_extensions)
                    
        finally:
            project.cleanup()
    
    def test_fallback_behavior_without_lsps(self, temp_workspace, mock_limited_environment):
        """Test fallback behavior when no LSPs are available."""
        project = MockProjectStructure(temp_workspace)
        project.create_rust_project()  # Create Rust project
        project.create_python_project()  # Create Python project
        
        try:
            # Initialize with limited environment (no LSPs)
            lsp_detector = LSPDetector()
            build_detector = BuildToolDetector()
            fallback_provider = FallbackExtensionProvider(
                lsp_detector=lsp_detector,
                build_tool_detector=build_detector
            )
            
            with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                # Should still get reasonable coverage via fallbacks
                extensions = fallback_provider.get_comprehensive_extensions()
                
                # Should include essential extensions
                assert '.md' in extensions
                assert '.json' in extensions
                assert '.yaml' in extensions
                
                # Should include language fallbacks since no LSPs detected
                sources = fallback_provider.get_extensions_with_sources()
                assert len(sources['language_fallbacks']) > 0
                assert '.py' in sources['language_fallbacks']  # Python fallback
                assert '.rs' in sources['language_fallbacks']  # Rust fallback
                
                # Should include build tools that are available
                assert len(sources['build_tools']) > 0
                
                # Should gracefully handle missing LSPs
                detection_result = lsp_detector.scan_available_lsps()
                assert len(detection_result.detected_lsps) == 0
                assert detection_result.scan_duration >= 0
                
        finally:
            project.cleanup()


class TestNotificationSystemIntegration:
    """Test notification system integration with detection workflow."""
    
    def test_notification_workflow_with_missing_lsp(self, temp_workspace):
        """Test notification workflow when encountering unsupported file types."""
        # Create a project with unsupported file type
        kotlin_file = temp_workspace / "Main.kt"
        kotlin_file.write_text("""
fun main() {
    println("Hello, Kotlin!")
}
""")
        
        # Initialize notification system
        notification_manager = LSPNotificationManager(
            max_notifications_per_type=1,
            notification_cooldown=1
        )
        
        # Track notifications
        notifications_sent = []
        def notification_callback(entry):
            notifications_sent.append(entry)
        
        notification_manager.register_callback('test', notification_callback)
        
        # Simulate encountering unsupported file
        result = notification_manager.notify_missing_lsp('.kt', 'kotlin-language-server')
        assert result is True
        assert len(notifications_sent) == 1
        
        # Verify notification content
        notification = notifications_sent[0]
        assert notification.file_extension == '.kt'
        assert notification.lsp_name == 'kotlin-language-server'
        assert 'Installation options:' in notification.message
        
        # Test throttling
        result2 = notification_manager.notify_missing_lsp('.kt', 'kotlin-language-server')
        assert result2 is False  # Should be throttled
        
        # Test dismissal
        notification_manager.dismiss_file_type('.kt')
        time.sleep(1.1)  # Wait for cooldown
        result3 = notification_manager.notify_missing_lsp('.kt', 'kotlin-language-server')
        assert result3 is False  # Should be dismissed
    
    def test_notification_persistence(self, temp_workspace):
        """Test notification state persistence across sessions."""
        persist_file = temp_workspace / "notifications.json"
        
        # First session
        manager1 = LSPNotificationManager(persist_file=str(persist_file))
        manager1.notify_missing_lsp('.kt', 'kotlin-language-server')
        manager1.dismiss_file_type('.scala')
        
        # Second session
        manager2 = LSPNotificationManager(persist_file=str(persist_file))
        
        # Should load previous state
        assert '.scala' in manager2.dismissed_types
        history = manager2.get_notification_history()
        assert any(entry.file_extension == '.kt' for entry in history)
    
    def test_platform_specific_notifications(self):
        """Test platform-specific installation instructions."""
        notification_manager = LSPNotificationManager()
        
        # Test different LSPs have appropriate instructions
        lsps_to_test = ['rust-analyzer', 'ruff', 'typescript-language-server', 'gopls']
        
        for lsp in lsps_to_test:
            instructions = notification_manager._get_platform_specific_instructions(lsp)
            assert len(instructions) > 0
            
            # Should have package manager instructions
            assert any(key.startswith('via_') for key in instructions.keys())
            
            # Should have official documentation
            if 'official_docs' in instructions:
                assert 'http' in instructions['official_docs']


class TestPerformanceCharacteristics:
    """Test performance characteristics and resource usage."""
    
    def test_detection_performance(self, mock_lsp_environment):
        """Test detection system performance characteristics."""
        lsp_detector = LSPDetector()
        build_detector = BuildToolDetector()
        
        with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
            with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                # Measure LSP detection time
                start_time = time.time()
                lsp_result = lsp_detector.scan_available_lsps()
                lsp_duration = time.time() - start_time
                
                # Measure build tool detection time
                start_time = time.time()
                build_result = build_detector.scan_build_tools()
                build_duration = time.time() - start_time
                
                # Should complete within reasonable time
                assert lsp_duration < 5.0  # 5 seconds max
                assert build_duration < 5.0  # 5 seconds max
                
                # Cached calls should be faster
                start_time = time.time()
                lsp_result2 = lsp_detector.scan_available_lsps()
                cached_duration = time.time() - start_time
                
                assert cached_duration < lsp_duration / 2  # Should be significantly faster
    
    def test_extension_list_performance(self, mock_lsp_environment):
        """Test performance of extension list generation."""
        fallback_provider = FallbackExtensionProvider()
        
        # Measure comprehensive extension generation
        start_time = time.time()
        extensions = fallback_provider.get_comprehensive_extensions()
        duration = time.time() - start_time
        
        assert duration < 2.0  # Should be fast
        assert len(extensions) > 10  # Should find reasonable number of extensions
        assert len(extensions) < 500  # But not excessive
        
        # Test with different configurations
        start_time = time.time()
        priority_extensions = fallback_provider.get_priority_extensions(max_extensions=50)
        priority_duration = time.time() - start_time
        
        assert priority_duration < 1.0
        assert len(priority_extensions) <= 50
    
    def test_memory_usage_patterns(self):
        """Test memory usage doesn't grow excessively."""
        # Create multiple detector instances
        detectors = []
        for i in range(10):
            detector = LSPDetector()
            with patch('shutil.which', return_value=None):
                detector.scan_available_lsps()
            detectors.append(detector)
        
        # Each detector should have reasonable memory footprint
        for detector in detectors:
            assert len(detector.LSP_EXTENSION_MAP) < 50  # Reasonable mapping size
            if detector._cached_result:
                assert len(detector._cached_result.errors) < 100  # Don't accumulate excessive errors
    
    def test_concurrent_detection(self, mock_lsp_environment):
        """Test detection system under concurrent load."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def detect_worker():
            detector = LSPDetector()
            with patch.object(detector, '_get_lsp_version', return_value="1.0.0"):
                result = detector.scan_available_lsps()
                results.put(result)
        
        # Start multiple detection threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=detect_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # All should complete successfully
        assert results.qsize() == 5
        while not results.empty():
            result = results.get()
            assert isinstance(result, LSPDetectionResult)


class TestErrorRecoveryAndRobustness:
    """Test error recovery and robustness scenarios."""
    
    def test_binary_detection_failures(self):
        """Test graceful handling of binary detection failures."""
        detector = LSPDetector()
        
        # Test with which command failing
        with patch('shutil.which', side_effect=Exception("System error")):
            result = detector.scan_available_lsps()
            assert isinstance(result, LSPDetectionResult)
            # Should still return some extensions (essential ones)
            extensions = detector.get_supported_extensions()
            assert len(extensions) > 0
    
    def test_version_detection_failures(self, mock_lsp_environment):
        """Test handling of version detection failures."""
        detector = LSPDetector()
        
        # Mock version detection failures
        with patch.object(detector, '_get_lsp_version', side_effect=Exception("Version failed")):
            result = detector.scan_available_lsps()
            
            # Should still detect LSPs even without version info
            assert isinstance(result, LSPDetectionResult)
            # Should have some detected LSPs (with None versions)
            for lsp_info in result.detected_lsps.values():
                assert lsp_info.version is None  # Version detection failed
                assert lsp_info.binary_path is not None  # But binary was found
    
    def test_notification_system_resilience(self):
        """Test notification system resilience to failures."""
        manager = LSPNotificationManager()
        
        # Test with callback that raises exception
        def failing_callback(entry):
            raise Exception("Callback failed")
        
        manager.register_callback('failing', failing_callback)
        
        # Should handle callback failure gracefully
        result = manager.notify_missing_lsp('.test', 'test-lsp')
        assert isinstance(result, bool)  # Should not crash
    
    def test_configuration_failures(self):
        """Test handling of configuration system failures."""
        # Test detector with missing configuration
        detector = LSPDetector()
        assert detector.config is None  # Should handle missing config
        
        # Should still work with defaults
        extensions = detector.get_supported_extensions()
        assert isinstance(extensions, list)
        assert len(extensions) > 0
    
    def test_file_system_errors(self, temp_workspace):
        """Test handling of file system errors."""
        # Create notification manager with invalid persist file location
        invalid_path = temp_workspace / "nonexistent" / "deep" / "path" / "notifications.json"
        
        # Should handle invalid persist path gracefully
        manager = LSPNotificationManager(persist_file=str(invalid_path))
        result = manager.notify_missing_lsp('.test', 'test-lsp')
        assert isinstance(result, bool)


class TestCrossIntegrationScenarios:
    """Test integration across multiple components."""
    
    def test_daemon_integration_simulation(self, temp_workspace, mock_lsp_environment):
        """Simulate integration with daemon file watching system."""
        # Create project with multiple file types
        project = MockProjectStructure(temp_workspace)
        project.create_rust_project()
        project.create_python_project()
        project.create_mixed_project()
        
        try:
            # Initialize complete detection system
            lsp_detector = LSPDetector()
            build_detector = BuildToolDetector()
            notification_manager = LSPNotificationManager()
            fallback_provider = FallbackExtensionProvider(
                lsp_detector=lsp_detector,
                build_tool_detector=build_detector
            )
            
            with patch.object(lsp_detector, '_get_lsp_version', return_value="1.0.0"):
                with patch.object(build_detector, '_get_tool_version', return_value="1.0.0"):
                    # Step 1: Detect all available tools and extensions
                    comprehensive_extensions = fallback_provider.get_comprehensive_extensions()
                    
                    # Step 2: Create watch configuration
                    watch_config = WatchConfiguration(
                        watch_path=str(temp_workspace),
                        include_patterns=comprehensive_extensions[:100],  # Limit for test
                        exclude_patterns=['.git', 'target', 'node_modules', 'dist']
                    )
                    
                    # Step 3: Simulate file discovery
                    discovered_files = []
                    for file_path in project.files_created:
                        if file_path.is_file():
                            discovered_files.append(file_path)
                    
                    # Step 4: Check coverage
                    covered_files = 0
                    uncovered_files = []
                    
                    for file_path in discovered_files:
                        extension = file_path.suffix or file_path.name
                        if extension in comprehensive_extensions:
                            covered_files += 1
                        else:
                            uncovered_files.append(file_path)
                    
                    # Should have good coverage
                    coverage_ratio = covered_files / len(discovered_files) if discovered_files else 0
                    assert coverage_ratio > 0.8  # At least 80% coverage
                    
                    # Step 5: Handle uncovered files with notifications
                    for file_path in uncovered_files:
                        extension = file_path.suffix or file_path.name
                        # Would trigger notification in real system
                        notification_manager.notify_missing_lsp(extension, f'lsp-for-{extension}')
                    
                    # Step 6: Verify system state
                    stats = notification_manager.get_statistics()
                    summary = fallback_provider.get_summary()
                    
                    assert stats['total_notifications'] >= 0
                    assert summary['total_extensions'] > 0
                    
        finally:
            project.cleanup()
    
    def test_hot_reload_simulation(self, mock_lsp_environment):
        """Simulate hot reload of detection configuration."""
        detector = LSPDetector(cache_ttl=1)  # Short cache for testing
        
        with patch.object(detector, '_get_lsp_version', return_value="1.0.0"):
            # Initial detection
            initial_extensions = detector.get_supported_extensions()
            assert len(initial_extensions) > 0
            
            # Simulate LSP installation (cache should be cleared and redetected)
            detector.clear_cache()
            
            # Simulate new LSP available
            with patch('shutil.which') as mock_which:
                def enhanced_which(binary):
                    base_lsps = {
                        'rust-analyzer': '/usr/bin/rust-analyzer',
                        'ruff': '/usr/local/bin/ruff',
                    }
                    # Add new LSP
                    if binary == 'lua-language-server':
                        return '/usr/bin/lua-language-server'
                    return base_lsps.get(binary)
                
                mock_which.side_effect = enhanced_which
                
                # Re-detect with new LSP
                updated_extensions = detector.get_supported_extensions()
                
                # Should include new extensions
                assert '.lua' in updated_extensions or len(updated_extensions) >= len(initial_extensions)
    
    def test_configuration_validation_integration(self, temp_workspace):
        """Test integration with configuration validation."""
        # Create configuration that includes LSP detection settings
        config_data = {
            'watch_path': str(temp_workspace),
            'lsp_detection': {
                'enabled': True,
                'cache_ttl': 300,
                'detection_timeout': 5.0,
                'include_fallbacks': True,
                'include_build_tools': True,
                'include_infrastructure': True
            },
            'notification_settings': {
                'max_notifications_per_type': 3,
                'cooldown_seconds': 300,
                'handlers': ['console', 'file']
            }
        }
        
        # Simulate configuration loading and validation
        fallback_provider = FallbackExtensionProvider(
            include_language_fallbacks=config_data['lsp_detection']['include_fallbacks'],
            include_build_tools=config_data['lsp_detection']['include_build_tools'],
            include_infrastructure=config_data['lsp_detection']['include_infrastructure']
        )
        
        notification_manager = LSPNotificationManager(
            max_notifications_per_type=config_data['notification_settings']['max_notifications_per_type'],
            notification_cooldown=config_data['notification_settings']['cooldown_seconds']
        )
        
        # Test configuration is applied correctly
        assert fallback_provider.include_language_fallbacks is True
        assert fallback_provider.include_build_tools is True
        assert fallback_provider.include_infrastructure is True
        assert notification_manager.max_notifications_per_type == 3
        assert notification_manager.notification_cooldown == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])