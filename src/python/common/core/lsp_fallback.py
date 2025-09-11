"""
LSP Fallback and Default Handling System

This module provides fallback extension detection for build tools and
containerization files when no LSPs are detected, ensuring comprehensive
file coverage even in environments without LSP servers.
"""

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class BuildToolType(Enum):
    """Types of build tools that can be detected."""
    MAKE = "make"
    CMAKE = "cmake"
    GRADLE = "gradle"
    MAVEN = "maven"
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    PIP = "pip"
    POETRY = "poetry"
    CARGO = "cargo"
    COMPOSER = "composer"
    BUNDLER = "bundler"
    GO_MODULES = "go-modules"
    DENO = "deno"
    BUN = "bun"
    MESON = "meson"
    NINJA = "ninja"
    BAZEL = "bazel"
    BUCK = "buck"
    SCONS = "scons"


@dataclass
class BuildToolInfo:
    """Information about a detected build tool."""
    
    name: str
    tool_type: BuildToolType
    binary_path: str
    version: Optional[str] = None
    supported_extensions: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    detection_time: float = field(default_factory=time.time)


@dataclass
class FallbackDetectionResult:
    """Result of fallback extension detection."""
    
    detected_build_tools: Dict[str, BuildToolInfo] = field(default_factory=dict)
    essential_extensions: List[str] = field(default_factory=list)
    build_tool_extensions: List[str] = field(default_factory=list)
    infrastructure_extensions: List[str] = field(default_factory=list)
    language_fallback_extensions: List[str] = field(default_factory=list)
    total_extensions: List[str] = field(default_factory=list)
    scan_time: float = field(default_factory=time.time)
    scan_duration: float = 0.0


class BuildToolDetector:
    """
    Detects available build tools and their associated file extensions.
    Provides fallback extension support when LSP servers are not available.
    """
    
    # Essential extensions that should always be watched regardless of tool availability
    DEFAULT_FALLBACK_EXTENSIONS = [
        # Documentation and text files
        '.md', '.txt', '.rst', '.adoc', '.org',
        
        # Configuration files
        '.json', '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg', '.conf',
        
        # Shell and script files
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        
        # Data and database files
        '.sql', '.sqlite', '.db', '.csv', '.tsv', '.log',
        
        # Project and repository files
        '.gitignore', '.gitattributes', 'LICENSE', 'README*', 'CHANGELOG*',
        
        # Environment and configuration
        '.env', '.envrc', '.editorconfig',
    ]
    
    # Build tool detection configuration
    BUILD_TOOL_CONFIG = {
        BuildToolType.MAKE: {
            'binaries': ['make', 'gmake', 'mingw32-make'],
            'config_files': ['Makefile', 'makefile', 'GNUmakefile', '*.mk'],
            'extensions': ['.mk'],
            'version_args': ['--version']
        },
        BuildToolType.CMAKE: {
            'binaries': ['cmake'],
            'config_files': ['CMakeLists.txt', '*.cmake', '*.cmake.in'],
            'extensions': ['.cmake'],
            'version_args': ['--version']
        },
        BuildToolType.GRADLE: {
            'binaries': ['gradle', 'gradlew'],
            'config_files': ['build.gradle', 'build.gradle.kts', 'gradle.properties', 'settings.gradle'],
            'extensions': ['.gradle', '.gradle.kts'],
            'version_args': ['--version']
        },
        BuildToolType.MAVEN: {
            'binaries': ['mvn', 'maven'],
            'config_files': ['pom.xml', '*.pom'],
            'extensions': ['.pom'],
            'version_args': ['--version']
        },
        BuildToolType.NPM: {
            'binaries': ['npm'],
            'config_files': ['package.json', 'package-lock.json', '.npmrc', '.npmignore'],
            'extensions': ['.npmrc'],
            'version_args': ['--version']
        },
        BuildToolType.YARN: {
            'binaries': ['yarn'],
            'config_files': ['yarn.lock', '.yarnrc', '.yarnrc.yml', '.yarn.lock'],
            'extensions': ['.yarnrc'],
            'version_args': ['--version']
        },
        BuildToolType.PNPM: {
            'binaries': ['pnpm'],
            'config_files': ['pnpm-lock.yaml', '.pnpmrc', 'pnpm-workspace.yaml'],
            'extensions': ['.pnpmrc'],
            'version_args': ['--version']
        },
        BuildToolType.PIP: {
            'binaries': ['pip', 'pip3'],
            'config_files': ['requirements.txt', 'requirements-*.txt', 'setup.py', 'pyproject.toml', 'setup.cfg'],
            'extensions': [],  # No specific extensions, just config files
            'version_args': ['--version']
        },
        BuildToolType.POETRY: {
            'binaries': ['poetry'],
            'config_files': ['pyproject.toml', 'poetry.lock'],
            'extensions': [],
            'version_args': ['--version']
        },
        BuildToolType.CARGO: {
            'binaries': ['cargo'],
            'config_files': ['Cargo.toml', 'Cargo.lock'],
            'extensions': [],
            'version_args': ['--version']
        },
        BuildToolType.COMPOSER: {
            'binaries': ['composer'],
            'config_files': ['composer.json', 'composer.lock'],
            'extensions': [],
            'version_args': ['--version']
        },
        BuildToolType.BUNDLER: {
            'binaries': ['bundle', 'bundler'],
            'config_files': ['Gemfile', 'Gemfile.lock', '*.gemspec'],
            'extensions': ['.gemspec'],
            'version_args': ['--version']
        },
        BuildToolType.GO_MODULES: {
            'binaries': ['go'],
            'config_files': ['go.mod', 'go.sum', 'go.work'],
            'extensions': ['.mod', '.sum', '.work'],
            'version_args': ['version']
        },
        BuildToolType.DENO: {
            'binaries': ['deno'],
            'config_files': ['deno.json', 'deno.jsonc', 'deno.lock'],
            'extensions': ['.jsonc'],
            'version_args': ['--version']
        },
        BuildToolType.BUN: {
            'binaries': ['bun'],
            'config_files': ['bun.lockb', 'bunfig.toml'],
            'extensions': ['.lockb'],
            'version_args': ['--version']
        },
        BuildToolType.MESON: {
            'binaries': ['meson'],
            'config_files': ['meson.build', 'meson_options.txt'],
            'extensions': ['.build'],
            'version_args': ['--version']
        },
        BuildToolType.NINJA: {
            'binaries': ['ninja'],
            'config_files': ['build.ninja', '*.ninja'],
            'extensions': ['.ninja'],
            'version_args': ['--version']
        },
        BuildToolType.BAZEL: {
            'binaries': ['bazel', 'bazelisk'],
            'config_files': ['BUILD', 'BUILD.bazel', 'WORKSPACE', 'WORKSPACE.bazel', '*.bzl'],
            'extensions': ['.bzl', '.bazel'],
            'version_args': ['version']
        },
        BuildToolType.BUCK: {
            'binaries': ['buck'],
            'config_files': ['BUCK', '.buckconfig'],
            'extensions': [],
            'version_args': ['--version']
        },
        BuildToolType.SCONS: {
            'binaries': ['scons'],
            'config_files': ['SConstruct', 'SConscript'],
            'extensions': [],
            'version_args': ['--version']
        }
    }
    
    # Infrastructure and containerization tool patterns
    INFRASTRUCTURE_PATTERNS = {
        'docker': [
            'Dockerfile*', 'docker-compose*.yml', 'docker-compose*.yaml', 
            '.dockerignore', '.docker*'
        ],
        'kubernetes': [
            '*.k8s.yaml', '*.k8s.yml', 'kustomization.yaml', 'kustomization.yml',
            '*.kubernetes.yaml', '*.kubernetes.yml'
        ],
        'terraform': [
            '*.tf', '*.tfvars', '*.hcl', 'terraform.tfstate*', '*.tfstate',
            '.terraform*', 'terraform.rc', '.terraformrc'
        ],
        'ansible': [
            '*.yml', '*.yaml', 'hosts', 'ansible.cfg', 'playbook*.yml',
            'playbook*.yaml', 'inventory*'
        ],
        'vagrant': [
            'Vagrantfile', '.vagrant*'
        ],
        'helm': [
            'Chart.yaml', 'Chart.yml', 'values.yaml', 'values.yml',
            'requirements.yaml', 'requirements.yml'
        ],
        'pulumi': [
            'Pulumi.yaml', 'Pulumi.yml', 'Pulumi.*.yaml'
        ],
        'serverless': [
            'serverless.yml', 'serverless.yaml', '.serverless*'
        ]
    }
    
    def __init__(self, detection_timeout: float = 5.0, cache_ttl: int = 300):
        """
        Initialize build tool detector.
        
        Args:
            detection_timeout: Timeout for binary detection calls in seconds
            cache_ttl: Cache time-to-live in seconds
        """
        self.detection_timeout = detection_timeout
        self.cache_ttl = cache_ttl
        self._cached_result: Optional[FallbackDetectionResult] = None
        self._last_scan_time: float = 0.0
        
        logger.debug("Build tool detector initialized")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached detection result is still valid."""
        if self._cached_result is None:
            return False
        return (time.time() - self._last_scan_time) < self.cache_ttl
    
    def _check_binary_exists(self, binary_name: str) -> Optional[str]:
        """
        Check if a binary exists in PATH.
        
        Args:
            binary_name: Name of the binary to check
            
        Returns:
            Full path to binary if found, None otherwise
        """
        try:
            result = shutil.which(binary_name)
            if result:
                logger.debug(f"Found build tool binary: {binary_name} at {result}")
                return result
            return None
        except Exception as e:
            logger.debug(f"Error checking binary {binary_name}: {e}")
            return None
    
    def _get_tool_version(self, binary_path: str, version_args: List[str]) -> Optional[str]:
        """
        Get version information from a build tool binary.
        
        Args:
            binary_path: Full path to the binary
            version_args: Arguments to get version
            
        Returns:
            Version string if available, None otherwise
        """
        try:
            result = subprocess.run(
                [binary_path] + version_args,
                capture_output=True,
                text=True,
                timeout=self.detection_timeout
            )
            
            if result.returncode == 0:
                # Extract version from output (usually first line)
                version_output = result.stdout.strip().split('\n')[0]
                logger.debug(f"Got version for {binary_path}: {version_output}")
                return version_output
            else:
                logger.debug(f"Version check failed for {binary_path}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.debug(f"Version check timed out for {binary_path}")
            return None
        except Exception as e:
            logger.debug(f"Error getting version for {binary_path}: {e}")
            return None
    
    def scan_build_tools(self, force_refresh: bool = False) -> FallbackDetectionResult:
        """
        Scan for available build tools and infrastructure.
        
        Args:
            force_refresh: Force rescan even if cache is valid
            
        Returns:
            FallbackDetectionResult containing detected tools and extensions
        """
        if not force_refresh and self._is_cache_valid():
            logger.debug("Using cached build tool detection result")
            return self._cached_result
        
        logger.info("Scanning for available build tools...")
        start_time = time.time()
        
        result = FallbackDetectionResult()
        
        # Scan for build tools
        for tool_type, config in self.BUILD_TOOL_CONFIG.items():
            binary_path = None
            
            # Try all possible binary names
            for binary_name in config['binaries']:
                binary_path = self._check_binary_exists(binary_name)
                if binary_path:
                    break
            
            if binary_path:
                # Get version information
                version = self._get_tool_version(binary_path, config['version_args'])
                
                # Create BuildToolInfo
                tool_info = BuildToolInfo(
                    name=tool_type.value,
                    tool_type=tool_type,
                    binary_path=binary_path,
                    version=version,
                    supported_extensions=config['extensions'].copy(),
                    config_files=config['config_files'].copy()
                )
                
                result.detected_build_tools[tool_type.value] = tool_info
                logger.info(f"Detected build tool: {tool_type.value} at {binary_path}")
        
        # Compile extension lists
        result.essential_extensions = self.DEFAULT_FALLBACK_EXTENSIONS.copy()
        
        # Add build tool specific extensions
        build_tool_exts = set()
        for tool_info in result.detected_build_tools.values():
            build_tool_exts.update(tool_info.supported_extensions)
            build_tool_exts.update(tool_info.config_files)
        result.build_tool_extensions = sorted(list(build_tool_exts))
        
        # Add infrastructure extensions
        infra_exts = set()
        for patterns in self.INFRASTRUCTURE_PATTERNS.values():
            infra_exts.update(patterns)
        result.infrastructure_extensions = sorted(list(infra_exts))
        
        # Combine all extensions
        all_extensions = set()
        all_extensions.update(result.essential_extensions)
        all_extensions.update(result.build_tool_extensions)
        all_extensions.update(result.infrastructure_extensions)
        result.total_extensions = sorted(list(all_extensions))
        
        result.scan_duration = time.time() - start_time
        
        # Cache the result
        self._cached_result = result
        self._last_scan_time = time.time()
        
        logger.info(f"Build tool scan completed in {result.scan_duration:.2f}s, found {len(result.detected_build_tools)} tools")
        return result
    
    def get_fallback_extensions(self, force_refresh: bool = False) -> List[str]:
        """
        Get comprehensive list of fallback extensions.
        
        Args:
            force_refresh: Force rescan of build tools
            
        Returns:
            List of file extensions and patterns for fallback
        """
        detection_result = self.scan_build_tools(force_refresh)
        return detection_result.total_extensions
    
    def is_build_tool_available(self, tool_type: BuildToolType) -> bool:
        """
        Check if a specific build tool is available.
        
        Args:
            tool_type: Type of build tool to check
            
        Returns:
            True if the build tool is available, False otherwise
        """
        detection_result = self.scan_build_tools()
        return tool_type.value in detection_result.detected_build_tools
    
    def get_build_tool_info(self, tool_type: BuildToolType) -> Optional[BuildToolInfo]:
        """
        Get information about a specific build tool.
        
        Args:
            tool_type: Type of build tool
            
        Returns:
            BuildToolInfo if available, None otherwise
        """
        detection_result = self.scan_build_tools()
        return detection_result.detected_build_tools.get(tool_type.value)
    
    def get_infrastructure_extensions(self) -> List[str]:
        """Get all infrastructure and containerization extensions."""
        extensions = set()
        for patterns in self.INFRASTRUCTURE_PATTERNS.values():
            extensions.update(patterns)
        return sorted(list(extensions))
    
    def get_extensions_by_category(self) -> Dict[str, List[str]]:
        """
        Get extensions organized by category.
        
        Returns:
            Dictionary with categories as keys and extension lists as values
        """
        detection_result = self.scan_build_tools()
        
        return {
            'essential': detection_result.essential_extensions,
            'build_tools': detection_result.build_tool_extensions,
            'infrastructure': detection_result.infrastructure_extensions,
            'detected_tools': [info.name for info in detection_result.detected_build_tools.values()]
        }
    
    def clear_cache(self) -> None:
        """Clear the cached detection result."""
        self._cached_result = None
        self._last_scan_time = 0.0
        logger.debug("Build tool detection cache cleared")


class FallbackExtensionProvider:
    """
    Provides comprehensive fallback extension lists by combining LSP-detected
    extensions with build tool extensions and essential defaults.
    """
    
    def __init__(self, 
                 lsp_detector=None,
                 build_tool_detector: Optional[BuildToolDetector] = None,
                 include_language_fallbacks: bool = True,
                 include_build_tools: bool = True,
                 include_infrastructure: bool = True):
        """
        Initialize fallback extension provider.
        
        Args:
            lsp_detector: LSP detector instance (optional)
            build_tool_detector: Build tool detector instance (optional)
            include_language_fallbacks: Include language fallback extensions
            include_build_tools: Include build tool extensions
            include_infrastructure: Include infrastructure extensions
        """
        self.lsp_detector = lsp_detector
        self.build_tool_detector = build_tool_detector or BuildToolDetector()
        self.include_language_fallbacks = include_language_fallbacks
        self.include_build_tools = include_build_tools
        self.include_infrastructure = include_infrastructure
        
        logger.debug("Fallback extension provider initialized")
    
    def get_comprehensive_extensions(self, force_refresh: bool = False) -> List[str]:
        """
        Get comprehensive list of extensions from all sources.
        
        Args:
            force_refresh: Force refresh of all detection systems
            
        Returns:
            Comprehensive list of file extensions and patterns
        """
        all_extensions = set()
        
        # Add LSP-detected extensions if available
        if self.lsp_detector:
            try:
                lsp_extensions = self.lsp_detector.get_supported_extensions(
                    force_refresh=force_refresh,
                    include_fallbacks=False  # We'll add fallbacks ourselves
                )
                all_extensions.update(lsp_extensions)
                logger.debug(f"Added {len(lsp_extensions)} LSP-detected extensions")
            except Exception as e:
                logger.warning(f"Failed to get LSP extensions: {e}")
        
        # Add build tool and essential extensions
        fallback_result = self.build_tool_detector.scan_build_tools(force_refresh)
        
        # Always include essential extensions
        all_extensions.update(fallback_result.essential_extensions)
        
        if self.include_build_tools:
            all_extensions.update(fallback_result.build_tool_extensions)
        
        if self.include_infrastructure:
            all_extensions.update(fallback_result.infrastructure_extensions)
        
        # Add language fallbacks if no LSP detected for those languages
        if self.include_language_fallbacks and self.lsp_detector:
            try:
                # Get LSP-supported extensions to avoid duplicates
                lsp_extensions = set()
                if hasattr(self.lsp_detector, 'scan_available_lsps'):
                    detection_result = self.lsp_detector.scan_available_lsps()
                    for lsp_info in detection_result.detected_lsps.values():
                        lsp_extensions.update(lsp_info.supported_extensions)
                
                # Add fallbacks for unsupported languages
                if hasattr(self.lsp_detector, 'LANGUAGE_FALLBACKS'):
                    for language, fallback_exts in self.lsp_detector.LANGUAGE_FALLBACKS.items():
                        # Only add if no LSP covers these extensions
                        if not any(ext in lsp_extensions for ext in fallback_exts):
                            all_extensions.update(fallback_exts)
                            logger.debug(f"Added fallback extensions for {language}: {fallback_exts}")
            except Exception as e:
                logger.warning(f"Failed to add language fallbacks: {e}")
        
        return sorted(list(all_extensions))
    
    def get_extensions_with_sources(self, force_refresh: bool = False) -> Dict[str, Dict[str, List[str]]]:
        """
        Get extensions organized by source type.
        
        Args:
            force_refresh: Force refresh of detection systems
            
        Returns:
            Dictionary mapping source types to their extensions
        """
        sources = {
            'lsp_detected': [],
            'essential': [],
            'build_tools': [],
            'infrastructure': [],
            'language_fallbacks': []
        }
        
        # LSP extensions
        if self.lsp_detector:
            try:
                lsp_extensions = self.lsp_detector.get_supported_extensions(
                    force_refresh=force_refresh,
                    include_fallbacks=False
                )
                sources['lsp_detected'] = lsp_extensions
            except Exception as e:
                logger.warning(f"Failed to get LSP extensions: {e}")
        
        # Build tool and infrastructure extensions
        fallback_result = self.build_tool_detector.scan_build_tools(force_refresh)
        sources['essential'] = fallback_result.essential_extensions
        
        if self.include_build_tools:
            sources['build_tools'] = fallback_result.build_tool_extensions
        
        if self.include_infrastructure:
            sources['infrastructure'] = fallback_result.infrastructure_extensions
        
        # Language fallbacks
        if self.include_language_fallbacks and self.lsp_detector:
            try:
                lsp_extensions = set(sources['lsp_detected'])
                if hasattr(self.lsp_detector, 'LANGUAGE_FALLBACKS'):
                    fallback_exts = []
                    for language, lang_exts in self.lsp_detector.LANGUAGE_FALLBACKS.items():
                        if not any(ext in lsp_extensions for ext in lang_exts):
                            fallback_exts.extend(lang_exts)
                    sources['language_fallbacks'] = sorted(list(set(fallback_exts)))
            except Exception as e:
                logger.warning(f"Failed to get language fallbacks: {e}")
        
        return sources
    
    def get_priority_extensions(self, max_extensions: Optional[int] = None) -> List[str]:
        """
        Get prioritized list of most important extensions.
        
        Args:
            max_extensions: Maximum number of extensions to return
            
        Returns:
            List of prioritized extensions
        """
        # Priority order: LSP > Essential > Build Tools > Infrastructure > Language Fallbacks
        priority_extensions = []
        sources = self.get_extensions_with_sources()
        
        for source_type in ['lsp_detected', 'essential', 'build_tools', 'infrastructure', 'language_fallbacks']:
            priority_extensions.extend(sources.get(source_type, []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_extensions = []
        for ext in priority_extensions:
            if ext not in seen:
                seen.add(ext)
                unique_extensions.append(ext)
        
        if max_extensions:
            return unique_extensions[:max_extensions]
        
        return unique_extensions
    
    def is_extension_supported(self, extension: str) -> Tuple[bool, List[str]]:
        """
        Check if an extension is supported and by which sources.
        
        Args:
            extension: File extension to check
            
        Returns:
            Tuple of (is_supported, list_of_sources)
        """
        sources = self.get_extensions_with_sources()
        supporting_sources = []
        
        for source_type, extensions in sources.items():
            if extension in extensions:
                supporting_sources.append(source_type)
        
        return len(supporting_sources) > 0, supporting_sources
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of fallback extension provider state.
        
        Returns:
            Dictionary with summary information
        """
        sources = self.get_extensions_with_sources()
        build_tool_result = self.build_tool_detector.scan_build_tools()
        
        return {
            'total_extensions': len(self.get_comprehensive_extensions()),
            'sources_enabled': {
                'lsp_detector': self.lsp_detector is not None,
                'language_fallbacks': self.include_language_fallbacks,
                'build_tools': self.include_build_tools,
                'infrastructure': self.include_infrastructure
            },
            'extensions_by_source': {
                source: len(extensions) for source, extensions in sources.items()
            },
            'detected_build_tools': list(build_tool_result.detected_build_tools.keys()),
            'build_tool_scan_duration': build_tool_result.scan_duration
        }


# Global instances for convenient access
_default_build_tool_detector: Optional[BuildToolDetector] = None
_default_fallback_provider: Optional[FallbackExtensionProvider] = None


def get_default_build_tool_detector() -> BuildToolDetector:
    """Get the default global build tool detector instance."""
    global _default_build_tool_detector
    if _default_build_tool_detector is None:
        _default_build_tool_detector = BuildToolDetector()
    return _default_build_tool_detector


def get_default_fallback_provider() -> FallbackExtensionProvider:
    """Get the default global fallback extension provider instance."""
    global _default_fallback_provider
    if _default_fallback_provider is None:
        # Try to import LSP detector for integration
        lsp_detector = None
        try:
            from .lsp_detector import get_default_detector
            lsp_detector = get_default_detector()
        except ImportError:
            logger.debug("LSP detector not available for fallback provider")
        
        _default_fallback_provider = FallbackExtensionProvider(
            lsp_detector=lsp_detector,
            build_tool_detector=get_default_build_tool_detector()
        )
    return _default_fallback_provider


def get_fallback_extensions() -> List[str]:
    """Convenience function to get fallback extensions using default provider."""
    return get_default_fallback_provider().get_comprehensive_extensions()


def get_build_tool_extensions() -> List[str]:
    """Convenience function to get build tool extensions using default detector."""
    return get_default_build_tool_detector().get_fallback_extensions()