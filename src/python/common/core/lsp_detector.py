"""
LSP Detection and Management System

This module provides comprehensive LSP server detection capabilities,
including PATH scanning, extension mapping, and caching mechanisms.
"""

import asyncio
from loguru import logger
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Import configuration management
try:
    from .lsp_config import get_default_config, get_default_cache, LSPCacheEntry
except ImportError:
    # Fallback if config module is not available
    get_default_config = None
    get_default_cache = None
    LSPCacheEntry = None

# Import pattern management for ecosystem detection
try:
    from .pattern_manager import PatternManager
except ImportError:
    # Fallback if pattern manager is not available
    PatternManager = None

# logger imported from loguru


@dataclass
class LSPServerInfo:
    """Information about a detected LSP server."""
    
    name: str
    binary_path: str
    version: Optional[str] = None
    supported_extensions: List[str] = field(default_factory=list)
    priority: int = 0  # Higher priority = preferred for conflicting extensions
    capabilities: Set[str] = field(default_factory=set)
    detection_time: float = field(default_factory=time.time)


@dataclass
class LSPDetectionResult:
    """Result of LSP detection scan."""

    detected_lsps: Dict[str, LSPServerInfo] = field(default_factory=dict)
    scan_time: float = field(default_factory=time.time)
    scan_duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    # Ecosystem-aware detection metadata
    detected_ecosystems: List[str] = field(default_factory=list)
    ecosystem_lsp_recommendations: Dict[str, List[str]] = field(default_factory=dict)
    project_path: Optional[str] = None


class LSPDetector:
    """
    Core LSP detection system that scans PATH for available LSP servers
    and maintains mappings to file extensions.
    """
    
    # Essential extensions that should always be watched regardless of LSP availability
    ESSENTIAL_EXTENSIONS = [
        '.md', '.txt', '.rst', '.adoc',  # Documentation
        '.json', '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg',  # Configuration
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',  # Scripts
        '.sql', '.sqlite', '.db',  # Database
        '.log', '.csv', '.tsv',  # Data files
        '.dockerfile', 'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',  # Docker
        '.gitignore', '.gitattributes', 'LICENSE', 'README*',  # Project files
    ]
    
    # Build tool and configuration file patterns
    BUILD_TOOL_EXTENSIONS = {
        'make': ['.mk', 'Makefile', 'makefile', 'GNUmakefile'],
        'cmake': ['.cmake', 'CMakeLists.txt', '*.cmake.in'],
        'gradle': ['build.gradle', 'build.gradle.kts', 'gradle.properties', 'settings.gradle'],
        'maven': ['pom.xml', '*.pom'],
        'npm': ['package.json', 'package-lock.json', '.npmrc'],
        'yarn': ['yarn.lock', '.yarnrc', '.yarnrc.yml'],
        'pip': ['requirements.txt', 'requirements-*.txt', 'setup.py', 'pyproject.toml', 'Pipfile', 'Pipfile.lock'],
        'cargo': ['Cargo.toml', 'Cargo.lock'],
        'composer': ['composer.json', 'composer.lock'],
        'bundler': ['Gemfile', 'Gemfile.lock', '*.gemspec'],
        'go-modules': ['go.mod', 'go.sum'],
        'deno': ['deno.json', 'deno.jsonc'],
    }
    
    # Containerization and infrastructure patterns  
    INFRASTRUCTURE_EXTENSIONS = {
        'docker': ['Dockerfile*', 'docker-compose*.yml', 'docker-compose*.yaml', '.dockerignore'],
        'kubernetes': ['*.k8s.yaml', '*.k8s.yml', 'kustomization.yaml', 'kustomization.yml'],
        'terraform': ['*.tf', '*.tfvars', '*.hcl', 'terraform.tfstate*'],
        'ansible': ['*.yml', '*.yaml', 'hosts', 'ansible.cfg'],
        'vagrant': ['Vagrantfile'],
        'helm': ['Chart.yaml', 'Chart.yml', 'values.yaml', 'values.yml'],
    }
    
    # Fallback extensions for common languages when no LSP is available
    LANGUAGE_FALLBACKS = {
        'python': ['.py', '.pyi', '.pyx', '.pxd', '.pyw'],
        'javascript': ['.js', '.jsx', '.mjs', '.cjs'],
        'typescript': ['.ts', '.tsx', '.d.ts'],
        'rust': ['.rs'],
        'go': ['.go'],
        'java': ['.java', '.jar', '.class'],
        'c_cpp': ['.c', '.cpp', '.cc', '.cxx', '.c++', '.h', '.hpp', '.hxx', '.h++'],
        'csharp': ['.cs', '.csx', '.cake'],
        'php': ['.php', '.php3', '.php4', '.php5', '.phtml'],
        'ruby': ['.rb', '.rbw', '.rake', '.gemspec'],
        'swift': ['.swift'],
        'kotlin': ['.kt', '.kts'],
        'scala': ['.scala', '.sc'],
        'lua': ['.lua'],
        'perl': ['.pl', '.pm', '.t', '.pod'],
        'r': ['.r', '.R', '.rmd', '.Rmd'],
        'matlab': ['.m', '.mlx'],
        'julia': ['.jl'],
        'dart': ['.dart'],
        'elixir': ['.ex', '.exs'],
        'erlang': ['.erl', '.hrl'],
        'haskell': ['.hs', '.lhs'],
        'ocaml': ['.ml', '.mli', '.mll', '.mly'],
        'fsharp': ['.fs', '.fsi', '.fsx'],
        'clojure': ['.clj', '.cljs', '.cljc', '.edn'],
        'zig': ['.zig'],
        'nim': ['.nim', '.nims'],
        'crystal': ['.cr'],
        'v': ['.v'],
        'odin': ['.odin'],
    }
    
    # Comprehensive mapping of LSP servers to file extensions
    LSP_EXTENSION_MAP = {
        'rust-analyzer': {
            'extensions': ['.rs', '.toml'],
            'priority': 10,
            'alternative_names': ['rust-analyzer', 'ra_lsp_server'],
            'capabilities': {'semantic_tokens', 'hover', 'completion', 'diagnostics'}
        },
        'ruff': {
            'extensions': ['.py', '.pyi'],
            'priority': 8,
            'alternative_names': ['ruff', 'ruff-lsp'],
            'capabilities': {'diagnostics', 'formatting', 'code_actions'}
        },
        'typescript-language-server': {
            'extensions': ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'],
            'priority': 10,
            'alternative_names': ['typescript-language-server', 'tsserver'],
            'capabilities': {'semantic_tokens', 'hover', 'completion', 'diagnostics', 'formatting'}
        },
        'pyright': {
            'extensions': ['.py', '.pyi'],
            'priority': 9,
            'alternative_names': ['pyright-langserver', 'pyright'],
            'capabilities': {'semantic_tokens', 'hover', 'completion', 'diagnostics'}
        },
        'pylsp': {
            'extensions': ['.py', '.pyi'],
            'priority': 7,
            'alternative_names': ['pylsp', 'python-lsp-server'],
            'capabilities': {'hover', 'completion', 'diagnostics', 'formatting'}
        },
        'gopls': {
            'extensions': ['.go', '.mod', '.sum'],
            'priority': 10,
            'alternative_names': ['gopls'],
            'capabilities': {'semantic_tokens', 'hover', 'completion', 'diagnostics', 'formatting'}
        },
        'clangd': {
            'extensions': ['.c', '.cpp', '.cc', '.cxx', '.c++', '.h', '.hpp', '.hxx', '.h++'],
            'priority': 10,
            'alternative_names': ['clangd'],
            'capabilities': {'semantic_tokens', 'hover', 'completion', 'diagnostics'}
        },
        'java-language-server': {
            'extensions': ['.java'],
            'priority': 8,
            'alternative_names': ['java-language-server', 'jdtls'],
            'capabilities': {'semantic_tokens', 'hover', 'completion', 'diagnostics'}
        },
        'lua-language-server': {
            'extensions': ['.lua'],
            'priority': 8,
            'alternative_names': ['lua-language-server', 'lua-lsp'],
            'capabilities': {'hover', 'completion', 'diagnostics'}
        },
        'zls': {
            'extensions': ['.zig'],
            'priority': 9,
            'alternative_names': ['zls'],
            'capabilities': {'semantic_tokens', 'hover', 'completion', 'diagnostics'}
        },
        'ocaml-lsp': {
            'extensions': ['.ml', '.mli', '.mll', '.mly'],
            'priority': 9,
            'alternative_names': ['ocamllsp', 'ocaml-lsp'],
            'capabilities': {'hover', 'completion', 'diagnostics'}
        },
        'haskell-language-server': {
            'extensions': ['.hs', '.lhs'],
            'priority': 9,
            'alternative_names': ['haskell-language-server', 'hls'],
            'capabilities': {'hover', 'completion', 'diagnostics'}
        }
    }
    
    def __init__(self, cache_ttl: int = 300, detection_timeout: float = 5.0, config_file: Optional[str] = None, pattern_manager: Optional['PatternManager'] = None):
        """
        Initialize LSP detector.

        Args:
            cache_ttl: Cache time-to-live in seconds (overridden by config if available)
            detection_timeout: Timeout for binary detection calls in seconds (overridden by config)
            config_file: Optional path to configuration file
            pattern_manager: Optional PatternManager instance for ecosystem detection
        """
        # Initialize with defaults
        self.cache_ttl = cache_ttl
        self.detection_timeout = detection_timeout
        self._cached_result: Optional[LSPDetectionResult] = None
        self._last_scan_time: float = 0.0

        # Initialize pattern manager for ecosystem-aware detection
        self.pattern_manager = pattern_manager
        if self.pattern_manager is None and PatternManager is not None:
            try:
                self.pattern_manager = PatternManager()
                logger.debug("Initialized pattern manager for ecosystem detection")
            except Exception as e:
                logger.warning(f"Failed to initialize pattern manager: {e}")

        # Try to load configuration
        self.config = None
        self.cache = None

        if get_default_config is not None:
            try:
                self.config = get_default_config(config_file)
                # Override defaults with config values
                self.cache_ttl = self.config.cache_ttl
                self.detection_timeout = self.config.detection_timeout

                # Initialize cache if available
                if get_default_cache is not None:
                    self.cache = get_default_cache()

                logger.debug("LSP detector initialized with configuration management")
            except Exception as e:
                logger.warning(f"Failed to initialize LSP configuration: {e}")
                # Continue with defaults
    
    def _is_cache_valid(self) -> bool:
        """Check if cached detection result is still valid."""
        if self._cached_result is None:
            return False
        return (time.time() - self._last_scan_time) < self.cache_ttl
    
    def _load_from_persistent_cache(self) -> Optional[LSPDetectionResult]:
        """Load detection result from persistent cache if valid."""
        if not self.cache:
            return None
        
        try:
            result = LSPDetectionResult()
            valid_entries = 0
            
            for lsp_name in self.LSP_EXTENSION_MAP.keys():
                cache_entry = self.cache.get(lsp_name, self.cache_ttl)
                if cache_entry:
                    server_info = LSPServerInfo(
                        name=cache_entry.lsp_name,
                        binary_path=cache_entry.binary_path,
                        version=cache_entry.version,
                        supported_extensions=cache_entry.supported_extensions,
                        priority=cache_entry.priority,
                        capabilities=cache_entry.capabilities,
                        detection_time=cache_entry.detection_time
                    )
                    result.detected_lsps[lsp_name] = server_info
                    valid_entries += 1
            
            if valid_entries > 0:
                result.scan_time = time.time()
                result.scan_duration = 0.0  # From cache
                logger.debug(f"Loaded {valid_entries} LSPs from persistent cache")
                return result
            
        except Exception as e:
            logger.warning(f"Failed to load from persistent cache: {e}")
        
        return None

    def _check_binary_exists(self, binary_name: str) -> Optional[str]:
        """
        Check if a binary exists in PATH using subprocess with timeout.
        
        Args:
            binary_name: Name of the binary to check
            
        Returns:
            Full path to binary if found, None otherwise
        """
        try:
            # Use shutil.which for cross-platform compatibility
            result = shutil.which(binary_name)
            if result:
                logger.debug(f"Found LSP binary: {binary_name} at {result}")
                return result
            return None
        except Exception as e:
            logger.debug(f"Error checking binary {binary_name}: {e}")
            return None
    
    def _get_lsp_version(self, binary_path: str, lsp_name: str) -> Optional[str]:
        """
        Attempt to get version information from LSP binary.
        
        Args:
            binary_path: Full path to the LSP binary
            lsp_name: Name of the LSP server
            
        Returns:
            Version string if available, None otherwise
        """
        version_args_map = {
            'rust-analyzer': ['--version'],
            'ruff': ['--version'],
            'typescript-language-server': ['--version'],
            'pyright': ['--version'],
            'pylsp': ['--version'],
            'gopls': ['version'],
            'clangd': ['--version'],
            'java-language-server': ['--version'],
            'lua-language-server': ['--version'],
            'zls': ['--version'],
            'ocaml-lsp': ['--version'],
            'haskell-language-server': ['--version']
        }
        
        version_args = version_args_map.get(lsp_name, ['--version'])
        
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
                logger.debug(f"Got version for {lsp_name}: {version_output}")
                return version_output
            else:
                logger.debug(f"Version check failed for {lsp_name}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.debug(f"Version check timed out for {lsp_name}")
            return None
        except Exception as e:
            logger.debug(f"Error getting version for {lsp_name}: {e}")
            return None
    
    def scan_available_lsps(self, force_refresh: bool = False, project_path: Optional[Union[str, Path]] = None) -> LSPDetectionResult:
        """
        Scan PATH for available LSP servers, using persistent cache when available.

        Args:
            force_refresh: Force rescan even if cache is valid
            project_path: Optional project path for ecosystem-aware detection

        Returns:
            LSPDetectionResult containing detected LSP servers with ecosystem metadata
        """
        # Check memory cache first
        if not force_refresh and self._is_cache_valid():
            logger.debug("Using cached LSP detection result")
            return self._cached_result
        
        # Check persistent cache if available
        if not force_refresh and self.cache is not None:
            cached_result = self._load_from_persistent_cache()
            if cached_result:
                logger.debug("Using persistent cached LSP detection result")
                self._cached_result = cached_result
                self._last_scan_time = time.time()
                return cached_result
        
        logger.info("Scanning for available LSP servers...")
        start_time = time.time()
        
        result = LSPDetectionResult()
        
        for lsp_name, lsp_info in self.LSP_EXTENSION_MAP.items():
            # Check if LSP is enabled in configuration
            if self.config and not self.config.is_lsp_enabled(lsp_name):
                logger.debug(f"LSP {lsp_name} is disabled in configuration, skipping")
                continue
            
            # Check for custom binary path first
            custom_path = None
            if self.config:
                custom_path = self.config.get_lsp_binary_path(lsp_name)
            
            binary_path = None
            if custom_path:
                # Use custom path if configured
                if self._check_binary_exists(custom_path):
                    binary_path = custom_path
                    logger.debug(f"Using custom path for {lsp_name}: {custom_path}")
                else:
                    logger.warning(f"Custom path for {lsp_name} not found: {custom_path}")
            else:
                # Try all alternative names for this LSP
                for binary_name in lsp_info['alternative_names']:
                    binary_path = self._check_binary_exists(binary_name)
                    if binary_path:
                        break  # Found this LSP, no need to check other names
            
            if binary_path:
                # Get version information
                version = self._get_lsp_version(binary_path, lsp_name)
                
                # Create LSPServerInfo
                server_info = LSPServerInfo(
                    name=lsp_name,
                    binary_path=binary_path,
                    version=version,
                    supported_extensions=lsp_info['extensions'].copy(),
                    priority=lsp_info['priority'],
                    capabilities=lsp_info['capabilities'].copy()
                )
                
                result.detected_lsps[lsp_name] = server_info
                logger.info(f"Detected LSP: {lsp_name} at {binary_path}")
                
                # Store in persistent cache if available
                if self.cache and LSPCacheEntry:
                    cache_entry = LSPCacheEntry(
                        lsp_name=lsp_name,
                        binary_path=binary_path,
                        version=version,
                        supported_extensions=lsp_info['extensions'].copy(),
                        priority=lsp_info['priority'],
                        capabilities=lsp_info['capabilities'].copy()
                    )
                    self.cache.set(lsp_name, cache_entry)
        
        result.scan_duration = time.time() - start_time

        # Add ecosystem-aware information if project path is provided
        if project_path and self.pattern_manager:
            result.project_path = str(project_path)
            result.detected_ecosystems = self._detect_project_ecosystems(project_path)
            result.ecosystem_lsp_recommendations = self._generate_ecosystem_recommendations(result.detected_ecosystems, result.detected_lsps)
            logger.info(f"Detected ecosystems: {result.detected_ecosystems}")

        # Cache the result in memory
        self._cached_result = result
        self._last_scan_time = time.time()

        logger.info(f"LSP scan completed in {result.scan_duration:.2f}s, found {len(result.detected_lsps)} LSPs")
        return result
    
    def get_supported_extensions(self, force_refresh: bool = False, include_fallbacks: Optional[bool] = None) -> List[str]:
        """
        Get list of file extensions supported by detected LSP servers.
        
        Args:
            force_refresh: Force rescan of LSP servers
            include_fallbacks: Include fallback extensions and build tools (uses config default if None)
            
        Returns:
            List of file extensions (including the dot)
        """
        detection_result = self.scan_available_lsps(force_refresh)
        
        # Use configuration values if not explicitly specified
        if include_fallbacks is None:
            include_fallbacks = True  # Default
            if self.config:
                include_fallbacks = self.config.include_fallbacks
        
        include_build_tools = True
        include_infrastructure = True
        if self.config:
            include_build_tools = self.config.include_build_tools
            include_infrastructure = self.config.include_infrastructure
        
        # Start with essential extensions
        extensions = set(self.ESSENTIAL_EXTENSIONS)
        
        # Add extensions from detected LSPs
        for lsp_info in detection_result.detected_lsps.values():
            extensions.update(lsp_info.supported_extensions)
        
        if include_fallbacks:
            # Add fallback extensions for languages without detected LSPs
            detected_extensions = set()
            for lsp_info in detection_result.detected_lsps.values():
                detected_extensions.update(lsp_info.supported_extensions)
            
            # Add fallbacks for languages that don't have LSPs detected
            for language, fallback_exts in self.LANGUAGE_FALLBACKS.items():
                # Check if any fallback extension is already covered by LSPs
                if not any(ext in detected_extensions for ext in fallback_exts):
                    extensions.update(fallback_exts)
            
            # Add build tool extensions if enabled
            if include_build_tools:
                for tool_exts in self.BUILD_TOOL_EXTENSIONS.values():
                    extensions.update(tool_exts)
            
            # Add infrastructure extensions if enabled
            if include_infrastructure:
                for infra_exts in self.INFRASTRUCTURE_EXTENSIONS.values():
                    extensions.update(infra_exts)
        
        return sorted(list(extensions))
    
    def get_lsp_for_extension(self, extension: str) -> Optional[LSPServerInfo]:
        """
        Get the best LSP server for a given file extension.
        
        Args:
            extension: File extension (including the dot)
            
        Returns:
            LSPServerInfo for the best LSP, or None if no LSP supports this extension
        """
        detection_result = self.scan_available_lsps()
        
        # Find all LSPs that support this extension
        candidates = []
        for lsp_info in detection_result.detected_lsps.values():
            if extension in lsp_info.supported_extensions:
                candidates.append(lsp_info)
        
        if not candidates:
            return None
        
        # Return the LSP with highest priority
        return max(candidates, key=lambda lsp: lsp.priority)
    
    def get_extension_lsp_map(self) -> Dict[str, LSPServerInfo]:
        """
        Get mapping of file extensions to their best LSP servers.
        
        Returns:
            Dictionary mapping extensions to LSPServerInfo objects
        """
        detection_result = self.scan_available_lsps()
        extension_map = {}
        
        # Build mapping with priority-based selection
        for lsp_info in detection_result.detected_lsps.values():
            for extension in lsp_info.supported_extensions:
                current_lsp = extension_map.get(extension)
                if current_lsp is None or lsp_info.priority > current_lsp.priority:
                    extension_map[extension] = lsp_info
        
        return extension_map
    
    def is_lsp_available(self, lsp_name: str) -> bool:
        """
        Check if a specific LSP server is available.
        
        Args:
            lsp_name: Name of the LSP server to check
            
        Returns:
            True if the LSP is available, False otherwise
        """
        detection_result = self.scan_available_lsps()
        return lsp_name in detection_result.detected_lsps
    
    def clear_cache(self) -> None:
        """Clear the cached detection result."""
        self._cached_result = None
        self._last_scan_time = 0.0
        logger.debug("LSP detection cache cleared")
    
    def get_fallback_extensions_for_language(self, language: str) -> List[str]:
        """
        Get fallback extensions for a specific language.
        
        Args:
            language: Language name (e.g., 'python', 'javascript')
            
        Returns:
            List of fallback extensions for the language
        """
        return self.LANGUAGE_FALLBACKS.get(language, [])
    
    def get_build_tool_extensions(self) -> List[str]:
        """
        Get all build tool and configuration file extensions.
        
        Returns:
            List of build tool related extensions
        """
        extensions = set()
        for tool_exts in self.BUILD_TOOL_EXTENSIONS.values():
            extensions.update(tool_exts)
        return sorted(list(extensions))
    
    def get_infrastructure_extensions(self) -> List[str]:
        """
        Get all infrastructure and containerization file extensions.
        
        Returns:
            List of infrastructure related extensions
        """
        extensions = set()
        for infra_exts in self.INFRASTRUCTURE_EXTENSIONS.values():
            extensions.update(infra_exts)
        return sorted(list(extensions))
    
    def get_extensions_by_category(self) -> Dict[str, List[str]]:
        """
        Get extensions organized by category.
        
        Returns:
            Dictionary with categories as keys and extension lists as values
        """
        detection_result = self.scan_available_lsps()
        
        categories = {
            'lsp_supported': [],
            'essential': list(self.ESSENTIAL_EXTENSIONS),
            'build_tools': self.get_build_tool_extensions(),
            'infrastructure': self.get_infrastructure_extensions(),
            'language_fallbacks': []
        }
        
        # Get LSP supported extensions
        for lsp_info in detection_result.detected_lsps.values():
            categories['lsp_supported'].extend(lsp_info.supported_extensions)
        categories['lsp_supported'] = sorted(list(set(categories['lsp_supported'])))
        
        # Get fallback extensions for languages without LSPs
        detected_extensions = set(categories['lsp_supported'])
        for language, fallback_exts in self.LANGUAGE_FALLBACKS.items():
            if not any(ext in detected_extensions for ext in fallback_exts):
                categories['language_fallbacks'].extend(fallback_exts)
        categories['language_fallbacks'] = sorted(list(set(categories['language_fallbacks'])))
        
        return categories
    
    def get_priority_ordered_lsps(self) -> List[Tuple[str, LSPServerInfo]]:
        """
        Get detected LSPs ordered by priority (highest first).
        
        Returns:
            List of tuples containing (lsp_name, LSPServerInfo) ordered by priority
        """
        detection_result = self.scan_available_lsps()
        
        lsp_list = [(name, info) for name, info in detection_result.detected_lsps.items()]
        return sorted(lsp_list, key=lambda x: x[1].priority, reverse=True)
    
    def get_detection_summary(self) -> Dict[str, any]:
        """
        Get a comprehensive summary of the current LSP detection state.

        Returns:
            Dictionary with detection summary information
        """
        detection_result = self.scan_available_lsps()
        categories = self.get_extensions_by_category()

        summary = {
            'total_lsps_detected': len(detection_result.detected_lsps),
            'total_extensions_supported': len(self.get_supported_extensions()),
            'scan_duration': detection_result.scan_duration,
            'cache_age': time.time() - self._last_scan_time,
            'extensions_by_category': {
                category: len(extensions) for category, extensions in categories.items()
            },
            'detected_lsps': {
                name: {
                    'binary_path': info.binary_path,
                    'version': info.version,
                    'extensions': info.supported_extensions,
                    'priority': info.priority,
                    'capabilities': list(info.capabilities)
                }
                for name, info in detection_result.detected_lsps.items()
            },
            'priority_order': [name for name, _ in self.get_priority_ordered_lsps()],
            'fallback_languages_available': list(self.LANGUAGE_FALLBACKS.keys()),
            'build_tools_supported': list(self.BUILD_TOOL_EXTENSIONS.keys()),
            'infrastructure_tools_supported': list(self.INFRASTRUCTURE_EXTENSIONS.keys())
        }

        # Add ecosystem information if available
        if detection_result.detected_ecosystems:
            summary['detected_ecosystems'] = detection_result.detected_ecosystems
        if detection_result.ecosystem_lsp_recommendations:
            summary['ecosystem_lsp_recommendations'] = detection_result.ecosystem_lsp_recommendations
        if detection_result.project_path:
            summary['project_path'] = detection_result.project_path

        return summary


    def _detect_project_ecosystems(self, project_path: Union[str, Path]) -> List[str]:
        """
        Detect project ecosystems using the pattern manager.

        Args:
            project_path: Path to the project directory

        Returns:
            List of detected ecosystem names
        """
        if not self.pattern_manager:
            logger.debug("Pattern manager not available for ecosystem detection")
            return []

        try:
            ecosystems = self.pattern_manager.detect_ecosystem(project_path)
            logger.debug(f"Detected ecosystems in {project_path}: {ecosystems}")
            return ecosystems
        except Exception as e:
            logger.warning(f"Failed to detect ecosystems for {project_path}: {e}")
            return []

    def _generate_ecosystem_recommendations(self, ecosystems: List[str], detected_lsps: Dict[str, LSPServerInfo]) -> Dict[str, List[str]]:
        """
        Generate LSP recommendations based on detected ecosystems.

        Args:
            ecosystems: List of detected ecosystems
            detected_lsps: Dictionary of available LSP servers

        Returns:
            Dictionary mapping ecosystems to recommended LSP names
        """
        recommendations = {}

        # Ecosystem to LSP mapping based on common patterns
        ecosystem_lsp_map = {
            'python': ['ruff', 'pyright', 'pylsp'],
            'javascript': ['typescript-language-server'],
            'typescript': ['typescript-language-server'],
            'node': ['typescript-language-server'],
            'rust': ['rust-analyzer'],
            'go': ['gopls'],
            'c': ['clangd'],
            'cpp': ['clangd'],
            'java': ['java-language-server'],
            'lua': ['lua-language-server'],
            'zig': ['zls'],
            'ocaml': ['ocaml-lsp'],
            'haskell': ['haskell-language-server'],
            'kotlin': [],  # Add when we have Kotlin LSP
            'scala': [],   # Add when we have Scala LSP
            'swift': [],   # Add when we have Swift LSP
        }

        for ecosystem in ecosystems:
            recommended_lsps = []
            potential_lsps = ecosystem_lsp_map.get(ecosystem.lower(), [])

            # Filter to only include LSPs that are actually detected/available
            for lsp_name in potential_lsps:
                if lsp_name in detected_lsps:
                    recommended_lsps.append(lsp_name)

            if recommended_lsps:
                # Sort by priority (higher priority first)
                recommended_lsps.sort(
                    key=lambda lsp: detected_lsps[lsp].priority,
                    reverse=True
                )
                recommendations[ecosystem] = recommended_lsps
                logger.debug(f"Ecosystem {ecosystem} -> recommended LSPs: {recommended_lsps}")

        return recommendations

    def get_ecosystem_aware_lsps(self, project_path: Union[str, Path], force_refresh: bool = False) -> LSPDetectionResult:
        """
        Get LSP detection results with ecosystem-aware recommendations.

        Args:
            project_path: Path to the project directory
            force_refresh: Force rescan even if cache is valid

        Returns:
            LSPDetectionResult with ecosystem metadata and recommendations
        """
        return self.scan_available_lsps(force_refresh=force_refresh, project_path=project_path)

    def get_recommended_lsps_for_project(self, project_path: Union[str, Path]) -> Dict[str, LSPServerInfo]:
        """
        Get recommended LSP servers for a specific project.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary of recommended LSP servers with their info
        """
        detection_result = self.get_ecosystem_aware_lsps(project_path)
        recommended_lsps = {}

        # Add LSPs recommended by ecosystem analysis
        for ecosystem, lsp_names in detection_result.ecosystem_lsp_recommendations.items():
            for lsp_name in lsp_names:
                if lsp_name in detection_result.detected_lsps:
                    recommended_lsps[lsp_name] = detection_result.detected_lsps[lsp_name]

        # If no ecosystem-specific recommendations, fall back to highest priority LSPs
        if not recommended_lsps:
            # Get top 3 highest priority LSPs as fallback
            priority_lsps = self.get_priority_ordered_lsps()
            for name, info in priority_lsps[:3]:
                recommended_lsps[name] = info

        logger.info(f"Recommended LSPs for {project_path}: {list(recommended_lsps.keys())}")
        return recommended_lsps

    def get_lsp_for_extension_with_context(self, extension: str, project_path: Optional[Union[str, Path]] = None) -> Optional[LSPServerInfo]:
        """
        Get the best LSP server for a file extension with project context.

        Args:
            extension: File extension (including the dot)
            project_path: Optional project path for ecosystem-aware selection

        Returns:
            LSPServerInfo for the best LSP, or None if no LSP supports this extension
        """
        if project_path:
            # Use ecosystem-aware detection
            detection_result = self.get_ecosystem_aware_lsps(project_path)

            # First try to find LSPs from ecosystem recommendations
            for ecosystem, lsp_names in detection_result.ecosystem_lsp_recommendations.items():
                for lsp_name in lsp_names:
                    lsp_info = detection_result.detected_lsps.get(lsp_name)
                    if lsp_info and extension in lsp_info.supported_extensions:
                        logger.debug(f"Selected {lsp_name} for {extension} (ecosystem: {ecosystem})")
                        return lsp_info

        # Fall back to standard extension-based selection
        return self.get_lsp_for_extension(extension)


# Global instance for convenient access
_default_detector: Optional[LSPDetector] = None


def get_default_detector() -> LSPDetector:
    """Get the default global LSP detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = LSPDetector()
    return _default_detector


def scan_lsps() -> LSPDetectionResult:
    """Convenience function to scan for LSPs using the default detector."""
    return get_default_detector().scan_available_lsps()


def get_supported_extensions() -> List[str]:
    """Convenience function to get supported extensions from default detector."""
    return get_default_detector().get_supported_extensions()


def scan_lsps_for_project(project_path: Union[str, Path]) -> LSPDetectionResult:
    """Convenience function to scan LSPs with ecosystem awareness for a project."""
    return get_default_detector().get_ecosystem_aware_lsps(project_path)


def get_recommended_lsps(project_path: Union[str, Path]) -> Dict[str, LSPServerInfo]:
    """Convenience function to get recommended LSPs for a project."""
    return get_default_detector().get_recommended_lsps_for_project(project_path)


def get_lsp_with_context(extension: str, project_path: Optional[Union[str, Path]] = None) -> Optional[LSPServerInfo]:
    """Convenience function to get LSP for extension with project context."""
    return get_default_detector().get_lsp_for_extension_with_context(extension, project_path)