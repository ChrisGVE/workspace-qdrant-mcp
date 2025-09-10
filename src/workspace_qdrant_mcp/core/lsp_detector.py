"""
LSP Detection and Management System

This module provides comprehensive LSP server detection capabilities,
including PATH scanning, extension mapping, and caching mechanisms.
"""

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


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


class LSPDetector:
    """
    Core LSP detection system that scans PATH for available LSP servers
    and maintains mappings to file extensions.
    """
    
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
    
    def __init__(self, cache_ttl: int = 300, detection_timeout: float = 5.0):
        """
        Initialize LSP detector.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            detection_timeout: Timeout for binary detection calls in seconds
        """
        self.cache_ttl = cache_ttl
        self.detection_timeout = detection_timeout
        self._cached_result: Optional[LSPDetectionResult] = None
        self._last_scan_time: float = 0.0
    
    def _is_cache_valid(self) -> bool:
        """Check if cached detection result is still valid."""
        if self._cached_result is None:
            return False
        return (time.time() - self._last_scan_time) < self.cache_ttl
    
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
    
    def scan_available_lsps(self, force_refresh: bool = False) -> LSPDetectionResult:
        """
        Scan PATH for available LSP servers.
        
        Args:
            force_refresh: Force rescan even if cache is valid
            
        Returns:
            LSPDetectionResult containing detected LSP servers
        """
        if not force_refresh and self._is_cache_valid():
            logger.debug("Using cached LSP detection result")
            return self._cached_result
        
        logger.info("Scanning for available LSP servers...")
        start_time = time.time()
        
        result = LSPDetectionResult()
        
        for lsp_name, lsp_info in self.LSP_EXTENSION_MAP.items():
            # Try all alternative names for this LSP
            for binary_name in lsp_info['alternative_names']:
                binary_path = self._check_binary_exists(binary_name)
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
                    break  # Found this LSP, no need to check other names
        
        result.scan_duration = time.time() - start_time
        
        # Cache the result
        self._cached_result = result
        self._last_scan_time = time.time()
        
        logger.info(f"LSP scan completed in {result.scan_duration:.2f}s, found {len(result.detected_lsps)} LSPs")
        return result
    
    def get_supported_extensions(self, force_refresh: bool = False) -> List[str]:
        """
        Get list of file extensions supported by detected LSP servers.
        
        Args:
            force_refresh: Force rescan of LSP servers
            
        Returns:
            List of file extensions (including the dot)
        """
        detection_result = self.scan_available_lsps(force_refresh)
        
        # Collect all extensions from detected LSPs
        extensions = set()
        for lsp_info in detection_result.detected_lsps.values():
            extensions.update(lsp_info.supported_extensions)
        
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
    
    def get_detection_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current LSP detection state.
        
        Returns:
            Dictionary with detection summary information
        """
        detection_result = self.scan_available_lsps()
        
        return {
            'total_lsps_detected': len(detection_result.detected_lsps),
            'total_extensions_supported': len(self.get_supported_extensions()),
            'scan_duration': detection_result.scan_duration,
            'cache_age': time.time() - self._last_scan_time,
            'detected_lsps': {
                name: {
                    'binary_path': info.binary_path,
                    'version': info.version,
                    'extensions': info.supported_extensions,
                    'priority': info.priority
                }
                for name, info in detection_result.detected_lsps.items()
            }
        }


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