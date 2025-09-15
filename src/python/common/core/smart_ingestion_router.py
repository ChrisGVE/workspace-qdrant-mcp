"""
Smart Ingestion Differentiation Logic for Workspace Qdrant MCP

This module implements intelligent file processing system that differentiates between
code files requiring LSP enrichment and regular files for standard ingestion.

Key Features:
    - Intelligent file type classification using extensions, MIME types, and content analysis
    - Smart routing between LSP-enriched and standard ingestion pipelines
    - Robust fallback strategy when LSP servers are unavailable
    - Batch processing optimization for similar file types
    - Configurable file type mappings and processing overrides
    - Comprehensive statistics and decision logging for debugging
    - Performance monitoring and resource management

Example:
    ```python
    from workspace_qdrant_mcp.core.smart_ingestion_router import SmartIngestionRouter
    
    # Initialize router
    router = SmartIngestionRouter()
    await router.initialize()
    
    # Route single file
    strategy, metadata = await router.route_file("/path/to/file.py")
    
    # Process batch of files
    results = await router.process_batch(["/path/to/files"])
    ```
"""

import asyncio
import json
from loguru import logger
import mimetypes
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from .error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from .language_filters import LanguageAwareFilter, FilterStatistics
from .lsp_metadata_extractor import LspMetadataExtractor, FileMetadata
from .pattern_manager import PatternManager

# logger imported from loguru


class ProcessingStrategy(Enum):
    """Processing strategy for different file types"""
    LSP_ENRICHED = "lsp_enriched"        # Code files with LSP metadata extraction
    STANDARD_INGESTION = "standard"      # Non-code files with standard processing
    FALLBACK_CODE = "fallback_code"      # Code files processed as standard when LSP unavailable
    SKIP = "skip"                        # Files to skip processing
    BINARY = "binary"                    # Binary files requiring special handling


class FileClassification(Enum):
    """File classification categories"""
    CODE = "code"                        # Programming language source files
    DOCUMENTATION = "documentation"     # Documentation files (markdown, rst, etc.)
    DATA = "data"                       # Data files (json, yaml, csv, etc.)
    BINARY = "binary"                   # Binary files (images, executables, etc.)
    CONFIGURATION = "configuration"     # Configuration files
    TEXT = "text"                       # Plain text files
    UNKNOWN = "unknown"                 # Unidentified file types


@dataclass
class ClassificationResult:
    """Result of file classification analysis"""
    classification: FileClassification
    confidence: float  # 0.0 to 1.0
    strategy: ProcessingStrategy
    reason: str
    detected_language: Optional[str] = None
    mime_type: Optional[str] = None
    file_size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "confidence": self.confidence,
            "strategy": self.strategy.value,
            "reason": self.reason,
            "detected_language": self.detected_language,
            "mime_type": self.mime_type,
            "file_size_mb": self.file_size_mb
        }


@dataclass
class RouterConfiguration:
    """Configuration for smart ingestion router"""
    
    # File classification settings
    enable_content_analysis: bool = True
    enable_mime_detection: bool = True
    max_content_sample_bytes: int = 4096
    
    # Processing strategy overrides
    force_lsp_extensions: Set[str] = field(default_factory=lambda: {
        '.py', '.pyi', '.rs', '.js', '.jsx', '.ts', '.tsx', '.java', 
        '.go', '.c', '.h', '.cpp', '.cxx', '.hpp', '.hxx', '.cc', '.hh'
    })
    force_standard_extensions: Set[str] = field(default_factory=lambda: {
        '.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.xml', '.csv'
    })
    skip_extensions: Set[str] = field(default_factory=lambda: {
        '.pyc', '.pyo', '.class', '.o', '.so', '.dll', '.exe', '.bin'
    })
    
    # Custom language mappings
    custom_language_map: Dict[str, str] = field(default_factory=dict)
    
    # Fallback behavior
    fallback_on_lsp_error: bool = True
    fallback_timeout_seconds: float = 30.0
    
    # Batch processing settings
    batch_size_limit: int = 100
    max_concurrent_batches: int = 3
    batch_timeout_seconds: float = 300.0
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: float = 3600.0
    max_cache_entries: int = 10000
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "RouterConfiguration":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning("Router config file not found, using defaults", path=str(config_path))
            return cls()
        
        try:
            with config_path.open() as f:
                data = yaml.safe_load(f)
            
            router_config = data.get('smart_router', {})
            
            config = cls()
            
            # Update basic settings
            config.enable_content_analysis = router_config.get('enable_content_analysis', True)
            config.enable_mime_detection = router_config.get('enable_mime_detection', True)
            config.max_content_sample_bytes = router_config.get('max_content_sample_bytes', 4096)
            
            # Update extension overrides
            if 'force_lsp_extensions' in router_config:
                config.force_lsp_extensions = set(router_config['force_lsp_extensions'])
            if 'force_standard_extensions' in router_config:
                config.force_standard_extensions = set(router_config['force_standard_extensions'])
            if 'skip_extensions' in router_config:
                config.skip_extensions = set(router_config['skip_extensions'])
            
            # Update custom mappings
            config.custom_language_map = router_config.get('custom_language_map', {})
            
            # Update fallback settings
            config.fallback_on_lsp_error = router_config.get('fallback_on_lsp_error', True)
            config.fallback_timeout_seconds = router_config.get('fallback_timeout_seconds', 30.0)
            
            # Update batch settings
            config.batch_size_limit = router_config.get('batch_size_limit', 100)
            config.max_concurrent_batches = router_config.get('max_concurrent_batches', 3)
            config.batch_timeout_seconds = router_config.get('batch_timeout_seconds', 300.0)
            
            # Update performance settings
            config.enable_caching = router_config.get('enable_caching', True)
            config.cache_ttl_seconds = router_config.get('cache_ttl_seconds', 3600.0)
            config.max_cache_entries = router_config.get('max_cache_entries', 10000)
            
            logger.info("Router configuration loaded successfully", config_path=str(config_path))
            return config
            
        except Exception as e:
            logger.error("Failed to load router configuration", 
                        config_path=str(config_path), error=str(e))
            return cls()


@dataclass
class RouterStatistics:
    """Statistics for router operations"""
    
    # Classification stats
    files_classified: int = 0
    classification_time_ms: float = 0.0
    classification_by_type: Dict[str, int] = field(default_factory=dict)
    strategy_decisions: Dict[str, int] = field(default_factory=dict)
    
    # Processing stats
    files_processed: int = 0
    files_failed: int = 0
    lsp_processed: int = 0
    standard_processed: int = 0
    fallback_processed: int = 0
    skipped_files: int = 0
    
    # Performance stats
    total_processing_time_ms: float = 0.0
    batch_processing_time_ms: float = 0.0
    lsp_processing_time_ms: float = 0.0
    standard_processing_time_ms: float = 0.0
    
    # Cache stats
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    
    # Error stats
    lsp_errors: int = 0
    classification_errors: int = 0
    fallback_triggers: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": {
                "files_classified": self.files_classified,
                "classification_time_ms": self.classification_time_ms,
                "avg_classification_time_ms": self.classification_time_ms / max(1, self.files_classified),
                "classification_by_type": dict(self.classification_by_type),
                "strategy_decisions": dict(self.strategy_decisions)
            },
            "processing": {
                "files_processed": self.files_processed,
                "files_failed": self.files_failed,
                "success_rate": self.files_processed / max(1, self.files_processed + self.files_failed),
                "lsp_processed": self.lsp_processed,
                "standard_processed": self.standard_processed,
                "fallback_processed": self.fallback_processed,
                "skipped_files": self.skipped_files
            },
            "performance": {
                "total_processing_time_ms": self.total_processing_time_ms,
                "avg_processing_time_ms": self.total_processing_time_ms / max(1, self.files_processed),
                "batch_processing_time_ms": self.batch_processing_time_ms,
                "lsp_processing_time_ms": self.lsp_processing_time_ms,
                "standard_processing_time_ms": self.standard_processing_time_ms
            },
            "cache": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                "cache_evictions": self.cache_evictions
            },
            "errors": {
                "lsp_errors": self.lsp_errors,
                "classification_errors": self.classification_errors,
                "fallback_triggers": self.fallback_triggers
            }
        }


class FileClassifier:
    """
    Intelligent file classification system using multiple detection methods.
    
    Combines file extension analysis, MIME type detection, and content analysis
    to accurately classify files and determine optimal processing strategy.
    """
    
    def __init__(self, config: RouterConfiguration, pattern_manager: Optional[PatternManager] = None):
        self.config = config
        self.mime_types = mimetypes.MimeTypes()
        self.pattern_manager = pattern_manager or PatternManager()

        # Initialize known extensions for different categories
        # These are kept as fallbacks, but PatternManager takes precedence
        self.code_extensions = {
            '.py', '.pyi', '.rs', '.js', '.jsx', '.ts', '.tsx', '.java',
            '.go', '.c', '.h', '.cpp', '.cxx', '.hpp', '.hxx', '.cc', '.hh',
            '.cs', '.php', '.rb', '.swift', '.kt', '.scala', '.clj', '.hs',
            '.ml', '.fs', '.vb', '.pas', '.pl', '.r', '.m', '.sh', '.ps1',
            '.lua', '.dart', '.jl', '.nim', '.crystal', '.zig'
        }

        self.documentation_extensions = {
            '.md', '.rst', '.txt', '.adoc', '.tex', '.org'
        }

        self.data_extensions = {
            '.json', '.yaml', '.yml', '.xml', '.csv', '.tsv', '.toml', '.ini'
        }

        self.config_extensions = {
            '.conf', '.cfg', '.config', '.properties', '.env'
        }

        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.a', '.lib', '.bin', '.out',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
            '.mp3', '.wav', '.mp4', '.avi', '.mov', '.pdf', '.zip', '.tar',
            '.gz', '.rar', '.7z'
        }
        
        # Content-based detection patterns
        self.code_patterns = [
            # Function definitions
            re.compile(r'^\s*(def|function|func|fn|public|private|protected|static)\s+\w+', re.MULTILINE),
            # Class definitions  
            re.compile(r'^\s*(class|interface|struct|enum|trait)\s+\w+', re.MULTILINE),
            # Import/include statements
            re.compile(r'^\s*(import|from|#include|use|require)\s+', re.MULTILINE),
            # Variable declarations with types
            re.compile(r'^\s*(var|let|const|int|string|bool|float|double)\s+\w+', re.MULTILINE),
            # Common programming constructs
            re.compile(r'[{}();]|if\s*\(|for\s*\(|while\s*\(', re.MULTILINE)
        ]
        
        logger.info("File classifier initialized", 
                   code_extensions_count=len(self.code_extensions),
                   total_patterns=len(self.code_patterns))
    
    def classify_file(self, file_path: Path) -> ClassificationResult:
        """
        Classify a file and determine processing strategy.
        
        Args:
            file_path: Path to file to classify
            
        Returns:
            ClassificationResult with classification and processing strategy
        """
        start_time = time.perf_counter()
        
        try:
            if not file_path.exists():
                return ClassificationResult(
                    classification=FileClassification.UNKNOWN,
                    confidence=0.0,
                    strategy=ProcessingStrategy.SKIP,
                    reason="file_not_found"
                )
            
            if not file_path.is_file():
                return ClassificationResult(
                    classification=FileClassification.UNKNOWN,
                    confidence=0.0,
                    strategy=ProcessingStrategy.SKIP,
                    reason="not_a_file"
                )
            
            # Get basic file info
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            extension = file_path.suffix.lower()
            
            # Check configuration overrides first
            if extension in self.config.skip_extensions:
                return ClassificationResult(
                    classification=FileClassification.BINARY,
                    confidence=1.0,
                    strategy=ProcessingStrategy.SKIP,
                    reason="extension_skip_override",
                    file_size_mb=file_size_mb
                )
            
            if extension in self.config.force_lsp_extensions:
                return ClassificationResult(
                    classification=FileClassification.CODE,
                    confidence=1.0,
                    strategy=ProcessingStrategy.LSP_ENRICHED,
                    reason="extension_lsp_override",
                    detected_language=self._detect_language(file_path),
                    file_size_mb=file_size_mb
                )
            
            if extension in self.config.force_standard_extensions:
                return ClassificationResult(
                    classification=FileClassification.DOCUMENTATION,
                    confidence=1.0,
                    strategy=ProcessingStrategy.STANDARD_INGESTION,
                    reason="extension_standard_override",
                    file_size_mb=file_size_mb
                )
            
            # Extension-based classification
            classification, confidence = self._classify_by_extension(extension)
            
            # MIME type detection if enabled
            mime_type = None
            if self.config.enable_mime_detection:
                mime_type = self._detect_mime_type(file_path)
                if mime_type:
                    mime_classification, mime_confidence = self._classify_by_mime_type(mime_type)
                    if mime_confidence > confidence:
                        classification = mime_classification
                        confidence = mime_confidence
            
            # Content analysis if enabled and confidence is low
            if self.config.enable_content_analysis and confidence < 0.8:
                content_classification, content_confidence = self._classify_by_content(file_path)
                if content_confidence > confidence:
                    classification = content_classification
                    confidence = content_confidence
            
            # Determine processing strategy
            strategy = self._determine_strategy(classification, file_path)
            
            # Detect language for code files
            detected_language = None
            if classification == FileClassification.CODE:
                detected_language = self._detect_language(file_path)
            
            result = ClassificationResult(
                classification=classification,
                confidence=confidence,
                strategy=strategy,
                reason=f"multi_method_analysis_{classification.value}",
                detected_language=detected_language,
                mime_type=mime_type,
                file_size_mb=file_size_mb
            )
            
            classification_time = (time.perf_counter() - start_time) * 1000
            logger.debug("File classified",
                        file_path=str(file_path),
                        classification=classification.value,
                        strategy=strategy.value,
                        confidence=confidence,
                        language=detected_language,
                        time_ms=classification_time)
            
            return result
            
        except Exception as e:
            logger.error("File classification failed",
                        file_path=str(file_path),
                        error=str(e))
            
            return ClassificationResult(
                classification=FileClassification.UNKNOWN,
                confidence=0.0,
                strategy=ProcessingStrategy.SKIP,
                reason=f"classification_error: {e}",
                file_size_mb=file_size_mb if 'file_size_mb' in locals() else 0.0
            )
    
    def _classify_by_extension(self, extension: str) -> Tuple[FileClassification, float]:
        """Classify file by extension with confidence score"""
        if extension in self.code_extensions:
            return FileClassification.CODE, 0.9
        elif extension in self.documentation_extensions:
            return FileClassification.DOCUMENTATION, 0.9
        elif extension in self.data_extensions:
            return FileClassification.DATA, 0.9
        elif extension in self.config_extensions:
            return FileClassification.CONFIGURATION, 0.9
        elif extension in self.binary_extensions:
            return FileClassification.BINARY, 0.9
        elif extension == '.txt':
            return FileClassification.TEXT, 0.7
        else:
            return FileClassification.UNKNOWN, 0.1
    
    def _detect_mime_type(self, file_path: Path) -> Optional[str]:
        """Detect MIME type using python mimetypes"""
        try:
            mime_type, _ = self.mime_types.guess_type(str(file_path))
            return mime_type
        except Exception as e:
            logger.debug("MIME type detection failed", file_path=str(file_path), error=str(e))
            return None
    
    def _classify_by_mime_type(self, mime_type: str) -> Tuple[FileClassification, float]:
        """Classify file by MIME type"""
        if mime_type.startswith('text/'):
            if 'html' in mime_type or 'xml' in mime_type:
                return FileClassification.DATA, 0.8
            else:
                return FileClassification.TEXT, 0.7
        elif mime_type.startswith('application/'):
            if any(code_type in mime_type for code_type in ['javascript', 'json', 'xml']):
                if 'javascript' in mime_type:
                    return FileClassification.CODE, 0.8
                else:
                    return FileClassification.DATA, 0.8
            else:
                return FileClassification.BINARY, 0.8
        elif mime_type.startswith(('image/', 'audio/', 'video/')):
            return FileClassification.BINARY, 0.9
        else:
            return FileClassification.UNKNOWN, 0.3
    
    def _classify_by_content(self, file_path: Path) -> Tuple[FileClassification, float]:
        """Classify file by analyzing content"""
        try:
            # Read sample of file content
            with open(file_path, 'rb') as f:
                raw_content = f.read(self.config.max_content_sample_bytes)
            
            # Try to decode as text
            try:
                content = raw_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = raw_content.decode('latin1')
                except UnicodeDecodeError:
                    # Binary file
                    return FileClassification.BINARY, 0.9
            
            # Check for binary content indicators
            if '\0' in content or len([b for b in raw_content if b < 32 and b not in [9, 10, 13]]) > len(raw_content) * 0.1:
                return FileClassification.BINARY, 0.9
            
            # Count matches for code patterns
            code_matches = 0
            for pattern in self.code_patterns:
                if pattern.search(content):
                    code_matches += 1
            
            # Analyze content characteristics
            lines = content.split('\n')
            if not lines:
                return FileClassification.TEXT, 0.1
            
            # Code-like characteristics
            has_indentation = any(line.startswith((' ', '\t')) for line in lines)
            has_braces = '{' in content and '}' in content
            has_semicolons = content.count(';') > len(lines) * 0.1
            has_keywords = any(keyword in content for keyword in [
                'function', 'def', 'class', 'import', 'return', 'if', 'for', 'while'
            ])
            
            # Documentation-like characteristics
            has_markdown = any(line.startswith('#') for line in lines)
            has_long_paragraphs = any(len(line) > 80 and '(' not in line for line in lines)
            
            # Calculate confidence based on features
            if code_matches >= 2 or (has_braces and has_semicolons and has_keywords):
                return FileClassification.CODE, 0.8
            elif code_matches >= 1 or (has_indentation and has_keywords):
                return FileClassification.CODE, 0.6
            elif has_markdown or has_long_paragraphs:
                return FileClassification.DOCUMENTATION, 0.7
            elif content.strip().startswith(('{', '[')):
                return FileClassification.DATA, 0.8
            else:
                return FileClassification.TEXT, 0.5
            
        except Exception as e:
            logger.debug("Content analysis failed", file_path=str(file_path), error=str(e))
            return FileClassification.UNKNOWN, 0.1
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file path and extension using PatternManager"""
        extension = file_path.suffix.lower()

        # Check custom mappings first
        if extension in self.config.custom_language_map:
            return self.config.custom_language_map[extension]

        # Use PatternManager for language detection
        language_info = self.pattern_manager.get_language_info(file_path)
        if language_info:
            return language_info['language']

        # Fallback to hardcoded mappings for compatibility
        language_map = {
            '.py': 'python', '.pyi': 'python',
            '.rs': 'rust',
            '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.cxx': 'cpp', '.cc': 'cpp',
            '.hpp': 'cpp', '.hxx': 'cpp', '.hh': 'cpp',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.clj': 'clojure',
            '.hs': 'haskell',
            '.ml': 'ocaml',
            '.fs': 'fsharp',
            '.vb': 'vb',
            '.pas': 'pascal',
            '.pl': 'perl',
            '.r': 'r',
            '.m': 'objective-c',
            '.sh': 'bash',
            '.ps1': 'powershell',
            '.lua': 'lua',
            '.dart': 'dart',
            '.jl': 'julia'
        }

        return language_map.get(extension)
    
    def _determine_strategy(self, classification: FileClassification, file_path: Path) -> ProcessingStrategy:
        """Determine processing strategy based on classification"""
        if classification == FileClassification.CODE:
            return ProcessingStrategy.LSP_ENRICHED
        elif classification == FileClassification.BINARY:
            return ProcessingStrategy.SKIP
        elif classification in [
            FileClassification.DOCUMENTATION,
            FileClassification.DATA,
            FileClassification.TEXT,
            FileClassification.CONFIGURATION
        ]:
            return ProcessingStrategy.STANDARD_INGESTION
        else:
            return ProcessingStrategy.SKIP


class SmartIngestionRouter:
    """
    Smart ingestion router that intelligently routes files between LSP-enriched 
    and standard ingestion based on file type classification and system capabilities.
    
    This is the main orchestration component that coordinates:
    - File classification using FileClassifier
    - LSP-based code metadata extraction
    - Standard document ingestion
    - Fallback processing when LSP is unavailable
    - Batch processing optimization
    - Performance monitoring and statistics
    """
    
    def __init__(
        self,
        config: Optional[RouterConfiguration] = None,
        file_filter: Optional[LanguageAwareFilter] = None,
        lsp_extractor: Optional[LspMetadataExtractor] = None,
        pattern_manager: Optional[PatternManager] = None
    ):
        """
        Initialize the smart ingestion router.

        Args:
            config: Router configuration (uses defaults if None)
            file_filter: File filtering system (created if None)
            lsp_extractor: LSP metadata extractor (created if None)
            pattern_manager: Pattern management system (created if None)
        """
        self.config = config or RouterConfiguration()
        self.pattern_manager = pattern_manager or PatternManager()
        self.file_filter = file_filter or LanguageAwareFilter(pattern_manager=self.pattern_manager)
        self.lsp_extractor = lsp_extractor

        # Initialize components
        self.classifier = FileClassifier(self.config, self.pattern_manager)
        self.statistics = RouterStatistics()
        
        # Caching system
        self.classification_cache: Dict[str, Tuple[ClassificationResult, float]] = {}
        
        # Processing state
        self._processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        self._initialized = False
        
        logger.info("Smart ingestion router initialized", 
                   config_summary=self._get_config_summary())
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            "content_analysis_enabled": self.config.enable_content_analysis,
            "mime_detection_enabled": self.config.enable_mime_detection,
            "force_lsp_extensions_count": len(self.config.force_lsp_extensions),
            "force_standard_extensions_count": len(self.config.force_standard_extensions),
            "skip_extensions_count": len(self.config.skip_extensions),
            "batch_size_limit": self.config.batch_size_limit,
            "caching_enabled": self.config.enable_caching
        }
    
    async def initialize(self, workspace_root: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the router and its dependencies.
        
        Args:
            workspace_root: Root directory of the workspace
        """
        if self._initialized:
            return
        
        logger.info("Initializing smart ingestion router", workspace_root=str(workspace_root) if workspace_root else None)
        
        # Initialize file filter
        if not self.file_filter._initialized:
            await self.file_filter.load_configuration()
        
        # Initialize LSP extractor if available
        if self.lsp_extractor and not self.lsp_extractor._initialized:
            await self.lsp_extractor.initialize(workspace_root)
        elif not self.lsp_extractor:
            # Create LSP extractor if none provided
            try:
                self.lsp_extractor = LspMetadataExtractor(file_filter=self.file_filter)
                await self.lsp_extractor.initialize(workspace_root)
                logger.info("LSP extractor created and initialized")
            except Exception as e:
                logger.warning("Failed to initialize LSP extractor", error=str(e))
                self.lsp_extractor = None
        
        self._initialized = True
        logger.info("Smart ingestion router initialization completed")
    
    async def route_file(
        self, 
        file_path: Union[str, Path],
        force_refresh: bool = False
    ) -> Tuple[ProcessingStrategy, Optional[ClassificationResult]]:
        """
        Route a single file to appropriate processing pipeline.
        
        Args:
            file_path: Path to file to route
            force_refresh: Bypass classification cache
            
        Returns:
            Tuple of (processing_strategy, classification_result)
        """
        if not self._initialized:
            await self.initialize()
        
        file_path = Path(file_path)
        start_time = time.perf_counter()
        
        try:
            # Check file filter first
            should_process, filter_reason = self.file_filter.should_process_file(file_path)
            if not should_process:
                logger.debug("File filtered out by file filter",
                           file_path=str(file_path), reason=filter_reason)
                return ProcessingStrategy.SKIP, None
            
            # Get classification (with caching)
            classification_result = await self._get_file_classification(file_path, force_refresh)
            
            # Update statistics
            self.statistics.files_classified += 1
            classification_time = (time.perf_counter() - start_time) * 1000
            self.statistics.classification_time_ms += classification_time
            
            # Update classification stats
            classification_type = classification_result.classification.value
            self.statistics.classification_by_type[classification_type] = (
                self.statistics.classification_by_type.get(classification_type, 0) + 1
            )
            
            strategy = classification_result.strategy.value
            self.statistics.strategy_decisions[strategy] = (
                self.statistics.strategy_decisions.get(strategy, 0) + 1
            )
            
            logger.debug("File routed",
                        file_path=str(file_path),
                        classification=classification_result.classification.value,
                        strategy=classification_result.strategy.value,
                        confidence=classification_result.confidence,
                        language=classification_result.detected_language)
            
            return classification_result.strategy, classification_result
            
        except Exception as e:
            self.statistics.classification_errors += 1
            logger.error("File routing failed", file_path=str(file_path), error=str(e))
            return ProcessingStrategy.SKIP, None
    
    async def _get_file_classification(
        self, 
        file_path: Path, 
        force_refresh: bool = False
    ) -> ClassificationResult:
        """Get file classification with caching support"""
        file_key = str(file_path.resolve())
        current_time = time.time()
        
        # Check cache if enabled
        if self.config.enable_caching and not force_refresh and file_key in self.classification_cache:
            cached_result, cache_time = self.classification_cache[file_key]
            if current_time - cache_time < self.config.cache_ttl_seconds:
                self.statistics.cache_hits += 1
                return cached_result
            else:
                # Cache expired
                del self.classification_cache[file_key]
        
        self.statistics.cache_misses += 1
        
        # Classify file
        classification_result = self.classifier.classify_file(file_path)
        
        # Cache result if enabled
        if self.config.enable_caching:
            self._cache_classification(file_key, classification_result, current_time)
        
        return classification_result
    
    def _cache_classification(self, file_key: str, result: ClassificationResult, timestamp: float) -> None:
        """Cache classification result with size limits"""
        # Limit cache size
        if len(self.classification_cache) >= self.config.max_cache_entries:
            # Remove oldest entries
            oldest_entries = sorted(
                self.classification_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )[:100]  # Remove 100 oldest
            
            for key, _ in oldest_entries:
                del self.classification_cache[key]
                self.statistics.cache_evictions += 1
        
        self.classification_cache[file_key] = (result, timestamp)
    
    async def process_single_file(
        self, 
        file_path: Union[str, Path],
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single file through the appropriate pipeline.
        
        Args:
            file_path: Path to file to process
            force_refresh: Force refresh of cached data
            
        Returns:
            Processing result dictionary or None if processing failed/skipped
        """
        if not self._initialized:
            await self.initialize()
        
        file_path = Path(file_path)
        start_time = time.perf_counter()
        
        try:
            # Route file to determine processing strategy
            strategy, classification = await self.route_file(file_path, force_refresh)
            
            if strategy == ProcessingStrategy.SKIP:
                self.statistics.skipped_files += 1
                return None
            
            # Process based on strategy
            if strategy == ProcessingStrategy.LSP_ENRICHED:
                result = await self._process_lsp_enriched(file_path, classification)
                if result:
                    self.statistics.lsp_processed += 1
                    processing_time = (time.perf_counter() - start_time) * 1000
                    self.statistics.lsp_processing_time_ms += processing_time
                else:
                    # LSP processing failed, try fallback if enabled
                    if self.config.fallback_on_lsp_error:
                        self.statistics.fallback_triggers += 1
                        result = await self._process_fallback_code(file_path, classification)
                        if result:
                            self.statistics.fallback_processed += 1
            
            elif strategy == ProcessingStrategy.STANDARD_INGESTION:
                result = await self._process_standard_ingestion(file_path, classification)
                if result:
                    self.statistics.standard_processed += 1
                    processing_time = (time.perf_counter() - start_time) * 1000
                    self.statistics.standard_processing_time_ms += processing_time
            
            elif strategy == ProcessingStrategy.FALLBACK_CODE:
                result = await self._process_fallback_code(file_path, classification)
                if result:
                    self.statistics.fallback_processed += 1
            
            else:  # ProcessingStrategy.BINARY or unknown
                self.statistics.skipped_files += 1
                return None
            
            # Update statistics
            if result:
                self.statistics.files_processed += 1
                total_time = (time.perf_counter() - start_time) * 1000
                self.statistics.total_processing_time_ms += total_time
                
                logger.debug("File processed successfully",
                           file_path=str(file_path),
                           strategy=strategy.value,
                           processing_time_ms=total_time)
            else:
                self.statistics.files_failed += 1
                logger.warning("File processing failed", 
                             file_path=str(file_path), strategy=strategy.value)
            
            return result
            
        except Exception as e:
            self.statistics.files_failed += 1
            processing_time = (time.perf_counter() - start_time) * 1000
            self.statistics.total_processing_time_ms += processing_time
            
            logger.error("File processing error",
                        file_path=str(file_path), error=str(e))
            return None
    
    async def _process_lsp_enriched(
        self, 
        file_path: Path, 
        classification: ClassificationResult
    ) -> Optional[Dict[str, Any]]:
        """Process code file with LSP enrichment"""
        if not self.lsp_extractor:
            logger.warning("LSP extractor not available", file_path=str(file_path))
            return None
        
        try:
            # Extract LSP metadata
            metadata = await self.lsp_extractor.extract_file_metadata(file_path)
            
            if metadata:
                # Combine LSP metadata with classification info
                result = {
                    "processing_strategy": "lsp_enriched",
                    "file_path": str(file_path),
                    "classification": classification.to_dict(),
                    "lsp_metadata": metadata.to_dict(),
                    "processing_timestamp": time.time()
                }
                
                logger.debug("LSP enriched processing completed",
                           file_path=str(file_path),
                           symbols_count=len(metadata.symbols),
                           relationships_count=len(metadata.relationships))
                
                return result
            else:
                logger.warning("LSP metadata extraction returned None", file_path=str(file_path))
                return None
                
        except Exception as e:
            self.statistics.lsp_errors += 1
            logger.error("LSP enriched processing failed",
                        file_path=str(file_path), error=str(e))
            return None
    
    async def _process_standard_ingestion(
        self, 
        file_path: Path, 
        classification: ClassificationResult
    ) -> Optional[Dict[str, Any]]:
        """Process non-code file with standard ingestion"""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Create standard processing result
            result = {
                "processing_strategy": "standard_ingestion",
                "file_path": str(file_path),
                "classification": classification.to_dict(),
                "content": content,
                "file_size": len(content),
                "line_count": content.count('\n') + 1 if content else 0,
                "processing_timestamp": time.time(),
                "metadata": {
                    "detected_language": classification.detected_language,
                    "mime_type": classification.mime_type,
                    "file_extension": file_path.suffix
                }
            }
            
            logger.debug("Standard ingestion completed",
                        file_path=str(file_path),
                        content_size=len(content),
                        line_count=result["line_count"])
            
            return result
            
        except Exception as e:
            logger.error("Standard ingestion failed",
                        file_path=str(file_path), error=str(e))
            return None
    
    async def _process_fallback_code(
        self, 
        file_path: Path, 
        classification: ClassificationResult
    ) -> Optional[Dict[str, Any]]:
        """Process code file as standard text when LSP is unavailable"""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Basic code analysis without LSP
            lines = content.splitlines()
            
            # Simple syntax highlighting metadata
            syntax_info = {
                "total_lines": len(lines),
                "non_empty_lines": len([line for line in lines if line.strip()]),
                "comment_lines": len([line for line in lines if line.strip().startswith(('#', '//', '/*'))]),
                "estimated_complexity": min(10, len([line for line in lines if any(kw in line for kw in ['if', 'for', 'while', 'def', 'class', 'function'])]))
            }
            
            result = {
                "processing_strategy": "fallback_code",
                "file_path": str(file_path),
                "classification": classification.to_dict(),
                "content": content,
                "file_size": len(content),
                "line_count": len(lines),
                "processing_timestamp": time.time(),
                "syntax_info": syntax_info,
                "metadata": {
                    "detected_language": classification.detected_language,
                    "mime_type": classification.mime_type,
                    "file_extension": file_path.suffix,
                    "fallback_reason": "lsp_unavailable"
                }
            }
            
            logger.debug("Fallback code processing completed",
                        file_path=str(file_path),
                        language=classification.detected_language,
                        complexity=syntax_info["estimated_complexity"])
            
            return result
            
        except Exception as e:
            logger.error("Fallback code processing failed",
                        file_path=str(file_path), error=str(e))
            return None
    
    async def process_batch(
        self, 
        file_paths: List[Union[str, Path]],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files in optimized batches.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Override default batch size
            
        Returns:
            List of processing results (excludes None/failed results)
        """
        if not self._initialized:
            await self.initialize()
        
        batch_size = batch_size or self.config.batch_size_limit
        start_time = time.perf_counter()
        
        logger.info("Starting batch processing",
                   total_files=len(file_paths),
                   batch_size=batch_size,
                   max_concurrent_batches=self.config.max_concurrent_batches)
        
        # Group files by processing strategy for optimization
        strategy_groups = await self._group_files_by_strategy(file_paths)
        
        # Process each strategy group
        all_results = []
        
        for strategy, files in strategy_groups.items():
            if not files:
                continue
            
            logger.info(f"Processing {strategy.value} batch",
                       files_count=len(files))
            
            # Process files in batches
            strategy_results = await self._process_strategy_batch(strategy, files, batch_size)
            all_results.extend(strategy_results)
        
        # Update batch processing statistics
        batch_time = (time.perf_counter() - start_time) * 1000
        self.statistics.batch_processing_time_ms += batch_time
        
        logger.info("Batch processing completed",
                   total_files=len(file_paths),
                   successful_results=len(all_results),
                   processing_time_ms=batch_time)
        
        return all_results
    
    async def _group_files_by_strategy(
        self, 
        file_paths: List[Union[str, Path]]
    ) -> Dict[ProcessingStrategy, List[Path]]:
        """Group files by their processing strategy for batch optimization"""
        strategy_groups: Dict[ProcessingStrategy, List[Path]] = {
            strategy: [] for strategy in ProcessingStrategy
        }
        
        # Classify all files first
        for file_path in file_paths:
            strategy, _ = await self.route_file(file_path)
            if strategy != ProcessingStrategy.SKIP:
                strategy_groups[strategy].append(Path(file_path))
        
        return strategy_groups
    
    async def _process_strategy_batch(
        self,
        strategy: ProcessingStrategy,
        files: List[Path],
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Process files of the same strategy in optimized batches"""
        all_results = []
        
        # Split into batches
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            async with self._processing_semaphore:
                # Process batch concurrently
                batch_tasks = [self.process_single_file(file_path) for file_path in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter successful results
                successful_results = []
                for result in batch_results:
                    if isinstance(result, dict):
                        successful_results.append(result)
                    elif isinstance(result, Exception):
                        logger.error("Batch processing exception", error=str(result))
                
                all_results.extend(successful_results)
                
                logger.debug(f"Processed {strategy.value} batch",
                           batch_size=len(batch),
                           successful_results=len(successful_results))
        
        return all_results
    
    def get_statistics(self) -> RouterStatistics:
        """Get current router statistics"""
        return self.statistics
    
    def reset_statistics(self) -> None:
        """Reset router statistics"""
        self.statistics = RouterStatistics()
        logger.info("Router statistics reset")
    
    def clear_classification_cache(self) -> None:
        """Clear classification cache"""
        cache_size = len(self.classification_cache)
        self.classification_cache.clear()
        logger.info("Classification cache cleared", entries_cleared=cache_size)
    
    async def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get information about current processing capabilities"""
        lsp_available = self.lsp_extractor is not None and self.lsp_extractor._initialized
        lsp_languages = []
        
        if lsp_available:
            lsp_languages = list(self.lsp_extractor.lsp_server_configs.keys())
        
        return {
            "lsp_available": lsp_available,
            "supported_lsp_languages": lsp_languages,
            "file_filter_initialized": self.file_filter._initialized,
            "classification_cache_size": len(self.classification_cache),
            "configuration": self._get_config_summary(),
            "statistics": self.statistics.to_dict()
        }
    
    async def shutdown(self) -> None:
        """Shutdown router and clean up resources"""
        logger.info("Shutting down smart ingestion router")
        
        # Shutdown LSP extractor if available
        if self.lsp_extractor:
            await self.lsp_extractor.shutdown()
        
        # Clear caches
        self.classification_cache.clear()
        
        self._initialized = False
        logger.info("Smart ingestion router shutdown completed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()