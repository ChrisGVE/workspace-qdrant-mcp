"""
Enhanced Auto-Ingestion Pipeline with Format Detection and Validation.

This module provides a sophisticated auto-ingestion system with MIME type detection,
file integrity checking, format-specific processing pipelines, error recovery, and
comprehensive validation mechanisms.
"""

import asyncio
import hashlib
import json
import mimetypes
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import magic
from loguru import logger

# Import existing components
from .advanced_priority_queue import AdvancedPriorityQueue, PriorityTask, TaskPriority
from .incremental_file_updates import ChangeRecord, IncrementalUpdateSystem


class FileStatus(Enum):
    """Status of file processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    QUARANTINED = "quarantined"


class ValidationResult(Enum):
    """Result of file validation."""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    CORRUPTED = "corrupted"
    TOO_LARGE = "too_large"
    UNSUPPORTED_FORMAT = "unsupported_format"


class ProcessingStrategy(Enum):
    """Strategy for processing files."""
    IMMEDIATE = "immediate"
    BATCH = "batch"
    DEFERRED = "deferred"
    BACKGROUND = "background"


@dataclass
class FileMetadata:
    """Comprehensive file metadata."""
    file_path: str
    original_name: str
    size_bytes: int
    mime_type: str
    detected_format: str
    encoding: str | None = None
    created_at: float = field(default_factory=time.time)
    modified_at: float | None = None
    checksum_md5: str | None = None
    checksum_sha256: str | None = None
    magic_signature: str | None = None
    content_preview: str | None = None
    language_detected: str | None = None
    confidence_score: float = 0.0
    custom_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = self.__dict__.copy()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'FileMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ValidationReport:
    """Report from file validation process."""
    file_path: str
    validation_result: ValidationResult
    mime_type_matches: bool
    integrity_check_passed: bool
    format_validation_passed: bool
    security_scan_passed: bool
    issues_found: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    validation_time: float = field(default_factory=time.time)
    validator_version: str = "1.0.0"

    def is_safe_to_process(self) -> bool:
        """Determine if file is safe to process."""
        return (
            self.validation_result in [ValidationResult.VALID, ValidationResult.SUSPICIOUS] and
            self.security_scan_passed and
            self.integrity_check_passed
        )

    def should_quarantine(self) -> bool:
        """Determine if file should be quarantined."""
        return (
            self.validation_result == ValidationResult.CORRUPTED or
            not self.security_scan_passed or
            len(self.issues_found) > 3  # Too many issues
        )


@dataclass
class ProcessingResult:
    """Result from file processing."""
    file_path: str
    status: FileStatus
    processing_time: float
    metadata: FileMetadata | None = None
    validation_report: ValidationReport | None = None
    error_message: str | None = None
    extracted_content: str | None = None
    processing_strategy_used: ProcessingStrategy | None = None
    retry_count: int = 0
    max_retries: int = 3
    collection_assigned: str | None = None
    custom_data: dict[str, Any] = field(default_factory=dict)

    def can_retry(self) -> bool:
        """Check if processing can be retried."""
        return self.retry_count < self.max_retries and self.status == FileStatus.FAILED


class FormatDetector:
    """Advanced format detection and validation."""

    def __init__(self):
        """Initialize format detector."""
        # Initialize python-magic
        try:
            self.magic = magic.Magic(mime=True)
            self.magic_mime_encoding = magic.Magic(mime_encoding=True)
            self.magic_description = magic.Magic()
            self._magic_available = True
        except Exception as e:
            logger.warning(f"Magic library not available: {e}")
            self._magic_available = False

        # File type patterns
        self.format_patterns = {
            'text/plain': {
                'extensions': {'.txt', '.text', '.log', '.cfg', '.conf', '.ini'},
                'signatures': [b'', b'\xef\xbb\xbf'],  # Empty or UTF-8 BOM
                'max_binary_ratio': 0.1
            },
            'text/markdown': {
                'extensions': {'.md', '.markdown', '.mdown', '.mkd'},
                'signatures': [],
                'content_patterns': [r'^#\s+', r'^\*\s+', r'^\-\s+', r'\[.*\]\(.*\)']
            },
            'application/json': {
                'extensions': {'.json', '.jsn'},
                'signatures': [b'{', b'['],
                'validation': lambda content: self._validate_json(content)
            },
            'text/yaml': {
                'extensions': {'.yaml', '.yml'},
                'signatures': [b'---', b'%YAML'],
                'content_patterns': [r'^\s*\w+:', r'^\s*-\s+\w+']
            },
            'application/pdf': {
                'extensions': {'.pdf'},
                'signatures': [b'%PDF-'],
                'binary': True
            },
            'text/x-python': {
                'extensions': {'.py', '.pyx', '.pyi'},
                'content_patterns': [r'^#!/usr/bin/(env )?python', r'^\s*import\s+', r'^\s*from\s+\w+\s+import']
            },
            'text/javascript': {
                'extensions': {'.js', '.mjs', '.jsx'},
                'content_patterns': [r'^\s*function\s+', r'^\s*const\s+', r'^\s*let\s+', r'^\s*var\s+']
            }
        }

    async def detect_format(self, file_path: Path) -> FileMetadata:
        """Detect comprehensive file format information."""
        file_path_str = str(file_path)

        try:
            # Get basic file info
            stat_info = file_path.stat()
            size_bytes = stat_info.st_size
            modified_at = stat_info.st_mtime

            # Initialize metadata
            metadata = FileMetadata(
                file_path=file_path_str,
                original_name=file_path.name,
                size_bytes=size_bytes,
                mime_type="application/octet-stream",  # Default
                detected_format="unknown",
                modified_at=modified_at
            )

            # MIME type detection using multiple methods
            mime_type = self._detect_mime_type(file_path)
            metadata.mime_type = mime_type

            # Magic signature detection
            if self._magic_available:
                try:
                    metadata.magic_signature = self.magic_description.from_file(file_path_str)
                    metadata.encoding = self.magic_mime_encoding.from_file(file_path_str)
                except Exception as e:
                    logger.debug(f"Magic detection failed for {file_path}: {e}")

            # Content-based detection
            await self._analyze_content(file_path, metadata)

            # Calculate checksums
            await self._calculate_checksums(file_path, metadata)

            # Determine final format
            metadata.detected_format = self._determine_format(metadata)

            # Calculate confidence score
            metadata.confidence_score = self._calculate_confidence(metadata)

            logger.debug(f"Detected format for {file_path.name}: {metadata.detected_format} (confidence: {metadata.confidence_score:.2f})")

            return metadata

        except Exception as e:
            logger.error(f"Error detecting format for {file_path}: {e}")
            return FileMetadata(
                file_path=file_path_str,
                original_name=file_path.name,
                size_bytes=0,
                mime_type="application/octet-stream",
                detected_format="unknown",
                confidence_score=0.0
            )

    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type using multiple methods."""
        mime_type = "application/octet-stream"

        # Method 1: python-magic
        if self._magic_available:
            try:
                mime_type = self.magic.from_file(str(file_path))
            except Exception as e:
                logger.debug(f"Magic MIME detection failed: {e}")

        # Method 2: mimetypes module (fallback)
        if mime_type == "application/octet-stream":
            guessed_type, _ = mimetypes.guess_type(str(file_path))
            if guessed_type:
                mime_type = guessed_type

        # Method 3: extension-based detection (final fallback)
        if mime_type == "application/octet-stream":
            extension = file_path.suffix.lower()
            extension_map = {
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.json': 'application/json',
                '.yaml': 'text/yaml',
                '.yml': 'text/yaml',
                '.py': 'text/x-python',
                '.js': 'text/javascript',
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            }
            mime_type = extension_map.get(extension, mime_type)

        return mime_type

    async def _analyze_content(self, file_path: Path, metadata: FileMetadata) -> None:
        """Analyze file content for format detection."""
        if metadata.size_bytes == 0:
            return

        try:
            # Read first chunk for analysis
            chunk_size = min(8192, metadata.size_bytes)  # Read up to 8KB

            with open(file_path, 'rb') as f:
                chunk = f.read(chunk_size)

            # Store content preview (first 500 chars if text)
            try:
                text_preview = chunk.decode('utf-8', errors='ignore')[:500]
                metadata.content_preview = text_preview
            except Exception:
                metadata.content_preview = f"<binary data: {len(chunk)} bytes>"

            # Detect language for code files
            if metadata.mime_type.startswith('text/'):
                detected_lang = self._detect_programming_language(file_path, chunk)
                if detected_lang:
                    metadata.language_detected = detected_lang

            # Check binary ratio
            if len(chunk) > 0:
                binary_bytes = sum(1 for byte in chunk if byte < 32 and byte not in [9, 10, 13])
                binary_ratio = binary_bytes / len(chunk)
                metadata.custom_attributes['binary_ratio'] = binary_ratio

        except Exception as e:
            logger.debug(f"Content analysis failed for {file_path}: {e}")

    async def _calculate_checksums(self, file_path: Path, metadata: FileMetadata) -> None:
        """Calculate file checksums."""
        try:
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()

            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)

            metadata.checksum_md5 = md5_hash.hexdigest()
            metadata.checksum_sha256 = sha256_hash.hexdigest()

        except Exception as e:
            logger.debug(f"Checksum calculation failed for {file_path}: {e}")

    def _determine_format(self, metadata: FileMetadata) -> str:
        """Determine final format based on all available information."""
        # Priority: content analysis > MIME type > extension

        file_path = Path(metadata.file_path)
        extension = file_path.suffix.lower()

        # Check for specific patterns in content
        if metadata.content_preview:
            content = metadata.content_preview.lower()

            # Specific format detection
            if content.startswith('{') or content.startswith('['):
                if self._validate_json(content):
                    return 'json'

            if content.startswith('---') or 'yaml' in content[:50]:
                return 'yaml'

            if any(pattern in content[:200] for pattern in ['import ', 'def ', 'class ', '#!/usr/bin/python']):
                return 'python'

        # Use MIME type
        mime_type = metadata.mime_type.lower()
        format_mapping = {
            'text/plain': 'text',
            'text/markdown': 'markdown',
            'application/json': 'json',
            'text/yaml': 'yaml',
            'text/x-python': 'python',
            'text/javascript': 'javascript',
            'application/pdf': 'pdf'
        }

        detected_format = format_mapping.get(mime_type)
        if detected_format:
            return detected_format

        # Fallback to extension
        extension_mapping = {
            '.txt': 'text',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.py': 'python',
            '.js': 'javascript',
            '.pdf': 'pdf'
        }

        return extension_mapping.get(extension, 'unknown')

    def _detect_programming_language(self, file_path: Path, content: bytes) -> str | None:
        """Detect programming language from file content."""
        try:
            text_content = content.decode('utf-8', errors='ignore').lower()

            # Language detection patterns
            language_patterns = {
                'python': [r'import\s+\w+', r'def\s+\w+', r'class\s+\w+', r'#!/usr/bin/(env\s+)?python'],
                'javascript': [r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+', r'require\('],
                'typescript': [r'interface\s+\w+', r'type\s+\w+', r'export\s+', r'import.*from'],
                'rust': [r'fn\s+\w+', r'struct\s+\w+', r'impl\s+', r'use\s+\w+'],
                'go': [r'package\s+\w+', r'func\s+\w+', r'import\s+\(', r'type\s+\w+\s+struct'],
                'java': [r'public\s+class', r'import\s+java\.', r'public\s+static\s+void\s+main'],
                'c++': [r'#include\s*<\w+>', r'using\s+namespace', r'std::', r'class\s+\w+'],
                'c': [r'#include\s*<\w+\.h>', r'int\s+main\s*\(', r'printf\s*\('],
            }

            import re
            for language, patterns in language_patterns.items():
                if any(re.search(pattern, text_content) for pattern in patterns):
                    return language

            return None

        except Exception:
            return None

    def _calculate_confidence(self, metadata: FileMetadata) -> float:
        """Calculate confidence score for format detection."""
        confidence = 0.0

        # MIME type detection confidence
        if metadata.mime_type != "application/octet-stream":
            confidence += 0.4

        # Magic signature confidence
        if metadata.magic_signature and "data" not in metadata.magic_signature.lower():
            confidence += 0.3

        # Content analysis confidence
        if metadata.content_preview and metadata.detected_format != "unknown":
            confidence += 0.3

        # Consistency check
        file_path = Path(metadata.file_path)
        extension = file_path.suffix.lower()
        format_extension_map = {
            'json': ['.json'],
            'yaml': ['.yaml', '.yml'],
            'python': ['.py'],
            'markdown': ['.md'],
            'text': ['.txt']
        }

        if metadata.detected_format in format_extension_map:
            if extension in format_extension_map[metadata.detected_format]:
                confidence += 0.2

        return min(confidence, 1.0)

    def _validate_json(self, content: str) -> bool:
        """Validate JSON content."""
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


class FileValidator:
    """Comprehensive file validation and security scanning."""

    def __init__(self, max_file_size: int = 100 * 1024 * 1024):  # 100MB default
        """Initialize file validator."""
        self.max_file_size = max_file_size
        self.suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'eval(',
            b'exec(',
            b'system(',
            b'shell_exec'
        ]

        # Known malicious file signatures (simplified examples)
        self.malicious_signatures = [
            b'\x4d\x5a\x90\x00',  # PE executable header
            b'\x7f\x45\x4c\x46',  # ELF executable header
        ]

    async def validate_file(self, file_path: Path, metadata: FileMetadata) -> ValidationReport:
        """Perform comprehensive file validation."""
        report = ValidationReport(
            file_path=str(file_path),
            validation_result=ValidationResult.VALID,
            mime_type_matches=True,
            integrity_check_passed=True,
            format_validation_passed=True,
            security_scan_passed=True
        )

        try:
            # Size validation
            if metadata.size_bytes > self.max_file_size:
                report.validation_result = ValidationResult.TOO_LARGE
                report.issues_found.append(f"File size ({metadata.size_bytes} bytes) exceeds limit ({self.max_file_size} bytes)")

            # File existence and accessibility
            if not file_path.exists():
                report.validation_result = ValidationResult.INVALID
                report.issues_found.append("File does not exist")
                return report

            if not file_path.is_file():
                report.validation_result = ValidationResult.INVALID
                report.issues_found.append("Path is not a regular file")
                return report

            # Basic integrity check
            await self._check_file_integrity(file_path, metadata, report)

            # MIME type validation
            await self._validate_mime_type(file_path, metadata, report)

            # Format-specific validation
            await self._validate_format(file_path, metadata, report)

            # Security scanning
            await self._security_scan(file_path, metadata, report)

            # Final validation result
            if len(report.issues_found) > 0:
                if report.validation_result == ValidationResult.VALID:
                    report.validation_result = ValidationResult.SUSPICIOUS

            logger.debug(f"Validation completed for {file_path.name}: {report.validation_result}")

        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            report.validation_result = ValidationResult.INVALID
            report.issues_found.append(f"Validation error: {str(e)}")

        return report

    async def _check_file_integrity(self, file_path: Path, metadata: FileMetadata, report: ValidationReport) -> None:
        """Check basic file integrity."""
        try:
            # Try to read the file
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB

            # Check if file size matches metadata
            actual_size = file_path.stat().st_size
            if actual_size != metadata.size_bytes:
                report.integrity_check_passed = False
                report.issues_found.append(f"File size mismatch: expected {metadata.size_bytes}, got {actual_size}")

        except PermissionError:
            report.integrity_check_passed = False
            report.issues_found.append("Permission denied when reading file")
        except Exception as e:
            report.integrity_check_passed = False
            report.issues_found.append(f"File integrity check failed: {str(e)}")

    async def _validate_mime_type(self, file_path: Path, metadata: FileMetadata, report: ValidationReport) -> None:
        """Validate MIME type consistency."""
        try:
            # Check if file extension matches detected MIME type
            extension = file_path.suffix.lower()
            expected_extensions = {
                'text/plain': ['.txt', '.text', '.log'],
                'text/markdown': ['.md', '.markdown'],
                'application/json': ['.json'],
                'text/yaml': ['.yaml', '.yml'],
                'text/x-python': ['.py'],
                'application/pdf': ['.pdf']
            }

            if metadata.mime_type in expected_extensions:
                if extension not in expected_extensions[metadata.mime_type]:
                    report.mime_type_matches = False
                    report.warnings.append(f"Extension {extension} doesn't match MIME type {metadata.mime_type}")

        except Exception as e:
            logger.debug(f"MIME type validation error: {e}")

    async def _validate_format(self, file_path: Path, metadata: FileMetadata, report: ValidationReport) -> None:
        """Validate file format specific rules."""
        try:
            if metadata.detected_format == 'json':
                await self._validate_json_format(file_path, report)
            elif metadata.detected_format == 'yaml':
                await self._validate_yaml_format(file_path, report)
            elif metadata.detected_format == 'pdf':
                await self._validate_pdf_format(file_path, report)
            # Add more format-specific validations as needed

        except Exception as e:
            logger.debug(f"Format validation error: {e}")
            report.format_validation_passed = False
            report.issues_found.append(f"Format validation failed: {str(e)}")

    async def _validate_json_format(self, file_path: Path, report: ValidationReport) -> None:
        """Validate JSON file format."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
                json.loads(content)  # Will raise JSONDecodeError if invalid
        except json.JSONDecodeError as e:
            report.format_validation_passed = False
            report.issues_found.append(f"Invalid JSON: {str(e)}")
        except UnicodeDecodeError as e:
            report.format_validation_passed = False
            report.issues_found.append(f"JSON encoding error: {str(e)}")

    async def _validate_yaml_format(self, file_path: Path, report: ValidationReport) -> None:
        """Validate YAML file format."""
        try:
            import yaml
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
                yaml.safe_load(content)  # Will raise YAMLError if invalid
        except ImportError:
            report.warnings.append("YAML library not available for validation")
        except yaml.YAMLError as e:
            report.format_validation_passed = False
            report.issues_found.append(f"Invalid YAML: {str(e)}")
        except Exception as e:
            report.warnings.append(f"YAML validation warning: {str(e)}")

    async def _validate_pdf_format(self, file_path: Path, report: ValidationReport) -> None:
        """Validate PDF file format."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    report.format_validation_passed = False
                    report.issues_found.append("Invalid PDF header")
        except Exception as e:
            report.warnings.append(f"PDF validation warning: {str(e)}")

    async def _security_scan(self, file_path: Path, metadata: FileMetadata, report: ValidationReport) -> None:
        """Perform basic security scanning."""
        try:
            # Check for suspicious file patterns
            with open(file_path, 'rb') as f:
                content = f.read(min(8192, metadata.size_bytes))  # Read up to 8KB

            # Check for malicious signatures
            for signature in self.malicious_signatures:
                if signature in content:
                    report.security_scan_passed = False
                    report.issues_found.append("Potentially malicious signature detected")
                    break

            # Check for suspicious patterns
            suspicious_found = []
            for pattern in self.suspicious_patterns:
                if pattern in content:
                    suspicious_found.append(pattern.decode('utf-8', errors='ignore'))

            if suspicious_found:
                report.warnings.append(f"Suspicious patterns found: {', '.join(suspicious_found)}")

            # Check file permissions (if on Unix-like system)
            try:
                mode = file_path.stat().st_mode
                if mode & 0o111:  # File is executable
                    if metadata.mime_type.startswith('text/'):
                        report.warnings.append("Text file with executable permissions")
            except Exception:
                pass  # Permission check failed, not critical

        except Exception as e:
            logger.debug(f"Security scan error: {e}")


class ProcessingOrchestrator:
    """Orchestrates file processing with different strategies and pipelines."""

    def __init__(self,
                 priority_queue: AdvancedPriorityQueue,
                 update_system: IncrementalUpdateSystem,
                 max_concurrent_processing: int = 5,
                 quarantine_dir: Path | None = None):
        """Initialize processing orchestrator."""
        self.priority_queue = priority_queue
        self.update_system = update_system
        self.max_concurrent_processing = max_concurrent_processing

        # Initialize components
        self.format_detector = FormatDetector()
        self.file_validator = FileValidator()

        # Setup quarantine directory
        self.quarantine_dir = quarantine_dir or Path(tempfile.gettempdir()) / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Processing semaphore
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_processing)

        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'files_quarantined': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'formats_detected': {},
            'validation_results': {},
        }

        # Registered processors for different formats
        self._format_processors: dict[str, Callable] = {}
        self._processing_callbacks: list[Callable] = []

    def register_format_processor(self, format_type: str, processor: Callable[[Path, FileMetadata], Any]) -> None:
        """Register a processor for a specific file format."""
        self._format_processors[format_type] = processor
        logger.info(f"Registered processor for format: {format_type}")

    def add_processing_callback(self, callback: Callable[[ProcessingResult], None]) -> None:
        """Add a callback to be called after each file is processed."""
        self._processing_callbacks.append(callback)

    async def process_file(self, file_path: Path, strategy: ProcessingStrategy = ProcessingStrategy.IMMEDIATE) -> ProcessingResult:
        """Process a single file with the specified strategy."""
        start_time = time.time()

        result = ProcessingResult(
            file_path=str(file_path),
            status=FileStatus.PENDING,
            processing_time=0.0,
            processing_strategy_used=strategy
        )

        try:
            async with self._processing_semaphore:
                result.status = FileStatus.PROCESSING

                # Step 1: Format detection
                logger.debug(f"Detecting format for {file_path.name}")
                metadata = await self.format_detector.detect_format(file_path)
                result.metadata = metadata

                # Step 2: Validation
                logger.debug(f"Validating {file_path.name}")
                validation_report = await self.file_validator.validate_file(file_path, metadata)
                result.validation_report = validation_report

                # Step 3: Security check
                if validation_report.should_quarantine():
                    await self._quarantine_file(file_path, validation_report)
                    result.status = FileStatus.QUARANTINED
                    self.stats['files_quarantined'] += 1
                    logger.warning(f"File quarantined: {file_path.name}")
                    return result

                if not validation_report.is_safe_to_process():
                    result.status = FileStatus.SKIPPED
                    result.error_message = f"File failed validation: {', '.join(validation_report.issues_found)}"
                    logger.warning(f"File skipped: {file_path.name} - {result.error_message}")
                    return result

                # Step 4: Process based on strategy
                if strategy == ProcessingStrategy.IMMEDIATE:
                    await self._process_immediately(file_path, result)
                elif strategy == ProcessingStrategy.BATCH:
                    await self._process_in_batch(file_path, result)
                elif strategy == ProcessingStrategy.DEFERRED:
                    await self._process_deferred(file_path, result)
                elif strategy == ProcessingStrategy.BACKGROUND:
                    await self._process_in_background(file_path, result)

                # Step 5: Update statistics
                self._update_statistics(result)

                logger.info(f"Successfully processed {file_path.name} in {result.processing_time:.2f}s")

        except Exception as e:
            result.status = FileStatus.FAILED
            result.error_message = str(e)
            self.stats['files_failed'] += 1
            logger.error(f"Failed to process {file_path.name}: {e}")
        finally:
            result.processing_time = time.time() - start_time

            # Call registered callbacks
            for callback in self._processing_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Processing callback error: {e}")

        return result

    async def _process_immediately(self, file_path: Path, result: ProcessingResult) -> None:
        """Process file immediately."""
        metadata = result.metadata

        # Try format-specific processor
        if metadata.detected_format in self._format_processors:
            processor = self._format_processors[metadata.detected_format]
            try:
                extracted_content = await processor(file_path, metadata)
                result.extracted_content = extracted_content
                result.status = FileStatus.COMPLETED
            except Exception as e:
                logger.error(f"Format processor failed for {file_path.name}: {e}")
                result.status = FileStatus.FAILED
                result.error_message = str(e)
        else:
            # Generic processing
            await self._generic_processing(file_path, result)

    async def _process_in_batch(self, file_path: Path, result: ProcessingResult) -> None:
        """Add file to batch for later processing."""
        # Create a priority task for batch processing
        task = PriorityTask(
            task_id=f"batch_{file_path.name}_{int(time.time())}",
            priority=TaskPriority.BACKGROUND,
            payload={
                'file_path': str(file_path),
                'processing_result': result
            },
            callback=self._batch_processor_callback
        )

        success = self.priority_queue.put(task)
        if success:
            result.status = FileStatus.PENDING
            logger.debug(f"Added {file_path.name} to batch processing queue")
        else:
            result.status = FileStatus.FAILED
            result.error_message = "Failed to add to processing queue"

    async def _process_deferred(self, file_path: Path, result: ProcessingResult) -> None:
        """Schedule file for deferred processing."""
        # Add to priority queue with low priority
        task = PriorityTask(
            task_id=f"deferred_{file_path.name}_{int(time.time())}",
            priority=TaskPriority.LOW,
            payload={
                'file_path': str(file_path),
                'processing_result': result
            }
        )

        success = self.priority_queue.put(task)
        if success:
            result.status = FileStatus.PENDING
            logger.debug(f"Scheduled {file_path.name} for deferred processing")
        else:
            result.status = FileStatus.FAILED
            result.error_message = "Failed to schedule for deferred processing"

    async def _process_in_background(self, file_path: Path, result: ProcessingResult) -> None:
        """Process file in background task."""
        # Create background task
        task = asyncio.create_task(self._background_processor(file_path, result))
        result.custom_data['background_task'] = task
        result.status = FileStatus.PROCESSING
        logger.debug(f"Started background processing for {file_path.name}")

    async def _generic_processing(self, file_path: Path, result: ProcessingResult) -> None:
        """Generic file processing for unsupported formats."""
        try:
            metadata = result.metadata

            if metadata.mime_type.startswith('text/'):
                # Text file processing
                with open(file_path, encoding=metadata.encoding or 'utf-8', errors='ignore') as f:
                    content = f.read()
                result.extracted_content = content
                result.status = FileStatus.COMPLETED
            else:
                # Binary file - just store metadata
                result.extracted_content = f"Binary file: {metadata.mime_type}"
                result.status = FileStatus.COMPLETED

        except Exception as e:
            result.status = FileStatus.FAILED
            result.error_message = f"Generic processing failed: {str(e)}"

    async def _batch_processor_callback(self, payload: dict[str, Any]) -> None:
        """Callback for batch processing."""
        file_path = Path(payload['file_path'])
        result = payload['processing_result']
        await self._process_immediately(file_path, result)

    async def _background_processor(self, file_path: Path, result: ProcessingResult) -> None:
        """Background processor task."""
        try:
            await self._process_immediately(file_path, result)
        except Exception as e:
            result.status = FileStatus.FAILED
            result.error_message = f"Background processing failed: {str(e)}"

    async def _quarantine_file(self, file_path: Path, validation_report: ValidationReport) -> None:
        """Move file to quarantine directory."""
        try:
            quarantine_path = self.quarantine_dir / f"{file_path.name}.quarantined"

            # Copy file to quarantine (don't move original)
            import shutil
            shutil.copy2(file_path, quarantine_path)

            # Create report file
            report_path = quarantine_path.with_suffix('.report.json')
            with open(report_path, 'w') as f:
                json.dump({
                    'original_path': str(file_path),
                    'quarantined_at': datetime.now(timezone.utc).isoformat(),
                    'validation_report': {
                        'validation_result': validation_report.validation_result.value,
                        'issues_found': validation_report.issues_found,
                        'security_scan_passed': validation_report.security_scan_passed
                    }
                }, f, indent=2)

            logger.warning(f"File quarantined: {file_path.name} -> {quarantine_path}")

        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path.name}: {e}")

    def _update_statistics(self, result: ProcessingResult) -> None:
        """Update processing statistics."""
        self.stats['files_processed'] += 1
        self.stats['total_processing_time'] += result.processing_time
        self.stats['average_processing_time'] = self.stats['total_processing_time'] / self.stats['files_processed']

        if result.metadata:
            format_type = result.metadata.detected_format
            self.stats['formats_detected'][format_type] = self.stats['formats_detected'].get(format_type, 0) + 1

        if result.validation_report:
            validation_result = result.validation_report.validation_result.value
            self.stats['validation_results'][validation_result] = self.stats['validation_results'].get(validation_result, 0) + 1

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive processing statistics."""
        return self.stats.copy()

    async def process_directory(self, directory_path: Path, recursive: bool = True,
                              strategy: ProcessingStrategy = ProcessingStrategy.BATCH) -> list[ProcessingResult]:
        """Process all files in a directory."""
        results = []

        if recursive:
            files = list(directory_path.rglob('*'))
        else:
            files = list(directory_path.iterdir())

        # Filter to only regular files
        files = [f for f in files if f.is_file()]

        logger.info(f"Processing {len(files)} files from {directory_path}")

        # Process files concurrently
        tasks = [self.process_file(file_path, strategy) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ProcessingResult(
                    file_path=str(files[i]),
                    status=FileStatus.FAILED,
                    processing_time=0.0,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        return processed_results

    async def retry_failed_processing(self, max_retries: int = 3) -> int:
        """Retry files that failed processing."""
        # This would typically work with a persistent storage of failed files
        # For now, it's a placeholder for the retry mechanism
        logger.info("Retry mechanism would be implemented here")
        return 0


class EnhancedAutoIngestionSystem:
    """
    Complete auto-ingestion system with format detection, validation, and processing.

    Integrates format detection, validation, priority queuing, and incremental updates
    for a comprehensive file processing solution.
    """

    def __init__(self,
                 backup_dir: Path,
                 quarantine_dir: Path | None = None,
                 max_concurrent_tasks: int = 10,
                 enable_incremental_updates: bool = True):
        """Initialize the enhanced auto-ingestion system."""

        # Core components
        self.priority_queue = AdvancedPriorityQueue(
            max_concurrent_tasks=max_concurrent_tasks,
            enable_resource_monitoring=True
        )

        if enable_incremental_updates:
            self.update_system = IncrementalUpdateSystem(backup_dir)
        else:
            self.update_system = None

        self.processing_orchestrator = ProcessingOrchestrator(
            priority_queue=self.priority_queue,
            update_system=self.update_system,
            quarantine_dir=quarantine_dir
        )

        # State tracking
        self._initialized = False
        self._running = False

        # Callbacks
        self._ingestion_callbacks: list[Callable] = []

    async def initialize(self) -> None:
        """Initialize the auto-ingestion system."""
        if self._initialized:
            return

        # Start priority queue
        await self.priority_queue.start()

        # Set up processing callback for incremental updates
        if self.update_system:
            self.update_system.set_processing_callback(self._handle_incremental_update)

        # Add processing callback
        self.processing_orchestrator.add_processing_callback(self._processing_callback)

        self._initialized = True
        logger.info("Enhanced auto-ingestion system initialized")

    async def shutdown(self) -> None:
        """Shutdown the auto-ingestion system."""
        self._running = False
        await self.priority_queue.stop()
        logger.info("Enhanced auto-ingestion system shutdown")

    def register_ingestion_callback(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register callback for successful ingestion."""
        self._ingestion_callbacks.append(callback)

    def register_format_processor(self, format_type: str, processor: Callable) -> None:
        """Register a processor for a specific file format."""
        self.processing_orchestrator.register_format_processor(format_type, processor)

    async def ingest_file(self, file_path: Path, collection: str = "default",
                         strategy: ProcessingStrategy = ProcessingStrategy.IMMEDIATE) -> ProcessingResult:
        """Ingest a single file."""
        if not self._initialized:
            await self.initialize()

        logger.info(f"Ingesting file: {file_path.name} -> {collection}")

        result = await self.processing_orchestrator.process_file(file_path, strategy)
        result.collection_assigned = collection

        # Call ingestion callbacks if successful
        if result.status == FileStatus.COMPLETED:
            for callback in self._ingestion_callbacks:
                try:
                    await callback(str(file_path), {
                        'collection': collection,
                        'metadata': result.metadata.to_dict() if result.metadata else {},
                        'validation_report': result.validation_report.__dict__ if result.validation_report else {}
                    })
                except Exception as e:
                    logger.error(f"Ingestion callback error: {e}")

        return result

    async def ingest_directory(self, directory_path: Path, collection: str = "default",
                             recursive: bool = True, strategy: ProcessingStrategy = ProcessingStrategy.BATCH) -> list[ProcessingResult]:
        """Ingest all files in a directory."""
        if not self._initialized:
            await self.initialize()

        logger.info(f"Ingesting directory: {directory_path} -> {collection} (recursive: {recursive})")

        results = await self.processing_orchestrator.process_directory(directory_path, recursive, strategy)

        # Set collection for all results
        for result in results:
            result.collection_assigned = collection

        return results

    async def watch_and_ingest(self, watch_path: Path, collection: str = "default") -> None:
        """Watch a directory and automatically ingest new/modified files."""
        if not self._initialized:
            await self.initialize()

        if not self.update_system:
            raise ValueError("Incremental updates must be enabled for watching")

        # Initialize baseline
        files_to_watch = list(watch_path.rglob('*')) if watch_path.is_dir() else [watch_path]
        files_to_watch = [f for f in files_to_watch if f.is_file()]

        await self.update_system.initialize_baseline(files_to_watch)

        logger.info(f"Started watching {watch_path} for changes")

        # Start watching loop
        self._running = True
        while self._running:
            try:
                # Check for changes
                results = await self.update_system.process_file_changes(files_to_watch)

                if results['changes_detected'] > 0:
                    logger.info(f"Detected {results['changes_detected']} file changes")

                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                await asyncio.sleep(10)  # Wait longer after error

    async def _handle_incremental_update(self, file_path: str, change_record: ChangeRecord) -> None:
        """Handle incremental update from the update system."""
        try:
            path = Path(file_path)
            result = await self.processing_orchestrator.process_file(path)
            logger.debug(f"Processed incremental update: {path.name} -> {result.status}")
        except Exception as e:
            logger.error(f"Failed to handle incremental update for {file_path}: {e}")

    async def _processing_callback(self, result: ProcessingResult) -> None:
        """Callback called after each file is processed."""
        # Log processing results
        if result.status == FileStatus.COMPLETED:
            logger.info(f"Successfully processed: {Path(result.file_path).name}")
        elif result.status == FileStatus.FAILED:
            logger.error(f"Failed to process: {Path(result.file_path).name} - {result.error_message}")
        elif result.status == FileStatus.QUARANTINED:
            logger.warning(f"Quarantined file: {Path(result.file_path).name}")

    def get_comprehensive_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics from all components."""
        stats = {
            'ingestion_system': {
                'initialized': self._initialized,
                'running': self._running,
                'processing_stats': self.processing_orchestrator.get_statistics()
            },
            'priority_queue': self.priority_queue.get_statistics(),
            'quarantine_dir': str(self.processing_orchestrator.quarantine_dir)
        }

        if self.update_system:
            stats['incremental_updates'] = self.update_system.get_statistics()

        return stats


# Default format processors
async def default_json_processor(file_path: Path, metadata: FileMetadata) -> str:
    """Default JSON file processor."""
    with open(file_path, encoding='utf-8') as f:
        content = f.read()
        # Parse and reformat JSON for consistency
        data = json.loads(content)
        return json.dumps(data, indent=2, sort_keys=True)

async def default_text_processor(file_path: Path, metadata: FileMetadata) -> str:
    """Default text file processor."""
    encoding = metadata.encoding or 'utf-8'
    with open(file_path, encoding=encoding, errors='ignore') as f:
        return f.read()

async def default_python_processor(file_path: Path, metadata: FileMetadata) -> str:
    """Default Python file processor."""
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()
        # Add metadata about the Python file
        metadata.custom_attributes['imports'] = []
        metadata.custom_attributes['functions'] = []
        metadata.custom_attributes['classes'] = []

        import re
        # Extract imports
        imports = re.findall(r'^(?:from\s+\w+(?:\.\w+)*\s+)?import\s+[\w\s,*.]+', content, re.MULTILINE)
        metadata.custom_attributes['imports'] = imports

        # Extract function definitions
        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        metadata.custom_attributes['functions'] = functions

        # Extract class definitions
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        metadata.custom_attributes['classes'] = classes

        return content
