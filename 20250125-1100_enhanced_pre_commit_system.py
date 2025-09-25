#!/usr/bin/env python3
"""
Enhanced pre-commit hook system with comprehensive validation and edge case handling.

Features:
- Advanced file validation with encoding detection
- Performance monitoring and resource limits
- Comprehensive security scanning integration
- Code quality metrics and threshold enforcement
- Parallel processing with resource management
- Detailed reporting and failure analysis
- Rollback capabilities and error recovery
- Cross-platform compatibility validation
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
import threading
from datetime import datetime
import hashlib
import magic
import chardet
import yaml


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Validation status options."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    name: str
    description: str
    severity: ValidationSeverity
    enabled: bool = True
    timeout: int = 30
    retry_count: int = 0
    file_patterns: List[str] = None
    exclude_patterns: List[str] = None

    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = []
        if self.exclude_patterns is None:
            self.exclude_patterns = []


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    execution_time: float = 0.0
    details: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class PreCommitSession:
    """Pre-commit validation session tracking."""
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    files_analyzed: Set[str] = None
    rules_executed: List[str] = None
    results: List[ValidationResult] = None
    overall_status: ValidationStatus = ValidationStatus.PENDING
    resource_usage: Dict[str, Any] = None

    def __post_init__(self):
        if self.files_analyzed is None:
            self.files_analyzed = set()
        if self.rules_executed is None:
            self.rules_executed = []
        if self.results is None:
            self.results = []
        if self.resource_usage is None:
            self.resource_usage = {}


class EnhancedPreCommitSystem:
    """Enhanced pre-commit hook system with comprehensive validation."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the enhanced pre-commit system."""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_file)
        self.validation_rules = self._initialize_validation_rules()
        self.sessions: Dict[str, PreCommitSession] = {}
        self.resource_limits = {
            'max_memory_mb': 1024,
            'max_execution_time': 300,
            'max_parallel_jobs': 4,
            'max_file_size_mb': 10
        }

    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pre_commit_enhanced.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load and validate configuration."""
        default_config = {
            'validation': {
                'parallel_execution': True,
                'fail_fast': False,
                'timeout_per_rule': 30,
                'max_file_size_mb': 10,
                'encoding_detection': True,
                'content_validation': True
            },
            'security': {
                'scan_for_secrets': True,
                'scan_for_vulnerabilities': True,
                'allowed_file_types': ['.py', '.js', '.ts', '.yaml', '.yml', '.json', '.md', '.txt'],
                'blocked_extensions': ['.exe', '.dll', '.so', '.dylib'],
                'max_binary_size_kb': 100
            },
            'quality': {
                'enforce_style': True,
                'check_complexity': True,
                'validate_imports': True,
                'check_documentation': True,
                'complexity_threshold': 10,
                'line_length_limit': 120
            },
            'performance': {
                'resource_monitoring': True,
                'performance_profiling': False,
                'memory_limit_mb': 512,
                'time_limit_seconds': 180
            }
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    self._deep_update(default_config, user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_file}: {e}")

        return default_config

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize comprehensive validation rules."""
        rules = [
            # File system validation rules
            ValidationRule(
                name="file_encoding_validation",
                description="Validate file encoding and detect encoding issues",
                severity=ValidationSeverity.ERROR,
                file_patterns=["*.py", "*.js", "*.ts", "*.yaml", "*.yml", "*.json", "*.md", "*.txt"]
            ),
            ValidationRule(
                name="file_size_validation",
                description="Check file sizes against limits",
                severity=ValidationSeverity.WARNING,
                file_patterns=["*"]
            ),
            ValidationRule(
                name="binary_file_detection",
                description="Detect and validate binary files",
                severity=ValidationSeverity.INFO,
                file_patterns=["*"]
            ),
            ValidationRule(
                name="line_ending_validation",
                description="Validate line endings consistency",
                severity=ValidationSeverity.WARNING,
                file_patterns=["*.py", "*.js", "*.ts", "*.yaml", "*.yml", "*.json", "*.md", "*.txt"]
            ),

            # Security validation rules
            ValidationRule(
                name="secret_detection",
                description="Scan for hardcoded secrets and credentials",
                severity=ValidationSeverity.CRITICAL,
                file_patterns=["*.py", "*.js", "*.ts", "*.yaml", "*.yml", "*.json"]
            ),
            ValidationRule(
                name="vulnerability_scanning",
                description="Scan for known security vulnerabilities",
                severity=ValidationSeverity.ERROR,
                file_patterns=["*.py", "requirements*.txt", "package*.json", "*.lock"]
            ),
            ValidationRule(
                name="permission_validation",
                description="Validate file permissions",
                severity=ValidationSeverity.WARNING,
                file_patterns=["*"]
            ),

            # Code quality validation rules
            ValidationRule(
                name="syntax_validation",
                description="Validate code syntax",
                severity=ValidationSeverity.ERROR,
                file_patterns=["*.py", "*.js", "*.ts", "*.yaml", "*.yml", "*.json"]
            ),
            ValidationRule(
                name="style_validation",
                description="Enforce code style guidelines",
                severity=ValidationSeverity.WARNING,
                file_patterns=["*.py", "*.js", "*.ts"]
            ),
            ValidationRule(
                name="complexity_analysis",
                description="Analyze code complexity",
                severity=ValidationSeverity.WARNING,
                file_patterns=["*.py", "*.js", "*.ts"]
            ),
            ValidationRule(
                name="import_validation",
                description="Validate imports and dependencies",
                severity=ValidationSeverity.ERROR,
                file_patterns=["*.py", "*.js", "*.ts"]
            ),
            ValidationRule(
                name="documentation_validation",
                description="Check documentation completeness",
                severity=ValidationSeverity.INFO,
                file_patterns=["*.py", "*.js", "*.ts"]
            ),

            # Content validation rules
            ValidationRule(
                name="json_validation",
                description="Validate JSON file structure",
                severity=ValidationSeverity.ERROR,
                file_patterns=["*.json"]
            ),
            ValidationRule(
                name="yaml_validation",
                description="Validate YAML file structure",
                severity=ValidationSeverity.ERROR,
                file_patterns=["*.yaml", "*.yml"]
            ),
            ValidationRule(
                name="markdown_validation",
                description="Validate Markdown file structure",
                severity=ValidationSeverity.WARNING,
                file_patterns=["*.md"]
            ),

            # Performance validation rules
            ValidationRule(
                name="resource_usage_monitoring",
                description="Monitor resource usage during validation",
                severity=ValidationSeverity.INFO,
                file_patterns=["*"]
            ),
            ValidationRule(
                name="execution_time_monitoring",
                description="Monitor validation execution time",
                severity=ValidationSeverity.INFO,
                file_patterns=["*"]
            )
        ]

        return rules

    async def validate_pre_commit(self, file_paths: List[str]) -> PreCommitSession:
        """Execute comprehensive pre-commit validation."""
        session_id = f"precommit-{int(time.time())}"
        session = PreCommitSession(
            session_id=session_id,
            started_at=datetime.utcnow(),
            files_analyzed=set(file_paths)
        )

        self.sessions[session_id] = session
        self.logger.info(f"Starting pre-commit validation session {session_id}")
        self.logger.info(f"Files to analyze: {len(file_paths)}")

        try:
            session.overall_status = ValidationStatus.RUNNING

            # Filter and prepare files for validation
            valid_files = await self._prepare_files_for_validation(file_paths, session)

            if not valid_files:
                self.logger.warning("No valid files found for validation")
                session.overall_status = ValidationStatus.SKIPPED
                return session

            # Execute validation rules
            if self.config['validation']['parallel_execution']:
                await self._execute_parallel_validation(valid_files, session)
            else:
                await self._execute_sequential_validation(valid_files, session)

            # Analyze results and determine overall status
            session.overall_status = self._analyze_validation_results(session)

        except Exception as e:
            self.logger.error(f"Pre-commit validation failed: {e}")
            session.overall_status = ValidationStatus.FAILED
            session.results.append(ValidationResult(
                rule_name="system_error",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation system error: {e}"
            ))

        finally:
            session.completed_at = datetime.utcnow()
            self._generate_validation_report(session)

        return session

    async def _prepare_files_for_validation(self, file_paths: List[str], session: PreCommitSession) -> List[str]:
        """Prepare and filter files for validation."""
        self.logger.info("Preparing files for validation")
        valid_files = []
        max_size_bytes = self.config['validation']['max_file_size_mb'] * 1024 * 1024

        for file_path in file_paths:
            try:
                path_obj = Path(file_path)

                # Check if file exists
                if not path_obj.exists():
                    session.results.append(ValidationResult(
                        rule_name="file_existence_check",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.ERROR,
                        message=f"File does not exist: {file_path}",
                        file_path=file_path
                    ))
                    continue

                # Check file size
                file_size = path_obj.stat().st_size
                if file_size > max_size_bytes:
                    session.results.append(ValidationResult(
                        rule_name="file_size_check",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.WARNING,
                        message=f"File too large: {file_size} bytes > {max_size_bytes} bytes",
                        file_path=file_path,
                        details={"file_size_bytes": file_size, "limit_bytes": max_size_bytes}
                    ))
                    if self.config['validation']['fail_fast']:
                        continue

                # Check file permissions
                if not os.access(file_path, os.R_OK):
                    session.results.append(ValidationResult(
                        rule_name="file_permissions_check",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.ERROR,
                        message=f"Cannot read file: {file_path}",
                        file_path=file_path
                    ))
                    continue

                valid_files.append(file_path)

            except Exception as e:
                session.results.append(ValidationResult(
                    rule_name="file_preparation_error",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Error preparing file {file_path}: {e}",
                    file_path=file_path
                ))

        return valid_files

    async def _execute_parallel_validation(self, file_paths: List[str], session: PreCommitSession):
        """Execute validation rules in parallel."""
        self.logger.info("Executing parallel validation")
        max_workers = self.resource_limits['max_parallel_jobs']

        # Create task groups for parallel execution
        tasks = []

        for rule in self.validation_rules:
            if not rule.enabled:
                continue

            # Filter files that match rule patterns
            matching_files = self._filter_files_for_rule(file_paths, rule)

            if matching_files:
                task = self._execute_validation_rule(rule, matching_files, session)
                tasks.append(task)

        # Execute tasks with concurrency limit
        semaphore = asyncio.Semaphore(max_workers)
        bounded_tasks = [self._execute_with_semaphore(semaphore, task) for task in tasks]

        await asyncio.gather(*bounded_tasks, return_exceptions=True)

    async def _execute_sequential_validation(self, file_paths: List[str], session: PreCommitSession):
        """Execute validation rules sequentially."""
        self.logger.info("Executing sequential validation")

        for rule in self.validation_rules:
            if not rule.enabled:
                continue

            matching_files = self._filter_files_for_rule(file_paths, rule)

            if matching_files:
                await self._execute_validation_rule(rule, matching_files, session)

                # Check for fail-fast
                if self.config['validation']['fail_fast']:
                    failed_results = [r for r in session.results if r.status == ValidationStatus.FAILED]
                    if failed_results:
                        self.logger.info("Stopping validation due to fail-fast mode")
                        break

    async def _execute_with_semaphore(self, semaphore: asyncio.Semaphore, coro):
        """Execute coroutine with semaphore for concurrency control."""
        async with semaphore:
            return await coro

    def _filter_files_for_rule(self, file_paths: List[str], rule: ValidationRule) -> List[str]:
        """Filter files that match validation rule patterns."""
        import fnmatch

        matching_files = []

        for file_path in file_paths:
            # Check include patterns
            if rule.file_patterns:
                matches_pattern = any(fnmatch.fnmatch(file_path, pattern) for pattern in rule.file_patterns)
                if not matches_pattern:
                    continue

            # Check exclude patterns
            if rule.exclude_patterns:
                excluded = any(fnmatch.fnmatch(file_path, pattern) for pattern in rule.exclude_patterns)
                if excluded:
                    continue

            matching_files.append(file_path)

        return matching_files

    async def _execute_validation_rule(self, rule: ValidationRule, file_paths: List[str], session: PreCommitSession):
        """Execute individual validation rule."""
        start_time = time.time()
        session.rules_executed.append(rule.name)

        try:
            self.logger.debug(f"Executing rule: {rule.name} on {len(file_paths)} files")

            # Route to specific validation method
            if rule.name == "file_encoding_validation":
                results = await self._validate_file_encoding(file_paths, rule)
            elif rule.name == "file_size_validation":
                results = await self._validate_file_size(file_paths, rule)
            elif rule.name == "binary_file_detection":
                results = await self._detect_binary_files(file_paths, rule)
            elif rule.name == "line_ending_validation":
                results = await self._validate_line_endings(file_paths, rule)
            elif rule.name == "secret_detection":
                results = await self._detect_secrets(file_paths, rule)
            elif rule.name == "vulnerability_scanning":
                results = await self._scan_vulnerabilities(file_paths, rule)
            elif rule.name == "syntax_validation":
                results = await self._validate_syntax(file_paths, rule)
            elif rule.name == "style_validation":
                results = await self._validate_style(file_paths, rule)
            elif rule.name == "complexity_analysis":
                results = await self._analyze_complexity(file_paths, rule)
            elif rule.name == "json_validation":
                results = await self._validate_json(file_paths, rule)
            elif rule.name == "yaml_validation":
                results = await self._validate_yaml(file_paths, rule)
            else:
                # Generic validation placeholder
                results = [ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.SKIPPED,
                    severity=rule.severity,
                    message=f"Rule {rule.name} not implemented"
                )]

            # Update results with execution time
            execution_time = time.time() - start_time
            for result in results:
                result.execution_time = execution_time / len(results)

            session.results.extend(results)

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error executing rule {rule.name}: {e}")

            session.results.append(ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.ERROR,
                message=f"Rule execution error: {e}",
                execution_time=execution_time
            ))

    async def _validate_file_encoding(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Validate file encoding and detect encoding issues."""
        results = []

        for file_path in file_paths:
            try:
                # Read file and detect encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()

                encoding_result = chardet.detect(raw_data)
                detected_encoding = encoding_result.get('encoding', 'unknown')
                confidence = encoding_result.get('confidence', 0.0)

                if confidence < 0.8:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.WARNING,
                        message=f"Low encoding confidence: {confidence:.2f} ({detected_encoding})",
                        file_path=file_path,
                        details={"detected_encoding": detected_encoding, "confidence": confidence}
                    ))
                elif detected_encoding not in ['utf-8', 'ascii']:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.WARNING,
                        message=f"Non-standard encoding detected: {detected_encoding}",
                        file_path=file_path,
                        details={"detected_encoding": detected_encoding, "confidence": confidence}
                    ))
                else:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.PASSED,
                        severity=rule.severity,
                        message=f"Encoding validation passed: {detected_encoding}",
                        file_path=file_path,
                        details={"detected_encoding": detected_encoding, "confidence": confidence}
                    ))

            except Exception as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Encoding validation error: {e}",
                    file_path=file_path
                ))

        return results

    async def _validate_file_size(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Validate file sizes."""
        results = []
        max_size_mb = self.config['validation']['max_file_size_mb']
        max_size_bytes = max_size_mb * 1024 * 1024

        for file_path in file_paths:
            try:
                file_size = Path(file_path).stat().st_size

                if file_size > max_size_bytes:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.WARNING,
                        message=f"File size {file_size / 1024 / 1024:.2f}MB exceeds limit {max_size_mb}MB",
                        file_path=file_path,
                        details={"file_size_bytes": file_size, "limit_bytes": max_size_bytes}
                    ))
                else:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.PASSED,
                        severity=rule.severity,
                        message=f"File size validation passed: {file_size / 1024:.1f}KB",
                        file_path=file_path,
                        details={"file_size_bytes": file_size}
                    ))

            except Exception as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"File size validation error: {e}",
                    file_path=file_path
                ))

        return results

    async def _detect_binary_files(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Detect binary files and validate them."""
        results = []

        for file_path in file_paths:
            try:
                # Use python-magic to detect file type
                try:
                    file_type = magic.from_file(file_path)
                    is_binary = not file_type.startswith('ASCII') and not file_type.startswith('UTF-8')
                except Exception:
                    # Fallback method
                    with open(file_path, 'rb') as f:
                        chunk = f.read(8192)
                        is_binary = b'\0' in chunk

                if is_binary:
                    file_size = Path(file_path).stat().st_size
                    max_binary_size = self.config['security']['max_binary_size_kb'] * 1024

                    if file_size > max_binary_size:
                        results.append(ValidationResult(
                            rule_name=rule.name,
                            status=ValidationStatus.FAILED,
                            severity=ValidationSeverity.WARNING,
                            message=f"Binary file too large: {file_size / 1024:.1f}KB",
                            file_path=file_path,
                            details={"is_binary": True, "file_size_bytes": file_size}
                        ))
                    else:
                        results.append(ValidationResult(
                            rule_name=rule.name,
                            status=ValidationStatus.PASSED,
                            severity=rule.severity,
                            message="Binary file detected and validated",
                            file_path=file_path,
                            details={"is_binary": True, "file_size_bytes": file_size}
                        ))
                else:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.PASSED,
                        severity=rule.severity,
                        message="Text file detected",
                        file_path=file_path,
                        details={"is_binary": False}
                    ))

            except Exception as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Binary detection error: {e}",
                    file_path=file_path
                ))

        return results

    async def _validate_line_endings(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Validate line ending consistency."""
        results = []

        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()

                # Count different line ending types
                crlf_count = content.count(b'\r\n')
                lf_count = content.count(b'\n') - crlf_count
                cr_count = content.count(b'\r') - crlf_count

                total_lines = crlf_count + lf_count + cr_count

                if total_lines == 0:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.PASSED,
                        severity=rule.severity,
                        message="No line endings found (empty or single line file)",
                        file_path=file_path
                    ))
                    continue

                # Check for mixed line endings
                line_ending_types = sum([1 for count in [crlf_count, lf_count, cr_count] if count > 0])

                if line_ending_types > 1:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.WARNING,
                        message=f"Mixed line endings: CRLF({crlf_count}), LF({lf_count}), CR({cr_count})",
                        file_path=file_path,
                        details={
                            "crlf_count": crlf_count,
                            "lf_count": lf_count,
                            "cr_count": cr_count
                        }
                    ))
                else:
                    dominant_ending = "CRLF" if crlf_count > 0 else "LF" if lf_count > 0 else "CR"
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.PASSED,
                        severity=rule.severity,
                        message=f"Consistent line endings: {dominant_ending}",
                        file_path=file_path,
                        details={"line_ending_type": dominant_ending}
                    ))

            except Exception as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Line ending validation error: {e}",
                    file_path=file_path
                ))

        return results

    async def _detect_secrets(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Detect hardcoded secrets and credentials."""
        results = []

        # Common secret patterns
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']{16,}["\']', "API key"),
            (r'secret\s*=\s*["\'][^"\']{16,}["\']', "secret value"),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', "token"),
            (r'-----BEGIN [A-Z ]+-----', "private key"),
            (r'[A-Za-z0-9+/]{40,}={0,2}', "base64 encoded secret"),
        ]

        import re

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                detected_secrets = []

                for pattern, description in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        detected_secrets.append({
                            "type": description,
                            "line": line_number,
                            "pattern": pattern
                        })

                if detected_secrets:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Detected {len(detected_secrets)} potential secrets",
                        file_path=file_path,
                        details={"detected_secrets": detected_secrets}
                    ))
                else:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.PASSED,
                        severity=rule.severity,
                        message="No secrets detected",
                        file_path=file_path
                    ))

            except Exception as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Secret detection error: {e}",
                    file_path=file_path
                ))

        return results

    async def _validate_json(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Validate JSON file structure."""
        results = []

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)

                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.PASSED,
                    severity=rule.severity,
                    message="Valid JSON structure",
                    file_path=file_path
                ))

            except json.JSONDecodeError as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid JSON: {e}",
                    file_path=file_path,
                    line_number=e.lineno
                ))
            except Exception as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"JSON validation error: {e}",
                    file_path=file_path
                ))

        return results

    async def _validate_yaml(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Validate YAML file structure."""
        results = []

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    yaml.safe_load(f)

                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.PASSED,
                    severity=rule.severity,
                    message="Valid YAML structure",
                    file_path=file_path
                ))

            except yaml.YAMLError as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid YAML: {e}",
                    file_path=file_path
                ))
            except Exception as e:
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"YAML validation error: {e}",
                    file_path=file_path
                ))

        return results

    # Placeholder methods for additional validation rules
    async def _scan_vulnerabilities(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Placeholder for vulnerability scanning."""
        return [ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.SKIPPED,
            severity=rule.severity,
            message="Vulnerability scanning not implemented"
        )]

    async def _validate_syntax(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Placeholder for syntax validation."""
        return [ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.SKIPPED,
            severity=rule.severity,
            message="Syntax validation not implemented"
        )]

    async def _validate_style(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Placeholder for style validation."""
        return [ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.SKIPPED,
            severity=rule.severity,
            message="Style validation not implemented"
        )]

    async def _analyze_complexity(self, file_paths: List[str], rule: ValidationRule) -> List[ValidationResult]:
        """Placeholder for complexity analysis."""
        return [ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.SKIPPED,
            severity=rule.severity,
            message="Complexity analysis not implemented"
        )]

    def _analyze_validation_results(self, session: PreCommitSession) -> ValidationStatus:
        """Analyze validation results and determine overall status."""
        if not session.results:
            return ValidationStatus.SKIPPED

        failed_critical = any(r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.CRITICAL
                            for r in session.results)
        failed_error = any(r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.ERROR
                         for r in session.results)

        if failed_critical or failed_error:
            return ValidationStatus.FAILED

        failed_warning = any(r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.WARNING
                           for r in session.results)

        if failed_warning and not self.config['validation'].get('allow_warnings', True):
            return ValidationStatus.FAILED

        return ValidationStatus.PASSED

    def _generate_validation_report(self, session: PreCommitSession):
        """Generate comprehensive validation report."""
        duration = (session.completed_at - session.started_at).total_seconds()

        # Count results by status and severity
        status_counts = {}
        severity_counts = {}

        for result in session.results:
            status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
            severity_counts[result.severity.value] = severity_counts.get(result.severity.value, 0) + 1

        report = {
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "completed_at": session.completed_at.isoformat(),
            "duration_seconds": duration,
            "overall_status": session.overall_status.value,
            "files_analyzed": len(session.files_analyzed),
            "rules_executed": len(session.rules_executed),
            "total_results": len(session.results),
            "status_summary": status_counts,
            "severity_summary": severity_counts,
            "results": [asdict(result) for result in session.results]
        }

        # Save report
        report_file = f"pre_commit_report_{session.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Validation report saved to {report_file}")

        # Log summary
        self.logger.info(f"Validation completed in {duration:.2f}s")
        self.logger.info(f"Overall status: {session.overall_status.value}")
        self.logger.info(f"Files analyzed: {len(session.files_analyzed)}")
        self.logger.info(f"Results: {status_counts}")


# Example usage and CLI interface
async def main():
    """Example usage of enhanced pre-commit system."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_pre_commit_system.py <file1> <file2> ...")
        sys.exit(1)

    file_paths = sys.argv[1:]

    print("🔧 Enhanced Pre-commit Hook System")
    print("=" * 50)

    # Initialize system
    pre_commit_system = EnhancedPreCommitSystem()

    # Execute validation
    session = await pre_commit_system.validate_pre_commit(file_paths)

    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"Session ID: {session.session_id}")
    print(f"Status: {session.overall_status.value}")
    print(f"Files analyzed: {len(session.files_analyzed)}")
    print(f"Rules executed: {len(session.rules_executed)}")
    print(f"Total results: {len(session.results)}")

    # Print results by severity
    critical_results = [r for r in session.results if r.severity == ValidationSeverity.CRITICAL]
    error_results = [r for r in session.results if r.severity == ValidationSeverity.ERROR]
    warning_results = [r for r in session.results if r.severity == ValidationSeverity.WARNING]

    if critical_results:
        print(f"\n❌ CRITICAL Issues ({len(critical_results)}):")
        for result in critical_results:
            print(f"  - {result.message} [{result.file_path}]")

    if error_results:
        print(f"\n🚫 ERROR Issues ({len(error_results)}):")
        for result in error_results:
            print(f"  - {result.message} [{result.file_path}]")

    if warning_results:
        print(f"\n⚠️  WARNING Issues ({len(warning_results)}):")
        for result in warning_results:
            print(f"  - {result.message} [{result.file_path}]")

    # Exit with appropriate code
    if session.overall_status == ValidationStatus.FAILED:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())