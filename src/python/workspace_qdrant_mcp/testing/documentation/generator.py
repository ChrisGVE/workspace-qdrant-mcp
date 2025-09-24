"""
Test Documentation Generator

Main orchestrator for generating comprehensive test documentation
from Python test files with multiple output formats and coverage integration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
import concurrent.futures
from threading import Lock

from .parser import TestFileParser, TestFileInfo, TestMetadata, TestType
from .formatters import MarkdownFormatter, HTMLFormatter, JSONFormatter, BaseFormatter

logger = logging.getLogger(__name__)


class CoverageIntegrator:
    """Integrates test coverage data with documentation."""

    def __init__(self, coverage_file: Optional[Path] = None):
        """
        Initialize coverage integrator.

        Args:
            coverage_file: Path to coverage.json file
        """
        self.coverage_file = coverage_file
        self.coverage_data = {}
        self._load_coverage_data()

    def _load_coverage_data(self) -> None:
        """Load coverage data from file."""
        if not self.coverage_file or not self.coverage_file.exists():
            logger.warning(f"Coverage file not found: {self.coverage_file}")
            return

        try:
            with open(self.coverage_file, 'r') as f:
                data = json.load(f)

            # Extract file-level coverage from coverage.py format
            if 'files' in data:
                for file_path, file_data in data['files'].items():
                    if 'summary' in file_data:
                        coverage_percent = file_data['summary'].get('percent_covered', 0)
                        self.coverage_data[Path(file_path)] = coverage_percent

            logger.info(f"Loaded coverage data for {len(self.coverage_data)} files")

        except Exception as e:
            logger.error(f"Failed to load coverage data: {e}")

    def get_coverage(self, file_path: Path) -> Optional[float]:
        """Get coverage percentage for a file."""
        # Try exact match first
        if file_path in self.coverage_data:
            return self.coverage_data[file_path]

        # Try relative path matching
        for cov_path, coverage in self.coverage_data.items():
            if file_path.name == cov_path.name and str(file_path).endswith(str(cov_path.relative_to(cov_path.parts[0]))):
                return coverage

        return None


class TestDatabase:
    """SQLite database for caching parsed test metadata."""

    def __init__(self, db_path: Path):
        """Initialize test database."""
        self.db_path = db_path
        self.lock = Lock()
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS test_files (
                        path TEXT PRIMARY KEY,
                        last_modified REAL,
                        metadata TEXT,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS generation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        generated_at REAL DEFAULT (julianday('now')),
                        file_count INTEGER,
                        test_count INTEGER,
                        format TEXT,
                        output_path TEXT
                    )
                ''')

                conn.commit()
            finally:
                conn.close()

    def get_cached_metadata(self, file_path: Path) -> Optional[TestFileInfo]:
        """Get cached metadata if file hasn't changed."""
        if not file_path.exists():
            return None

        file_mtime = file_path.stat().st_mtime

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    'SELECT metadata, last_modified FROM test_files WHERE path = ?',
                    (str(file_path),)
                )
                result = cursor.fetchone()

                if result and result[1] >= file_mtime:
                    try:
                        metadata_dict = json.loads(result[0])
                        # Reconstruct TestFileInfo from dict
                        return self._dict_to_test_file_info(metadata_dict)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize cached metadata for {file_path}: {e}")

                return None

            finally:
                conn.close()

    def cache_metadata(self, file_path: Path, metadata: TestFileInfo) -> None:
        """Cache parsed metadata."""
        if not file_path.exists():
            return

        file_mtime = file_path.stat().st_mtime
        metadata_json = json.dumps(asdict(metadata), default=str, ensure_ascii=False)

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO test_files (path, last_modified, metadata)
                    VALUES (?, ?, ?)
                ''', (str(file_path), file_mtime, metadata_json))
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to cache metadata for {file_path}: {e}")
            finally:
                conn.close()

    def record_generation(self, file_count: int, test_count: int,
                         format_type: str, output_path: Optional[Path] = None) -> None:
        """Record a documentation generation event."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    INSERT INTO generation_history (file_count, test_count, format, output_path)
                    VALUES (?, ?, ?, ?)
                ''', (file_count, test_count, format_type, str(output_path) if output_path else None))
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to record generation: {e}")
            finally:
                conn.close()

    def _dict_to_test_file_info(self, data: Dict[str, Any]) -> TestFileInfo:
        """Convert dictionary back to TestFileInfo."""
        # This is a simplified conversion - in practice you'd need proper deserialization
        file_info = TestFileInfo(file_path=Path(data['file_path']))
        file_info.encoding = data.get('encoding', 'utf-8')
        file_info.total_lines = data.get('total_lines', 0)
        file_info.parse_errors = data.get('parse_errors', [])
        file_info.imports = data.get('imports', [])
        file_info.fixtures = data.get('fixtures', [])
        file_info.classes = data.get('classes', [])
        file_info.test_coverage = data.get('test_coverage')

        # Convert test metadata
        for test_data in data.get('tests', []):
            test = TestMetadata(
                name=test_data['name'],
                docstring=test_data.get('docstring'),
                file_path=Path(test_data['file_path']),
                line_number=test_data['line_number'],
                test_type=TestType(test_data['test_type']),
                is_async=test_data.get('is_async', False),
                is_parametrized=test_data.get('is_parametrized', False),
                expected_to_fail=test_data.get('expected_to_fail', False),
                skip_reason=test_data.get('skip_reason'),
                marks=set(test_data.get('marks', [])),
                complexity_score=test_data.get('complexity_score', 1)
            )
            file_info.tests.append(test)

        return file_info


class TestDocumentationGenerator:
    """
    Main test documentation generator.

    Orchestrates parsing of test files, integration with coverage data,
    and generation of documentation in multiple formats.
    """

    def __init__(self,
                 cache_db: Optional[Path] = None,
                 coverage_file: Optional[Path] = None,
                 template_dir: Optional[Path] = None,
                 max_workers: int = 4):
        """
        Initialize documentation generator.

        Args:
            cache_db: Path to SQLite cache database
            coverage_file: Path to coverage.json file
            template_dir: Directory containing custom templates
            max_workers: Maximum number of parallel workers
        """
        self.parser = TestFileParser()
        self.coverage_integrator = CoverageIntegrator(coverage_file)
        self.cache_db = TestDatabase(cache_db) if cache_db else None
        self.max_workers = max_workers

        # Initialize formatters
        self.formatters: Dict[str, BaseFormatter] = {
            'markdown': MarkdownFormatter(template_dir),
            'html': HTMLFormatter(template_dir),
            'json': JSONFormatter(template_dir)
        }

    def add_formatter(self, name: str, formatter: BaseFormatter) -> None:
        """Add a custom formatter."""
        self.formatters[name] = formatter

    def generate_file_documentation(self,
                                  file_path: Union[str, Path],
                                  format_type: str = 'markdown',
                                  include_coverage: bool = True) -> str:
        """
        Generate documentation for a single test file.

        Args:
            file_path: Path to test file
            format_type: Output format ('markdown', 'html', 'json')
            include_coverage: Whether to include coverage data

        Returns:
            Generated documentation as string

        Raises:
            ValueError: If format not supported
            FileNotFoundError: If file doesn't exist
        """
        if format_type not in self.formatters:
            raise ValueError(f"Unsupported format: {format_type}")

        file_path = Path(file_path)

        # Try cache first
        file_info = None
        if self.cache_db:
            file_info = self.cache_db.get_cached_metadata(file_path)

        if not file_info:
            file_info = self.parser.parse_file(file_path)
            if self.cache_db:
                self.cache_db.cache_metadata(file_path, file_info)

        # Add coverage data
        if include_coverage:
            coverage = self.coverage_integrator.get_coverage(file_path)
            if coverage is not None:
                file_info.test_coverage = coverage

        # Generate documentation
        formatter = self.formatters[format_type]
        documentation = formatter.format_test_file(file_info)

        # Record generation
        if self.cache_db:
            self.cache_db.record_generation(1, len(file_info.tests), format_type)

        return documentation

    def generate_suite_documentation(self,
                                   directory: Union[str, Path],
                                   format_type: str = 'markdown',
                                   pattern: str = 'test_*.py',
                                   title: str = 'Test Suite Documentation',
                                   include_coverage: bool = True,
                                   parallel: bool = True) -> str:
        """
        Generate documentation for an entire test suite.

        Args:
            directory: Directory containing test files
            format_type: Output format ('markdown', 'html', 'json')
            pattern: File pattern to match
            title: Title for the documentation
            include_coverage: Whether to include coverage data
            parallel: Whether to use parallel processing

        Returns:
            Generated documentation as string

        Raises:
            ValueError: If format not supported or directory invalid
        """
        if format_type not in self.formatters:
            raise ValueError(f"Unsupported format: {format_type}")

        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        # Find all test files
        test_files = list(directory.rglob(pattern))
        if not test_files:
            logger.warning(f"No test files found matching pattern: {pattern}")
            return self.formatters[format_type].format_test_suite([], title)

        # Parse files
        if parallel and len(test_files) > 1:
            file_infos = self._parse_files_parallel(test_files, include_coverage)
        else:
            file_infos = self._parse_files_sequential(test_files, include_coverage)

        # Filter out failed parses
        valid_file_infos = [fi for fi in file_infos if fi is not None]

        # Generate documentation
        formatter = self.formatters[format_type]
        documentation = formatter.format_test_suite(valid_file_infos, title)

        # Record generation
        if self.cache_db:
            total_tests = sum(len(fi.tests) for fi in valid_file_infos)
            self.cache_db.record_generation(len(valid_file_infos), total_tests, format_type)

        return documentation

    def generate_to_file(self,
                        source: Union[str, Path],
                        output_path: Union[str, Path],
                        format_type: str = 'markdown',
                        **kwargs) -> None:
        """
        Generate documentation and save to file.

        Args:
            source: Source file or directory
            output_path: Output file path
            format_type: Output format
            **kwargs: Additional arguments for generation
        """
        source = Path(source)
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate documentation
        if source.is_file():
            content = self.generate_file_documentation(source, format_type, **kwargs)
        else:
            content = self.generate_suite_documentation(source, format_type, **kwargs)

        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Documentation generated: {output_path}")

            # Record generation with output path
            if self.cache_db:
                self.cache_db.record_generation(
                    1 if source.is_file() else len(list(source.rglob('test_*.py'))),
                    0,  # Test count will be recorded separately
                    format_type,
                    output_path
                )

        except Exception as e:
            logger.error(f"Failed to write documentation to {output_path}: {e}")
            raise

    def _parse_files_parallel(self, files: List[Path], include_coverage: bool) -> List[Optional[TestFileInfo]]:
        """Parse files in parallel."""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit parsing tasks
            future_to_file = {
                executor.submit(self._parse_single_file, file_path, include_coverage): file_path
                for file_path in files
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")
                    results.append(None)

        return results

    def _parse_files_sequential(self, files: List[Path], include_coverage: bool) -> List[Optional[TestFileInfo]]:
        """Parse files sequentially."""
        results = []
        for file_path in files:
            try:
                result = self._parse_single_file(file_path, include_coverage)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                results.append(None)
        return results

    def _parse_single_file(self, file_path: Path, include_coverage: bool) -> Optional[TestFileInfo]:
        """Parse a single file with caching and coverage."""
        try:
            # Try cache first
            file_info = None
            if self.cache_db:
                file_info = self.cache_db.get_cached_metadata(file_path)

            if not file_info:
                file_info = self.parser.parse_file(file_path)
                if self.cache_db:
                    self.cache_db.cache_metadata(file_path, file_info)

            # Add coverage data
            if include_coverage:
                coverage = self.coverage_integrator.get_coverage(file_path)
                if coverage is not None:
                    file_info.test_coverage = coverage

            return file_info

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            # Return error info
            return TestFileInfo(
                file_path=file_path,
                parse_errors=[f"Parse failed: {e}"]
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics from cache database."""
        if not self.cache_db:
            return {}

        with self.cache_db.lock:
            conn = sqlite3.connect(self.cache_db.db_path)
            try:
                # Get cached files count
                cursor = conn.execute('SELECT COUNT(*) FROM test_files')
                cached_files = cursor.fetchone()[0]

                # Get generation history
                cursor = conn.execute('''
                    SELECT COUNT(*) as generations,
                           SUM(file_count) as total_files,
                           SUM(test_count) as total_tests,
                           MAX(generated_at) as last_generation
                    FROM generation_history
                ''')
                gen_stats = cursor.fetchone()

                return {
                    'cached_files': cached_files,
                    'total_generations': gen_stats[0] or 0,
                    'total_files_processed': gen_stats[1] or 0,
                    'total_tests_processed': gen_stats[2] or 0,
                    'last_generation': gen_stats[3]
                }

            finally:
                conn.close()

    def clear_cache(self) -> None:
        """Clear cached data."""
        if not self.cache_db:
            return

        with self.cache_db.lock:
            conn = sqlite3.connect(self.cache_db.db_path)
            try:
                conn.execute('DELETE FROM test_files')
                conn.commit()
                logger.info("Cache cleared")
            finally:
                conn.close()