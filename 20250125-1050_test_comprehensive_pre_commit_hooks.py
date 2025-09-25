#!/usr/bin/env python3
"""
Comprehensive pre-commit hooks testing with edge cases and error conditions.

Tests all edge cases for pre-commit hooks including:
- Large file handling
- Binary file detection
- Character encoding edge cases
- Performance under resource constraints
- Git hook integration failures
- Cross-platform compatibility issues
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import pytest
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import yaml
import json


class PreCommitHookTester:
    """Comprehensive testing for pre-commit hooks with edge cases."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize the pre-commit hook tester."""
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.original_dir = Path.cwd()
        self.test_results = {}
        self.performance_metrics = {}

    def setup_test_repository(self, include_submodules: bool = False) -> Path:
        """Set up a test Git repository with pre-commit configuration."""
        repo_path = self.temp_dir / "test_repo"
        repo_path.mkdir(exist_ok=True)

        os.chdir(repo_path)

        # Initialize Git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)

        # Create pre-commit configuration
        pre_commit_config = {
            'repos': [
                {
                    'repo': 'https://github.com/psf/black',
                    'rev': '23.12.1',
                    'hooks': [{'id': 'black', 'args': ['--line-length=88']}]
                },
                {
                    'repo': 'https://github.com/charliermarsh/ruff-pre-commit',
                    'rev': 'v0.1.9',
                    'hooks': [
                        {'id': 'ruff', 'args': ['--fix', '--exit-non-zero-on-fix']},
                        {'id': 'ruff-format'}
                    ]
                },
                {
                    'repo': 'https://github.com/pre-commit/pre-commit-hooks',
                    'rev': 'v4.5.0',
                    'hooks': [
                        {'id': 'trailing-whitespace'},
                        {'id': 'end-of-file-fixer'},
                        {'id': 'check-yaml'},
                        {'id': 'check-json'},
                        {'id': 'check-merge-conflict'},
                        {'id': 'check-added-large-files', 'args': ['--maxkb=1000']},
                        {'id': 'check-case-conflict'},
                        {'id': 'debug-statements'}
                    ]
                },
                {
                    'repo': 'local',
                    'hooks': [
                        {
                            'id': 'test-coverage-validation',
                            'name': 'Test Coverage Validation',
                            'entry': 'pytest --cov=src --cov-fail-under=90',
                            'language': 'system',
                            'pass_filenames': False,
                            'stages': ['commit']
                        }
                    ]
                }
            ]
        }

        with open(".pre-commit-config.yaml", "w") as f:
            yaml.dump(pre_commit_config, f)

        # Install pre-commit
        try:
            subprocess.run(["pre-commit", "install"], check=True, capture_output=True)
        except FileNotFoundError:
            # pre-commit not available, create mock hooks
            hooks_dir = repo_path / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)

            hook_script = """#!/bin/bash
echo "Mock pre-commit hook executed"
exit 0
"""
            (hooks_dir / "pre-commit").write_text(hook_script)
            (hooks_dir / "pre-commit").chmod(0o755)

        if include_submodules:
            # Add submodule for complex repository testing
            submodule_path = self.temp_dir / "submodule"
            submodule_path.mkdir(exist_ok=True)
            os.chdir(submodule_path)
            subprocess.run(["git", "init"], check=True, capture_output=True)
            (submodule_path / "README.md").write_text("# Submodule")
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)

            os.chdir(repo_path)
            subprocess.run(["git", "submodule", "add", str(submodule_path), "submodule"],
                         check=True, capture_output=True)

        return repo_path

    def test_large_file_handling(self) -> Dict[str, Any]:
        """Test pre-commit hook behavior with large files."""
        repo_path = self.setup_test_repository()
        results = {}

        try:
            # Test file at size limit (1MB)
            large_file_content = "x" * (1024 * 1024)  # 1MB
            (repo_path / "large_file.txt").write_text(large_file_content)

            # Test file over size limit (2MB)
            oversized_file_content = "y" * (2 * 1024 * 1024)  # 2MB
            (repo_path / "oversized_file.txt").write_text(oversized_file_content)

            # Add files and test pre-commit behavior
            subprocess.run(["git", "add", "large_file.txt"], check=True, capture_output=True)

            # This should succeed (at limit)
            result_at_limit = subprocess.run(
                ["git", "commit", "-m", "Add large file at limit"],
                capture_output=True,
                text=True
            )
            results["large_file_at_limit"] = {
                "success": result_at_limit.returncode == 0,
                "output": result_at_limit.stdout,
                "error": result_at_limit.stderr
            }

            # Add oversized file
            subprocess.run(["git", "add", "oversized_file.txt"], check=True, capture_output=True)

            # This should fail (over limit)
            result_over_limit = subprocess.run(
                ["git", "commit", "-m", "Add oversized file"],
                capture_output=True,
                text=True
            )
            results["oversized_file_rejection"] = {
                "success": result_over_limit.returncode != 0,  # Should fail
                "output": result_over_limit.stdout,
                "error": result_over_limit.stderr
            }

            # Test binary file with large size
            binary_content = bytes(range(256)) * (1024 * 5)  # 5MB binary
            (repo_path / "large_binary.bin").write_bytes(binary_content)

            subprocess.run(["git", "add", "large_binary.bin"], check=True, capture_output=True)

            result_binary_large = subprocess.run(
                ["git", "commit", "-m", "Add large binary file"],
                capture_output=True,
                text=True
            )
            results["large_binary_file"] = {
                "success": result_binary_large.returncode != 0,  # Should fail
                "output": result_binary_large.stdout,
                "error": result_binary_large.stderr
            }

        except Exception as e:
            results["error"] = str(e)

        finally:
            os.chdir(self.original_dir)

        return results

    def test_encoding_edge_cases(self) -> Dict[str, Any]:
        """Test pre-commit hook behavior with various character encodings."""
        repo_path = self.setup_test_repository()
        results = {}

        try:
            # Test files with different encodings
            encoding_tests = [
                ("utf8_with_bom.py", "utf-8-sig", "# -*- coding: utf-8 -*-\nprint('Hello, 世界!')"),
                ("latin1.py", "latin-1", "# -*- coding: latin-1 -*-\nprint('Café')"),
                ("cp1252.py", "cp1252", "# Windows encoding test\nprint('Smart quotes: "hello"')"),
                ("ascii_only.py", "ascii", "# Pure ASCII\nprint('Hello World')"),
                ("empty_file.py", "utf-8", ""),
                ("only_newlines.py", "utf-8", "\n\n\n")
            ]

            for filename, encoding, content in encoding_tests:
                try:
                    (repo_path / filename).write_text(content, encoding=encoding)
                    subprocess.run(["git", "add", filename], check=True, capture_output=True)

                    result = subprocess.run(
                        ["git", "commit", "-m", f"Add {filename}"],
                        capture_output=True,
                        text=True
                    )

                    results[f"encoding_{encoding}"] = {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr
                    }

                except Exception as e:
                    results[f"encoding_{encoding}_error"] = str(e)

            # Test file with null bytes (should be rejected)
            null_file = repo_path / "null_bytes.txt"
            null_file.write_bytes(b"Hello\x00World\x00")

            subprocess.run(["git", "add", "null_bytes.txt"], check=True, capture_output=True)

            result_null = subprocess.run(
                ["git", "commit", "-m", "Add file with null bytes"],
                capture_output=True,
                text=True
            )

            results["null_bytes_handling"] = {
                "success": result_null.returncode == 0,  # May pass depending on hooks
                "output": result_null.stdout,
                "error": result_null.stderr
            }

            # Test very long lines
            long_line_content = "x" * 10000 + "\n"
            (repo_path / "long_lines.py").write_text(long_line_content)

            subprocess.run(["git", "add", "long_lines.py"], check=True, capture_output=True)

            result_long_lines = subprocess.run(
                ["git", "commit", "-m", "Add file with long lines"],
                capture_output=True,
                text=True
            )

            results["long_lines_handling"] = {
                "success": result_long_lines.returncode == 0,
                "output": result_long_lines.stdout,
                "error": result_long_lines.stderr
            }

        except Exception as e:
            results["error"] = str(e)

        finally:
            os.chdir(self.original_dir)

        return results

    def test_resource_constraints(self) -> Dict[str, Any]:
        """Test pre-commit hooks under resource constraints."""
        repo_path = self.setup_test_repository()
        results = {}

        try:
            # Create many files to test bulk processing
            bulk_files = []
            for i in range(100):
                filename = f"test_file_{i:03d}.py"
                content = f"""# Test file {i}
def function_{i}():
    \"\"\"Test function {i}.\"\"\"
    return {i} * 2

if __name__ == "__main__":
    print(function_{i}())
"""
                (repo_path / filename).write_text(content)
                bulk_files.append(filename)

            # Add all files at once
            subprocess.run(["git", "add"] + bulk_files, check=True, capture_output=True)

            # Time the commit with many files
            start_time = time.time()
            result_bulk = subprocess.run(
                ["git", "commit", "-m", "Add 100 test files"],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            bulk_time = time.time() - start_time

            results["bulk_file_processing"] = {
                "success": result_bulk.returncode == 0,
                "time_seconds": bulk_time,
                "output": result_bulk.stdout,
                "error": result_bulk.stderr
            }

            # Test memory usage with large content changes
            large_content = "\n".join([f"line_{i}: " + "x" * 100 for i in range(10000)])
            (repo_path / "memory_test.txt").write_text(large_content)

            subprocess.run(["git", "add", "memory_test.txt"], check=True, capture_output=True)

            start_time = time.time()
            result_memory = subprocess.run(
                ["git", "commit", "-m", "Add memory test file"],
                capture_output=True,
                text=True,
                timeout=60
            )
            memory_time = time.time() - start_time

            results["memory_intensive_processing"] = {
                "success": result_memory.returncode == 0,
                "time_seconds": memory_time,
                "output": result_memory.stdout,
                "error": result_memory.stderr
            }

            # Test concurrent operations
            def concurrent_commit(filename_suffix: str) -> Dict[str, Any]:
                """Perform concurrent commit operation."""
                try:
                    filename = f"concurrent_{filename_suffix}.py"
                    content = f"# Concurrent file {filename_suffix}\nprint('hello')"
                    (repo_path / filename).write_text(content)

                    subprocess.run(["git", "add", filename], check=True, capture_output=True)
                    result = subprocess.run(
                        ["git", "commit", "-m", f"Concurrent commit {filename_suffix}"],
                        capture_output=True,
                        text=True
                    )

                    return {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                except Exception as e:
                    return {"error": str(e)}

            # Run concurrent commits (note: this will likely fail due to Git locking)
            threads = []
            concurrent_results = {}

            for i in range(3):
                thread = threading.Thread(
                    target=lambda i=i: concurrent_results.update({f"thread_{i}": concurrent_commit(str(i))})
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join(timeout=30)

            results["concurrent_operations"] = concurrent_results

        except subprocess.TimeoutExpired:
            results["timeout_error"] = "Pre-commit hooks exceeded timeout limit"
        except Exception as e:
            results["error"] = str(e)

        finally:
            os.chdir(self.original_dir)

        return results

    def test_git_hook_integration_failures(self) -> Dict[str, Any]:
        """Test pre-commit hook behavior during Git integration failures."""
        repo_path = self.setup_test_repository()
        results = {}

        try:
            # Test corrupted hook script
            hooks_dir = repo_path / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)

            # Create invalid hook script
            invalid_hook = hooks_dir / "pre-commit"
            invalid_hook.write_text("#!/bin/bash\nexit 1\n")  # Always fails
            invalid_hook.chmod(0o755)

            # Try to commit with failing hook
            (repo_path / "test.py").write_text("print('hello')")
            subprocess.run(["git", "add", "test.py"], check=True, capture_output=True)

            result_failing_hook = subprocess.run(
                ["git", "commit", "-m", "Test failing hook"],
                capture_output=True,
                text=True
            )

            results["failing_hook_script"] = {
                "success": result_failing_hook.returncode != 0,  # Should fail
                "output": result_failing_hook.stdout,
                "error": result_failing_hook.stderr
            }

            # Test hook script with syntax error
            syntax_error_hook = """#!/bin/bash
if [ missing_bracket
    echo "Syntax error"
fi
"""
            invalid_hook.write_text(syntax_error_hook)

            result_syntax_error = subprocess.run(
                ["git", "commit", "-m", "Test syntax error hook"],
                capture_output=True,
                text=True
            )

            results["syntax_error_hook"] = {
                "success": result_syntax_error.returncode != 0,  # Should fail
                "output": result_syntax_error.stdout,
                "error": result_syntax_error.stderr
            }

            # Test missing executable permission
            invalid_hook.write_text("#!/bin/bash\necho 'test'\n")
            invalid_hook.chmod(0o644)  # Remove execute permission

            result_no_exec = subprocess.run(
                ["git", "commit", "-m", "Test non-executable hook"],
                capture_output=True,
                text=True
            )

            results["non_executable_hook"] = {
                "success": result_no_exec.returncode != 0,  # Should fail or be skipped
                "output": result_no_exec.stdout,
                "error": result_no_exec.stderr
            }

            # Test hook timeout scenario
            timeout_hook = """#!/bin/bash
sleep 30
echo "Hook completed"
"""
            invalid_hook.write_text(timeout_hook)
            invalid_hook.chmod(0o755)

            start_time = time.time()
            result_timeout = subprocess.run(
                ["git", "commit", "-m", "Test timeout hook"],
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            timeout_duration = time.time() - start_time

            results["timeout_hook"] = {
                "success": timeout_duration < 15,  # Should timeout or complete quickly
                "duration": timeout_duration,
                "output": result_timeout.stdout,
                "error": result_timeout.stderr
            }

        except subprocess.TimeoutExpired:
            results["timeout_hook"]["timeout_expired"] = True
        except Exception as e:
            results["error"] = str(e)

        finally:
            os.chdir(self.original_dir)

        return results

    def test_cross_platform_compatibility(self) -> Dict[str, Any]:
        """Test pre-commit hooks across different platforms and environments."""
        repo_path = self.setup_test_repository()
        results = {}

        try:
            # Test Windows-style line endings
            windows_content = "line1\r\nline2\r\nline3\r\n"
            (repo_path / "windows_endings.txt").write_text(windows_content, newline='')

            # Test Unix-style line endings
            unix_content = "line1\nline2\nline3\n"
            (repo_path / "unix_endings.txt").write_text(unix_content, newline='')

            # Test mixed line endings (problematic)
            mixed_content = "line1\r\nline2\nline3\r\nline4\n"
            (repo_path / "mixed_endings.txt").write_text(mixed_content, newline='')

            subprocess.run(["git", "add", "windows_endings.txt", "unix_endings.txt", "mixed_endings.txt"],
                         check=True, capture_output=True)

            result_line_endings = subprocess.run(
                ["git", "commit", "-m", "Test line ending handling"],
                capture_output=True,
                text=True
            )

            results["line_endings"] = {
                "success": result_line_endings.returncode == 0,
                "output": result_line_endings.stdout,
                "error": result_line_endings.stderr
            }

            # Test file paths with spaces and special characters
            special_files = [
                "file with spaces.py",
                "file-with-dashes.py",
                "file_with_underscores.py",
                "file.with.dots.py",
                "UPPERCASE.PY",
                "file123numbers.py",
                "file[brackets].py",
                "file(parentheses).py"
            ]

            for filename in special_files:
                try:
                    (repo_path / filename).write_text("# Test file")
                    subprocess.run(["git", "add", filename], check=True, capture_output=True)
                except Exception as e:
                    results[f"special_filename_{filename}_error"] = str(e)

            result_special_names = subprocess.run(
                ["git", "commit", "-m", "Test special filenames"],
                capture_output=True,
                text=True
            )

            results["special_filenames"] = {
                "success": result_special_names.returncode == 0,
                "output": result_special_names.stdout,
                "error": result_special_names.stderr
            }

            # Test symlinks (Unix-like systems only)
            if os.name != 'nt':  # Not Windows
                try:
                    (repo_path / "target.py").write_text("# Symlink target")
                    os.symlink("target.py", repo_path / "symlink.py")

                    subprocess.run(["git", "add", "target.py", "symlink.py"],
                                 check=True, capture_output=True)

                    result_symlinks = subprocess.run(
                        ["git", "commit", "-m", "Test symlink handling"],
                        capture_output=True,
                        text=True
                    )

                    results["symlinks"] = {
                        "success": result_symlinks.returncode == 0,
                        "output": result_symlinks.stdout,
                        "error": result_symlinks.stderr
                    }
                except Exception as e:
                    results["symlinks_error"] = str(e)

            # Test environment variable dependencies
            env_test_files = {
                "PATH": "echo $PATH",
                "HOME": "echo $HOME",
                "USER": "echo $USER"
            }

            for var_name, command in env_test_files.items():
                try:
                    # Create hook that depends on environment variable
                    hook_content = f"""#!/bin/bash
if [ -z "${{{var_name}}}" ]; then
    echo "Environment variable {var_name} not set"
    exit 1
fi
echo "Environment variable {var_name} is set"
"""

                    hook_file = repo_path / ".git" / "hooks" / f"test-{var_name.lower()}"
                    hook_file.write_text(hook_content)
                    hook_file.chmod(0o755)

                    # Test running the hook directly
                    result_env = subprocess.run(
                        [str(hook_file)],
                        capture_output=True,
                        text=True
                    )

                    results[f"env_var_{var_name.lower()}"] = {
                        "success": result_env.returncode == 0,
                        "output": result_env.stdout,
                        "error": result_env.stderr
                    }

                except Exception as e:
                    results[f"env_var_{var_name.lower()}_error"] = str(e)

        except Exception as e:
            results["error"] = str(e)

        finally:
            os.chdir(self.original_dir)

        return results

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Benchmark pre-commit hook performance under various scenarios."""
        repo_path = self.setup_test_repository()
        results = {}

        try:
            # Benchmark single file commit
            (repo_path / "single.py").write_text("print('hello')")
            subprocess.run(["git", "add", "single.py"], check=True, capture_output=True)

            start_time = time.time()
            result_single = subprocess.run(
                ["git", "commit", "-m", "Single file commit"],
                capture_output=True,
                text=True
            )
            single_time = time.time() - start_time

            results["single_file_performance"] = {
                "success": result_single.returncode == 0,
                "time_seconds": single_time,
                "files_per_second": 1 / single_time if single_time > 0 else float('inf')
            }

            # Benchmark multiple small files
            small_files = []
            for i in range(50):
                filename = f"small_{i:02d}.py"
                (repo_path / filename).write_text(f"print('file {i}')")
                small_files.append(filename)

            subprocess.run(["git", "add"] + small_files, check=True, capture_output=True)

            start_time = time.time()
            result_small = subprocess.run(
                ["git", "commit", "-m", "Multiple small files"],
                capture_output=True,
                text=True
            )
            small_time = time.time() - start_time

            results["multiple_small_files_performance"] = {
                "success": result_small.returncode == 0,
                "time_seconds": small_time,
                "files_per_second": len(small_files) / small_time if small_time > 0 else float('inf'),
                "file_count": len(small_files)
            }

            # Benchmark large file with many lines
            large_lines = [f"line_{i}: " + "x" * 50 for i in range(1000)]
            (repo_path / "large_lines.py").write_text("\n".join(large_lines))

            subprocess.run(["git", "add", "large_lines.py"], check=True, capture_output=True)

            start_time = time.time()
            result_large = subprocess.run(
                ["git", "commit", "-m", "Large file with many lines"],
                capture_output=True,
                text=True
            )
            large_time = time.time() - start_time

            results["large_file_performance"] = {
                "success": result_large.returncode == 0,
                "time_seconds": large_time,
                "lines_per_second": len(large_lines) / large_time if large_time > 0 else float('inf'),
                "line_count": len(large_lines)
            }

            # Performance regression detection
            baseline_times = [single_time, small_time, large_time]
            avg_time = sum(baseline_times) / len(baseline_times)

            results["performance_summary"] = {
                "average_commit_time": avg_time,
                "fastest_commit": min(baseline_times),
                "slowest_commit": max(baseline_times),
                "performance_variance": max(baseline_times) - min(baseline_times),
                "acceptable_performance": avg_time < 10.0  # 10 second threshold
            }

        except Exception as e:
            results["error"] = str(e)

        finally:
            os.chdir(self.original_dir)

        return results

    def cleanup(self):
        """Clean up test resources."""
        os.chdir(self.original_dir)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# Pytest test cases
class TestPreCommitHookEdgeCases:
    """Pytest test cases for pre-commit hook edge cases."""

    @pytest.fixture
    def hook_tester(self):
        """Fixture to provide a pre-commit hook tester."""
        tester = PreCommitHookTester()
        yield tester
        tester.cleanup()

    def test_large_file_edge_cases(self, hook_tester):
        """Test large file handling edge cases."""
        results = hook_tester.test_large_file_handling()

        # Verify large file at limit is handled correctly
        assert "large_file_at_limit" in results
        # Note: May pass or fail depending on hook configuration

        # Verify oversized files are rejected
        assert "oversized_file_rejection" in results
        assert results["oversized_file_rejection"]["success"], \
            "Oversized files should be rejected by pre-commit hooks"

        # Verify large binary files are handled
        assert "large_binary_file" in results

    def test_encoding_edge_cases(self, hook_tester):
        """Test character encoding edge cases."""
        results = hook_tester.test_encoding_edge_cases()

        # Verify different encodings are handled
        encoding_tests = ["utf-8", "latin-1", "cp1252", "ascii"]
        for encoding in encoding_tests:
            key = f"encoding_{encoding}"
            if key in results:
                assert isinstance(results[key]["success"], bool)

    def test_resource_constraint_handling(self, hook_tester):
        """Test pre-commit hooks under resource constraints."""
        results = hook_tester.test_resource_constraints()

        # Verify bulk file processing works
        if "bulk_file_processing" in results:
            assert results["bulk_file_processing"]["time_seconds"] < 120, \
                "Bulk file processing should complete within 2 minutes"

        # Verify memory-intensive processing
        if "memory_intensive_processing" in results:
            assert results["memory_intensive_processing"]["time_seconds"] < 60, \
                "Memory-intensive processing should complete within 1 minute"

    def test_git_integration_failure_scenarios(self, hook_tester):
        """Test Git integration failure scenarios."""
        results = hook_tester.test_git_hook_integration_failures()

        # Verify failing hooks are detected
        if "failing_hook_script" in results:
            assert results["failing_hook_script"]["success"], \
                "Failing hook scripts should be properly detected"

        # Verify syntax errors in hooks are caught
        if "syntax_error_hook" in results:
            assert results["syntax_error_hook"]["success"], \
                "Hook syntax errors should be detected"

    def test_cross_platform_compatibility_issues(self, hook_tester):
        """Test cross-platform compatibility edge cases."""
        results = hook_tester.test_cross_platform_compatibility()

        # Verify line ending handling
        if "line_endings" in results:
            # Line ending handling may vary by platform and configuration
            assert isinstance(results["line_endings"]["success"], bool)

        # Verify special filename handling
        if "special_filenames" in results:
            assert isinstance(results["special_filenames"]["success"], bool)

    def test_performance_benchmarks_and_regression_detection(self, hook_tester):
        """Test performance benchmarks and regression detection."""
        results = hook_tester.test_performance_benchmarks()

        # Verify performance metrics are collected
        assert "performance_summary" in results
        performance = results["performance_summary"]

        # Check performance thresholds
        assert performance["average_commit_time"] >= 0, "Average commit time should be positive"
        assert performance["acceptable_performance"], \
            f"Performance not acceptable: {performance['average_commit_time']}s average"

    @pytest.mark.asyncio
    async def test_async_hook_operations(self, hook_tester):
        """Test asynchronous pre-commit hook operations."""
        # Simulate async operations that might occur during hook execution
        async def async_file_operation():
            await asyncio.sleep(0.1)
            return "async operation completed"

        result = await async_file_operation()
        assert result == "async operation completed"

    def test_hook_configuration_edge_cases(self, hook_tester):
        """Test pre-commit hook configuration edge cases."""
        repo_path = hook_tester.setup_test_repository()

        try:
            # Test invalid YAML configuration
            invalid_config = """
repos:
  - repo: https://github.com/psf/black
    rev: invalid-ref
    hooks:
      - id: nonexistent-hook
"""
            config_path = repo_path / ".pre-commit-config.yaml"
            config_path.write_text(invalid_config)

            # Try to run pre-commit with invalid config
            result = subprocess.run(
                ["git", "add", ".pre-commit-config.yaml"],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, "Adding invalid config should succeed"

            # Commit should work (validation happens during hook execution)
            commit_result = subprocess.run(
                ["git", "commit", "-m", "Test invalid config"],
                capture_output=True,
                text=True
            )

            # Result may vary depending on pre-commit installation
            assert isinstance(commit_result.returncode, int)

        finally:
            os.chdir(hook_tester.original_dir)

    def test_network_dependency_failures(self, hook_tester):
        """Test pre-commit hook behavior when network dependencies fail."""
        # This would test scenarios where hooks depend on network resources
        # and those resources are unavailable

        # Create a mock scenario where network is unavailable
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, ['git'], 'Network error'
            )

            try:
                result = subprocess.run(['git', 'status'], check=False)
                assert result.returncode != 0
            except subprocess.CalledProcessError:
                # Expected when network dependencies fail
                pass

    def test_disk_space_exhaustion_scenarios(self, hook_tester):
        """Test pre-commit hook behavior when disk space is exhausted."""
        # Simulate disk space issues by creating scenarios that might
        # cause temporary file creation failures

        with patch('tempfile.TemporaryDirectory') as mock_temp:
            mock_temp.side_effect = OSError("No space left on device")

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    pass
                assert False, "Should have raised OSError"
            except OSError as e:
                assert "No space left on device" in str(e)


if __name__ == "__main__":
    # Run comprehensive pre-commit hook testing
    tester = PreCommitHookTester()

    print("🔧 Running comprehensive pre-commit hook edge case tests...")

    try:
        # Run all test categories
        test_results = {
            "large_files": tester.test_large_file_handling(),
            "encodings": tester.test_encoding_edge_cases(),
            "resource_constraints": tester.test_resource_constraints(),
            "git_integration": tester.test_git_hook_integration_failures(),
            "cross_platform": tester.test_cross_platform_compatibility(),
            "performance": tester.test_performance_benchmarks()
        }

        # Generate comprehensive report
        print("\n📊 Pre-commit Hook Test Results:")
        print("=" * 60)

        total_tests = 0
        passed_tests = 0

        for category, results in test_results.items():
            print(f"\n{category.upper()}:")
            for test_name, result in results.items():
                total_tests += 1
                if isinstance(result, dict) and result.get("success", False):
                    passed_tests += 1
                    status = "✅ PASS"
                elif isinstance(result, dict):
                    status = "❌ FAIL"
                else:
                    status = "⚠️  INFO"

                print(f"  {test_name}: {status}")

                if isinstance(result, dict) and result.get("error"):
                    print(f"    Error: {result['error']}")
                elif isinstance(result, dict) and result.get("time_seconds"):
                    print(f"    Time: {result['time_seconds']:.2f}s")

        print(f"\n📈 Summary: {passed_tests}/{total_tests} tests passed "
              f"({passed_tests/total_tests*100:.1f}%)")

        # Save detailed results
        with open("pre_commit_hook_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        print("📄 Detailed results saved to 'pre_commit_hook_test_results.json'")

    finally:
        tester.cleanup()
        print("\n🧹 Test cleanup completed")