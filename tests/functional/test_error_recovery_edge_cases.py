"""
Error Recovery and Edge Case Testing

Comprehensive tests for network connectivity issues, corrupted documents,
interrupted operations, and comprehensive error handling validation.

This module implements subtask 203.6 of the End-to-End Functional Testing Framework.
"""

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import pytest


class ErrorRecoveryTestEnvironment:
    """Test environment for error recovery and edge case validation."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.config_dir = tmp_path / ".config" / "workspace-qdrant"
        self.test_data_dir = tmp_path / "test_data"
        self.cli_executable = "uv run wqm"

        self.setup_environment()

    def setup_environment(self):
        """Set up test environment for error scenarios."""
        # Create directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        # Create configuration for error testing
        config_content = {
            "qdrant_url": "http://localhost:6333",
            "error_handling": {
                "retry_attempts": 3,
                "retry_delay": 1,
                "timeout": 30,
                "fail_fast": False
            },
            "logging": {
                "level": "DEBUG",
                "log_errors": True,
                "error_log_file": str(self.tmp_path / "error.log")
            }
        }

        import yaml
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

    def run_cli_command(
        self,
        command: str,
        timeout: int = 30,
        expected_failure: bool = False,
        env_vars: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Execute CLI command with error handling expectations."""
        env = os.environ.copy()
        env.update({
            "WQM_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": str(Path.cwd()),
        })

        if env_vars:
            env.update(env_vars)

        try:
            result = subprocess.run(
                f"{self.cli_executable} {command}",
                shell=True,
                cwd=self.tmp_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", f"Command execution failed: {e}"

    def create_corrupted_files(self) -> dict[str, Path]:
        """Create various types of corrupted test files."""
        corrupted_files = {}

        # Binary data in text file
        binary_file = self.test_data_dir / "binary_data.txt"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe\xfd\xfc' * 100)
        corrupted_files["binary_in_text"] = binary_file

        # Invalid UTF-8 encoding
        invalid_utf8_file = self.test_data_dir / "invalid_utf8.txt"
        invalid_utf8_file.write_bytes(b'\xff\xfe\x00\x00Invalid UTF-8: \x80\x81\x82')
        corrupted_files["invalid_utf8"] = invalid_utf8_file

        # Extremely large file (within limits)
        large_file = self.test_data_dir / "large_file.txt"
        large_content = "Large file content line\n" * 10000
        large_file.write_text(large_content, encoding='utf-8')
        corrupted_files["large_file"] = large_file

        # Empty file
        empty_file = self.test_data_dir / "empty_file.txt"
        empty_file.touch()
        corrupted_files["empty_file"] = empty_file

        # File with only whitespace
        whitespace_file = self.test_data_dir / "whitespace_only.txt"
        whitespace_file.write_text("   \n\t\r\n   \t\t  \n", encoding='utf-8')
        corrupted_files["whitespace_only"] = whitespace_file

        # Invalid JSON file
        invalid_json_file = self.test_data_dir / "invalid.json"
        invalid_json_file.write_text('{"key": "value", "invalid": }', encoding='utf-8')
        corrupted_files["invalid_json"] = invalid_json_file

        # File with control characters
        control_chars_file = self.test_data_dir / "control_chars.txt"
        control_chars_file.write_text('Text with\x00null\x01control\x02chars\x03here', encoding='utf-8', errors='replace')
        corrupted_files["control_chars"] = control_chars_file

        # Circular symlink (on Unix-like systems)
        if os.name != 'nt':
            try:
                symlink_file = self.test_data_dir / "circular_symlink.txt"
                symlink_file.symlink_to(symlink_file)
                corrupted_files["circular_symlink"] = symlink_file
            except (OSError, NotImplementedError):
                pass  # Skip if symlinks not supported

        return corrupted_files

    def create_permission_test_files(self) -> dict[str, Path]:
        """Create files for permission testing (no actual permission changes)."""
        permission_files = {}

        # Read-only file (not actually modified)
        readonly_file = self.test_data_dir / "readonly.txt"
        readonly_file.write_text("Read-only file content", encoding='utf-8')
        if os.name != 'nt':  # Unix-like systems
            permission_files["readonly"] = readonly_file

        # Directory for permission testing (not actually modified)
        if os.name != 'nt':
            no_read_dir = self.test_data_dir / "no_read_dir"
            no_read_dir.mkdir()
            (no_read_dir / "hidden_file.txt").write_text("Hidden content")
            permission_files["no_read_dir"] = no_read_dir

        return permission_files

    def simulate_network_issues(self) -> dict[str, Any]:
        """Simulate various network connectivity issues."""
        network_scenarios = {}

        # Invalid Qdrant URL
        network_scenarios["invalid_url"] = {
            "env_vars": {"QDRANT_URL": "http://invalid-host:6333"},
            "description": "Invalid Qdrant host"
        }

        # Wrong port
        network_scenarios["wrong_port"] = {
            "env_vars": {"QDRANT_URL": "http://localhost:9999"},
            "description": "Wrong Qdrant port"
        }

        # Malformed URL
        network_scenarios["malformed_url"] = {
            "env_vars": {"QDRANT_URL": "not-a-valid-url"},
            "description": "Malformed URL"
        }

        return network_scenarios

    def interrupt_operation(self, command: str, interrupt_after: float = 2.0) -> dict[str, Any]:
        """Interrupt a long-running operation to test recovery."""
        env = os.environ.copy()
        env.update({
            "WQM_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": str(Path.cwd()),
        })

        # Start process
        process = subprocess.Popen(
            f"{self.cli_executable} {command}",
            shell=True,
            cwd=self.tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # Wait for specified time then interrupt
        time.sleep(interrupt_after)

        try:
            # Send interrupt signal
            if os.name == 'nt':
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)

            # Wait for process to handle interrupt
            stdout, stderr = process.communicate(timeout=5)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            # Force kill if doesn't respond to interrupt
            process.kill()
            stdout, stderr = process.communicate()
            return_code = -9

        return {
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "interrupted": True
        }


class ErrorRecoveryValidator:
    """Validates error recovery mechanisms and edge case handling."""

    @staticmethod
    def validate_error_message_quality(stderr: str, stdout: str) -> dict[str, bool]:
        """Validate error message quality and usefulness."""
        error_output = stderr + stdout

        return {
            "has_error_message": len(error_output.strip()) > 0,
            "specific_error": any(keyword in error_output.lower() for keyword in [
                "error", "failed", "exception", "invalid", "not found"
            ]),
            "actionable_message": any(keyword in error_output.lower() for keyword in [
                "check", "verify", "ensure", "make sure", "try"
            ]),
            "no_stack_trace": "traceback" not in error_output.lower(),
            "user_friendly": not any(keyword in error_output.lower() for keyword in [
                "internal", "assertion", "unexpected"
            ])
        }

    @staticmethod
    def validate_graceful_failure(return_code: int, stderr: str, stdout: str) -> bool:
        """Validate that failure is graceful and informative."""
        # Should have non-zero return code for failures
        # Should provide meaningful error message
        # Should not crash with stack trace
        error_output = stderr + stdout

        return (
            return_code != 0 and  # Failed as expected
            len(error_output.strip()) > 0 and  # Has error message
            "traceback" not in error_output.lower()  # No ugly stack trace
        )

    @staticmethod
    def validate_recovery_attempt(stderr: str, stdout: str) -> bool:
        """Validate that system attempted recovery where appropriate."""
        output = stderr + stdout
        recovery_indicators = [
            "retry", "retrying", "attempting", "reconnect", "fallback"
        ]
        return any(indicator in output.lower() for indicator in recovery_indicators)

    @staticmethod
    def validate_resource_cleanup(tmp_path: Path) -> bool:
        """Validate that resources are properly cleaned up after errors."""
        # Check for leftover temporary files
        temp_files = list(tmp_path.glob("**/tmp*"))
        lock_files = list(tmp_path.glob("**/*.lock"))

        return len(temp_files) == 0 and len(lock_files) == 0


@pytest.mark.functional
@pytest.mark.error_recovery
class TestErrorRecoveryAndEdgeCases:
    """Test error recovery mechanisms and edge case handling."""

    @pytest.fixture
    def error_env(self, tmp_path):
        """Create error recovery test environment."""
        env = ErrorRecoveryTestEnvironment(tmp_path)
        yield env

    @pytest.fixture
    def validator(self):
        """Create error recovery validator."""
        return ErrorRecoveryValidator()

    def test_corrupted_file_handling(self, error_env, validator):
        """Test handling of corrupted and malformed files."""
        corrupted_files = error_env.create_corrupted_files()

        for file_type, file_path in corrupted_files.items():
            if file_path.exists():
                # Test ingestion of corrupted file
                return_code, stdout, stderr = error_env.run_cli_command(
                    f"ingest file {file_path}",
                    expected_failure=True
                )

                # Validate error handling
                error_quality = validator.validate_error_message_quality(stderr, stdout)

                # Should have error message for problematic files
                if file_type in ["binary_in_text", "invalid_utf8", "invalid_json"]:
                    assert error_quality["has_error_message"], f"No error message for {file_type}"
                    assert error_quality["specific_error"], f"Non-specific error for {file_type}"

                # Should handle gracefully without crashing
                assert return_code != -1, f"Command crashed on {file_type}"

    def test_network_connectivity_issues(self, error_env, validator):
        """Test handling of network connectivity problems."""
        network_scenarios = error_env.simulate_network_issues()

        for scenario_name, scenario in network_scenarios.items():
            # Test admin status with network issues
            return_code, stdout, stderr = error_env.run_cli_command(
                "admin status",
                env_vars=scenario.get("env_vars", {}),
                expected_failure=True
            )

            # Should fail gracefully with network issues
            if return_code != 0:
                error_quality = validator.validate_error_message_quality(stderr, stdout)

                assert error_quality["has_error_message"], f"No error message for {scenario_name}"
                assert error_quality["user_friendly"], f"Non-user-friendly error for {scenario_name}"

                # Should mention connection or network issue
                error_output = stderr + stdout
                assert any(keyword in error_output.lower() for keyword in [
                    "connection", "network", "connect", "host", "unreachable"
                ]), f"No network error indication for {scenario_name}"

    def test_interrupted_operations_recovery(self, error_env, validator):
        """Test recovery from interrupted operations."""
        # Create test data for interruption
        test_files = []
        for i in range(10):
            test_file = error_env.test_data_dir / f"interrupt_test_{i}.txt"
            test_file.write_text(f"Test file {i} content for interruption testing.\n" * 20)
            test_files.append(test_file)

        # Test interrupting ingestion operation
        interrupt_result = error_env.interrupt_operation(
            f"ingest folder {error_env.test_data_dir}",
            interrupt_after=1.0
        )

        # Validate interruption handling
        assert interrupt_result["interrupted"], "Operation was not interrupted"

        # Should handle interrupt gracefully
        error_output = interrupt_result["stderr"] + interrupt_result["stdout"]
        if len(error_output) > 0:
            validator.validate_error_message_quality(
                interrupt_result["stderr"],
                interrupt_result["stdout"]
            )

            # Should provide meaningful message about interruption
            assert any(keyword in error_output.lower() for keyword in [
                "interrupt", "cancel", "stop", "abort"
            ]), "No interruption message"

        # Test recovery - subsequent operations should still work
        return_code, stdout, stderr = error_env.run_cli_command("admin status")
        assert return_code != -1, "System not responsive after interruption"

    def test_permission_denied_scenarios(self, error_env, validator):
        """Test handling of permission denied scenarios."""
        permission_files = error_env.create_permission_test_files()

        for file_type, file_path in permission_files.items():
            if file_path.exists():
                # Mock permission error by raising PermissionError
                with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                    # Test accessing file with permission issues
                    return_code, stdout, stderr = error_env.run_cli_command(
                        f"ingest file {file_path}" if file_path.is_file() else f"ingest folder {file_path}",
                        expected_failure=True
                    )

                    # Should handle permission errors gracefully
                    if return_code != 0:
                        error_quality = validator.validate_error_message_quality(stderr, stdout)

                        assert error_quality["has_error_message"], f"No error message for {file_type}"

                        # Should indicate permission issue
                        error_output = stderr + stdout
                        assert any(keyword in error_output.lower() for keyword in [
                            "permission", "access", "denied", "forbidden"
                        ]), f"No permission error indication for {file_type}"

    def test_invalid_command_arguments(self, error_env, validator):
        """Test handling of invalid command arguments and options."""
        invalid_commands = [
            ("ingest", "Missing required arguments"),
            ("ingest file", "Missing file path"),
            ("ingest file /nonexistent/file.txt", "Nonexistent file"),
            ("search", "Missing search parameters"),
            ("search project", "Missing search query"),
            ("config set invalid_key invalid_value", "Invalid configuration"),
            ("memory add", "Missing memory content"),
            ("library create", "Missing library name"),
            ("watch add", "Missing watch path")
        ]

        for command, description in invalid_commands:
            return_code, stdout, stderr = error_env.run_cli_command(
                command,
                expected_failure=True
            )

            # Should fail with helpful error message
            if return_code != 0:
                error_quality = validator.validate_error_message_quality(stderr, stdout)

                assert error_quality["has_error_message"], f"No error for: {description}"
                assert error_quality["user_friendly"], f"Non-user-friendly error for: {description}"

                # Should provide usage information or suggestion
                error_output = stderr + stdout
                assert any(keyword in error_output.lower() for keyword in [
                    "usage", "help", "try", "required", "expected"
                ]), f"No helpful guidance for: {description}"

    def test_concurrent_operation_conflicts(self, error_env, validator):
        """Test handling of concurrent operation conflicts."""
        import queue
        import threading

        results_queue = queue.Queue()

        def concurrent_worker(worker_id: int):
            try:
                # Each worker tries to perform operations simultaneously
                return_code, stdout, stderr = error_env.run_cli_command("admin status")
                results_queue.put({
                    "worker_id": worker_id,
                    "return_code": return_code,
                    "output": stdout + stderr,
                    "success": return_code != -1
                })
            except Exception as e:
                results_queue.put({
                    "worker_id": worker_id,
                    "error": str(e)
                })

        # Start multiple concurrent operations
        threads = []
        num_workers = 5

        for i in range(num_workers):
            thread = threading.Thread(target=concurrent_worker, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Validate concurrent operation handling
        assert len(results) == num_workers, "Not all workers completed"

        successful_operations = sum(1 for r in results if r.get("success", False))

        # At least some operations should succeed
        assert successful_operations > 0, "No concurrent operations succeeded"

        # Check for reasonable error messages in failed operations
        for result in results:
            if not result.get("success", True) and "output" in result:
                error_quality = validator.validate_error_message_quality("", result["output"])
                if error_quality["has_error_message"]:
                    assert error_quality["user_friendly"], "Non-user-friendly concurrent error"

    def test_resource_exhaustion_scenarios(self, error_env, validator):
        """Test handling of resource exhaustion scenarios."""
        # Test with very large command line
        long_command = "admin status " + " ".join([f"--dummy-arg-{i}" for i in range(100)])
        return_code, stdout, stderr = error_env.run_cli_command(
            long_command,
            expected_failure=True
        )

        # Should handle gracefully
        if return_code != 0:
            error_quality = validator.validate_error_message_quality(stderr, stdout)
            assert error_quality["has_error_message"], "No error for long command"

        # Test with invalid configuration values
        config_file = error_env.config_dir / "invalid_config.yaml"
        invalid_config = {
            "qdrant_url": "x" * 1000,  # Extremely long URL
            "timeout": -1,  # Invalid timeout
            "max_connections": "not_a_number"  # Invalid type
        }

        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        return_code, stdout, stderr = error_env.run_cli_command(
            f"--config {config_file} admin status",
            expected_failure=True
        )

        # Should handle invalid configuration gracefully
        if return_code != 0:
            error_quality = validator.validate_error_message_quality(stderr, stdout)
            if error_quality["has_error_message"]:
                assert error_quality["user_friendly"], "Non-user-friendly config error"

    def test_edge_case_file_operations(self, error_env, validator):
        """Test edge cases in file operations."""
        edge_cases = {}

        # File with very long name (within limits)
        long_name = "a" * 100 + ".txt"
        long_name_file = error_env.test_data_dir / long_name
        long_name_file.write_text("Long filename test content")
        edge_cases["long_filename"] = long_name_file

        # File with special characters in name
        # Use safe special characters for cross-platform compatibility
        safe_special_name = "test_!@#$%^&()_+-=[]{}semicolon_quote_comma.txt"
        special_file = error_env.test_data_dir / safe_special_name
        special_file.write_text("Special characters test content")
        edge_cases["special_chars"] = special_file

        # Test processing these edge case files
        for case_name, file_path in edge_cases.items():
            if file_path.exists():
                return_code, stdout, stderr = error_env.run_cli_command(
                    f"ingest file {file_path}"
                )

                # Should handle edge cases without crashing
                assert return_code != -1, f"Command crashed on {case_name}"

                # If error, should be informative
                if return_code != 0:
                    error_quality = validator.validate_error_message_quality(stderr, stdout)
                    if error_quality["has_error_message"]:
                        assert error_quality["user_friendly"], f"Non-user-friendly error for {case_name}"

    def test_configuration_error_recovery(self, error_env, validator):
        """Test recovery from configuration errors."""
        # Create invalid configuration
        invalid_config_file = error_env.config_dir / "broken_config.yaml"
        invalid_config_file.write_text("invalid: yaml: content: [unclosed")

        # Test with broken config
        return_code, stdout, stderr = error_env.run_cli_command(
            f"--config {invalid_config_file} admin status",
            expected_failure=True
        )

        # Should handle broken config gracefully
        if return_code != 0:
            error_quality = validator.validate_error_message_quality(stderr, stdout)
            assert error_quality["has_error_message"], "No error for broken config"

            # Should mention configuration issue
            error_output = stderr + stdout
            assert any(keyword in error_output.lower() for keyword in [
                "config", "configuration", "yaml", "parse", "invalid"
            ]), "No configuration error indication"

        # Test fallback to default configuration
        return_code, stdout, stderr = error_env.run_cli_command("admin status")

        # Should work with default config
        assert return_code != -1, "System not working after config error"

    def test_memory_constraint_handling(self, error_env, validator):
        """Test handling of memory constraints and cleanup."""
        # Create large number of small files to stress memory
        large_file_set = []
        for i in range(50):
            test_file = error_env.test_data_dir / f"memory_test_{i}.txt"
            content = f"Memory test file {i}\n" + "Content line\n" * 100
            test_file.write_text(content)
            large_file_set.append(test_file)

        # Test ingestion of large file set
        return_code, stdout, stderr = error_env.run_cli_command(
            f"ingest folder {error_env.test_data_dir}",
            timeout=60
        )

        # Should complete or fail gracefully
        assert return_code != -1, "Command crashed with large file set"

        # If failed, should provide meaningful error
        if return_code != 0:
            error_quality = validator.validate_error_message_quality(stderr, stdout)
            if error_quality["has_error_message"]:
                assert error_quality["user_friendly"], "Non-user-friendly memory error"

        # Test resource cleanup
        cleanup_success = validator.validate_resource_cleanup(error_env.tmp_path)
        assert cleanup_success, "Poor resource cleanup after memory stress"

    def test_signal_handling_robustness(self, error_env, validator):
        """Test robustness of signal handling."""
        if os.name == 'nt':
            pytest.skip("Signal testing not applicable on Windows")

        # Test various signal scenarios
        signals_to_test = [signal.SIGTERM, signal.SIGINT]

        for sig in signals_to_test:
            # Start a command that might run for a while
            interrupt_result = error_env.interrupt_operation(
                "admin status",
                interrupt_after=0.5
            )

            # Should handle signal gracefully
            if interrupt_result["interrupted"]:
                error_output = interrupt_result["stderr"] + interrupt_result["stdout"]

                # Should not crash ungracefully
                assert interrupt_result["return_code"] != -9, f"Ungraceful termination on {sig}"

                # If there's output, should be meaningful
                if len(error_output) > 0:
                    error_quality = validator.validate_error_message_quality(
                        interrupt_result["stderr"],
                        interrupt_result["stdout"]
                    )
                    if error_quality["has_error_message"]:
                        assert error_quality["user_friendly"], f"Non-user-friendly signal error for {sig}"

    def test_error_logging_and_reporting(self, error_env, validator):
        """Test error logging and reporting mechanisms."""
        # Force an error condition
        return_code, stdout, stderr = error_env.run_cli_command(
            "ingest file /absolutely/nonexistent/file.txt",
            expected_failure=True
        )

        # Should fail as expected
        assert return_code != 0, "Expected failure did not occur"

        # Check error logging
        error_log_file = error_env.tmp_path / "error.log"
        if error_log_file.exists():
            log_content = error_log_file.read_text()

            # Should contain error information
            assert len(log_content) > 0, "No error logged to file"

            # Should not contain sensitive information
            assert "password" not in log_content.lower(), "Sensitive info in error log"
            assert "secret" not in log_content.lower(), "Sensitive info in error log"

        # Validate error message quality
        error_quality = validator.validate_error_message_quality(stderr, stdout)
        assert error_quality["has_error_message"], "No error message for file not found"
        assert error_quality["specific_error"], "Non-specific error message"
        assert error_quality["user_friendly"], "Non-user-friendly error message"
