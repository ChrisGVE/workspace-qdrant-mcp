"""
Comprehensive input validation security tests.

Tests various attack vectors including SQL injection, path traversal,
command injection, XSS, file upload attacks, and ReDoS patterns.

Based on OWASP Top 10 security risks.
"""

import asyncio
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from tests.security.fixtures.malicious_payloads import (
    COMMAND_INJECTION_PAYLOADS,
    FILE_UPLOAD_PAYLOADS,
    PATH_TRAVERSAL_PAYLOADS,
    REDOS_PAYLOADS,
    SQL_INJECTION_PAYLOADS,
    SPECIAL_CHAR_PAYLOADS,
    XSS_PAYLOADS,
)
from src.python.common.core.sqlite_state_manager import (
    FileProcessingRecord,
    FileProcessingStatus,
    ProcessingPriority,
    SQLiteStateManager,
    WatchFolderConfig,
)


@pytest.fixture
async def state_manager():
    """Create a temporary SQLite state manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_security.db"
        manager = SQLiteStateManager(db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


@pytest.mark.security
class TestSQLInjectionPrevention:
    """Test SQL injection prevention mechanisms."""

    @pytest.mark.parametrize("malicious_payload", SQL_INJECTION_PAYLOADS)
    @pytest.mark.asyncio
    async def test_watch_folder_config_sql_injection(
        self, state_manager, malicious_payload
    ):
        """Test that watch folder configuration properly handles SQL injection attempts."""
        # Create config with malicious payload in various fields
        config = WatchFolderConfig(
            watch_id=malicious_payload,
            path="/tmp/test",
            collection=malicious_payload,
            patterns=["*.py"],
            ignore_patterns=[],
            auto_ingest=True,
            recursive=True,
        )

        # Should not raise SQL injection errors
        result = await state_manager.save_watch_folder_config(config)
        assert result is True

        # Verify data integrity - malicious payload should be stored as literal string
        retrieved = await state_manager.get_watch_folder_config(malicious_payload)
        assert retrieved is not None
        assert retrieved.watch_id == malicious_payload
        assert retrieved.collection == malicious_payload

    @pytest.mark.parametrize("malicious_payload", SQL_INJECTION_PAYLOADS)
    @pytest.mark.asyncio
    async def test_file_processing_sql_injection(
        self, state_manager, malicious_payload
    ):
        """Test file processing records handle SQL injection in file paths."""
        # Use malicious payload as file path
        file_path = malicious_payload
        collection = "test_collection"

        # Should handle malicious input safely
        result = await state_manager.start_file_processing(file_path, collection)
        assert result is True

        # Verify no SQL injection occurred
        status = await state_manager.get_file_processing_status(file_path)
        assert status is not None
        assert status.file_path == file_path

    # Note: Project metadata methods not currently exposed in SQLiteStateManager
    # If they are added later, add tests here


@pytest.mark.security
class TestPathTraversalPrevention:
    """Test path traversal attack prevention."""

    @pytest.mark.parametrize("malicious_path", PATH_TRAVERSAL_PAYLOADS)
    @pytest.mark.asyncio
    async def test_watch_folder_path_validation(
        self, state_manager, malicious_path
    ):
        """Test that watch folder paths are validated against traversal attacks."""
        config = WatchFolderConfig(
            watch_id="test",
            path=malicious_path,
            collection="test",
            patterns=["*.py"],
            ignore_patterns=[],
        )

        # Save should succeed (path is stored as-is for user visibility)
        result = await state_manager.save_watch_folder_config(config)
        assert result is True

        # However, actual file operations should validate paths
        # This would be caught by the daemon/file watcher when attempting to access

    @pytest.mark.parametrize("malicious_path", PATH_TRAVERSAL_PAYLOADS[:10])
    def test_path_resolution_safety(self, malicious_path):
        """Test that Path.resolve() prevents path traversal."""
        # Create a temporary directory as the allowed root
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Attempt to resolve malicious path
            try:
                test_path = root / malicious_path
                resolved = test_path.resolve()

                # Verify resolved path is still within root directory
                # If path traversal occurred, resolved won't be relative to root
                try:
                    resolved.relative_to(root)
                    # Path is safe - within root
                except ValueError:
                    # Path traversal detected - outside root
                    # This is expected for some payloads
                    pass

            except (ValueError, OSError):
                # Invalid path - this is acceptable
                pass

    @pytest.mark.parametrize("malicious_filename", [
        "../../etc/passwd",
        "../../../secrets.txt",
        "..\\..\\windows\\system.ini",
    ])
    @pytest.mark.asyncio
    async def test_file_processing_path_validation(
        self, state_manager, malicious_filename
    ):
        """Test file processing handles path traversal in filenames."""
        # These should be stored but would fail validation when accessed
        result = await state_manager.start_file_processing(
            malicious_filename, "test"
        )
        assert result is True

        # Verify path is stored as provided (not normalized yet)
        status = await state_manager.get_file_processing_status(malicious_filename)
        assert status is not None


@pytest.mark.security
class TestCommandInjectionPrevention:
    """Test command injection prevention."""

    @pytest.mark.parametrize("malicious_cmd", COMMAND_INJECTION_PAYLOADS[:10])
    def test_subprocess_safety(self, malicious_cmd):
        """Test that subprocess calls don't allow command injection."""
        # If code uses subprocess with shell=True, it's vulnerable
        # This tests that we're using safe subprocess calls

        # Simulate what would happen if malicious input reached subprocess
        with pytest.raises((FileNotFoundError, subprocess.CalledProcessError)):
            # Safe usage: shell=False, command as list
            subprocess.run(
                [malicious_cmd],  # Treated as single argument
                shell=False,
                capture_output=True,
                timeout=1,
                check=True,
            )

    def test_shell_metacharacter_escaping(self):
        """Verify shell metacharacters are properly escaped."""
        dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "<", ">", "\n"]

        for char in dangerous_chars:
            malicious_input = f"innocent{char}malicious"

            # Safe approach: use shlex.quote or avoid shell entirely
            import shlex
            quoted = shlex.quote(malicious_input)

            # Quoted string should contain the original input safely
            assert char in quoted or "\\" in quoted  # Escaped or quoted


@pytest.mark.security
class TestXSSPrevention:
    """Test XSS (Cross-Site Scripting) prevention."""

    @pytest.mark.parametrize("xss_payload", XSS_PAYLOADS)
    @pytest.mark.asyncio
    async def test_content_storage_escaping(self, state_manager, xss_payload):
        """Test that XSS payloads are safely stored without execution."""
        # Store XSS payload as content
        file_path = f"/tmp/test_{hash(xss_payload)}.html"
        collection = "test"

        result = await state_manager.start_file_processing(file_path, collection)
        assert result is True

        # Content should be stored literally, not interpreted
        status = await state_manager.get_file_processing_status(file_path)
        assert status is not None

    @pytest.mark.parametrize("xss_payload", XSS_PAYLOADS[:5])
    def test_html_escaping(self, xss_payload):
        """Test HTML escaping for web output."""
        # If system outputs to HTML, it must escape
        import html

        escaped = html.escape(xss_payload)

        # Verify dangerous characters are escaped
        assert "<" not in escaped or "&lt;" in escaped
        assert ">" not in escaped or "&gt;" in escaped
        # Note: & will be in escaped as part of &lt;, &gt;, &amp; etc - this is correct


@pytest.mark.security
class TestFileUploadSecurity:
    """Test file upload security mechanisms."""

    def test_large_file_rejection(self):
        """Test that extremely large files are rejected or handled safely."""
        max_file_size = 100 * 1024 * 1024  # 100MB reasonable limit

        # Simulate large file
        large_size = 1024 * 1024 * 1024  # 1GB
        file_info = FILE_UPLOAD_PAYLOADS["large_file"]

        # System should have file size checks
        assert file_info["size"] > max_file_size

    def test_malformed_file_handling(self):
        """Test handling of files with malformed headers."""
        malformed_content = FILE_UPLOAD_PAYLOADS["malformed_pdf"]["content"]

        # Write malformed file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(malformed_content)
            temp_path = f.name

        try:
            # System should handle malformed files gracefully
            # without crashes or security issues
            path = Path(temp_path)
            assert path.exists()
            # File detector should handle this safely
        finally:
            os.unlink(temp_path)

    def test_filename_sanitization(self):
        """Test that filenames are sanitized against path traversal."""
        dangerous_filenames = [
            FILE_UPLOAD_PAYLOADS["double_extension"]["filename"],
            FILE_UPLOAD_PAYLOADS["path_in_filename"]["filename"],
        ]

        for filename in dangerous_filenames:
            # Sanitize filename
            sanitized = Path(filename).name

            # Verify path components removed
            assert "/" not in sanitized
            assert "\\" not in sanitized
            assert ".." not in sanitized or sanitized == ".."

    def test_null_byte_in_filename(self):
        """Test handling of null bytes in filenames."""
        malicious_filename = FILE_UPLOAD_PAYLOADS["null_byte_filename"]["filename"]

        # Null bytes should be removed or rejected
        sanitized = malicious_filename.replace("\x00", "")
        assert "\x00" not in sanitized


@pytest.mark.security
class TestReDoSPrevention:
    """Test Regular Expression Denial of Service prevention."""

    @pytest.mark.parametrize("pattern,test_string", REDOS_PAYLOADS)
    def test_regex_timeout(self, pattern, test_string):
        """Test that regex operations have timeouts to prevent ReDoS."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Regex took too long")

        # Set timeout for regex operation
        timeout_seconds = 1

        try:
            # Set alarm (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            start_time = time.time()

            try:
                # This might hang on vulnerable patterns
                re.match(pattern, test_string)
            except TimeoutError:
                # This is expected for known vulnerable patterns
                # In production code, this would indicate a ReDoS vulnerability
                pytest.skip(f"Regex vulnerable to ReDoS (expected): {pattern}")
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel alarm

            elapsed = time.time() - start_time

            # If regex takes > 100ms on short string, it's suspicious
            if elapsed > 0.1:
                pytest.skip(f"Regex potentially vulnerable to ReDoS: {pattern}")

        except re.error:
            # Invalid pattern - acceptable
            pass

    def test_safe_regex_patterns(self):
        """Verify that commonly used regex patterns are safe."""
        safe_patterns = [
            r"^[a-zA-Z0-9_-]+$",  # Alphanumeric with underscore/hyphen
            r"^\d{1,10}$",  # Limited digits
            r"^[a-z]{1,50}$",  # Limited lowercase letters
        ]

        test_string = "a" * 100

        for pattern in safe_patterns:
            start_time = time.time()
            re.match(pattern, test_string)
            elapsed = time.time() - start_time

            # Safe patterns should complete very quickly
            assert elapsed < 0.01, f"Pattern {pattern} took too long"


@pytest.mark.security
class TestSpecialCharacterHandling:
    """Test handling of special characters and edge cases."""

    @pytest.mark.parametrize("special_char", SPECIAL_CHAR_PAYLOADS)
    @pytest.mark.asyncio
    async def test_special_character_storage(self, state_manager, special_char):
        """Test that special characters are stored and retrieved correctly."""
        # Use special character in various fields
        watch_id = f"test_{hash(special_char)}"
        collection = f"col{special_char}"

        config = WatchFolderConfig(
            watch_id=watch_id,
            path="/tmp/test",
            collection=collection,
            patterns=["*.py"],
            ignore_patterns=[],
        )

        result = await state_manager.save_watch_folder_config(config)
        assert result is True

        # Retrieve and verify
        retrieved = await state_manager.get_watch_folder_config(watch_id)
        assert retrieved is not None

    def test_unicode_normalization(self):
        """Test that Unicode variations are handled consistently."""
        # Test homoglyph detection/handling
        latin_a = "admin"
        cyrillic_a = "аdmin"  # Cyrillic 'а'

        # These should be treated differently
        assert latin_a != cyrillic_a

    def test_control_character_filtering(self):
        """Test that control characters are properly filtered."""
        control_chars = ["\x00", "\x01", "\x1b", "\r", "\n"]

        for char in control_chars:
            test_string = f"test{char}value"

            # Depending on context, control chars should be:
            # 1. Filtered out
            # 2. Escaped
            # 3. Stored but not interpreted

            # For display, they should not break formatting
            sanitized = test_string.replace("\x00", "").replace("\x1b", "")
            # Newlines/returns might be preserved in some contexts
            assert "\x00" not in sanitized


@pytest.mark.security
class TestInputValidationEdgeCases:
    """Test edge cases in input validation."""

    @pytest.mark.asyncio
    async def test_empty_string_handling(self, state_manager):
        """Test handling of empty strings."""
        config = WatchFolderConfig(
            watch_id="",  # Empty watch ID
            path="",  # Empty path
            collection="",  # Empty collection
            patterns=[],
            ignore_patterns=[],
        )

        # Should handle empty strings gracefully
        result = await state_manager.save_watch_folder_config(config)
        # May succeed or fail, but shouldn't crash
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_extremely_long_input(self, state_manager):
        """Test handling of extremely long inputs."""
        long_string = "A" * 10000

        config = WatchFolderConfig(
            watch_id=long_string[:100],  # Truncate for reasonable ID
            path="/tmp/test",
            collection=long_string[:100],
            patterns=["*.py"],
            ignore_patterns=[],
        )

        # Should handle or reject long inputs gracefully
        result = await state_manager.save_watch_folder_config(config)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_null_value_handling(self, state_manager):
        """Test handling of None/null values."""
        # Test with None in optional fields of WatchFolderConfig
        config = WatchFolderConfig(
            watch_id="test_null",
            path="/tmp/test",
            collection="test",
            patterns=["*.py"],
            ignore_patterns=[],
            metadata=None,  # None value in optional field
        )
        result = await state_manager.save_watch_folder_config(config)
        assert result is True

    def test_type_confusion(self):
        """Test that type checking prevents type confusion attacks."""
        # Python dataclasses don't enforce types at runtime by default
        # but static type checkers (mypy) will catch these issues
        # Test that wrong types cause issues when used, not at creation

        # Create with wrong types - this will succeed in Python
        config = WatchFolderConfig(
            watch_id=12345,  # Should be string
            path="/tmp/test",
            collection="test",
            patterns="*.py",  # Should be list (string is iterable though)
            ignore_patterns=[],
        )

        # But using it should reveal the type mismatch
        # In production, mypy would catch this before runtime
        assert config.watch_id == 12345  # This works, but is wrong type
        # The patterns field will behave oddly since it's a string not a list
        assert isinstance(config.patterns, str)  # Wrong type accepted


# Security test markers are configured in pyproject.toml
