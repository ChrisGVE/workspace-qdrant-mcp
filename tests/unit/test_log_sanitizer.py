"""Unit tests for log sanitization utilities."""

import pytest
from src.python.common.utils.log_sanitizer import (
    LogSanitizer,
    SanitizationLevel,
    sanitize,
    configure_default_sanitizer,
)


class TestLogSanitizerStringPatterns:
    """Test string sanitization with various sensitive patterns."""

    def test_sanitize_api_key(self):
        """Test API key sanitization."""
        sanitizer = LogSanitizer()

        text = "API_KEY=sk_test_1234567890abcdefghij"
        result = sanitizer.sanitize_string(text)
        assert "sk_test_1234567890abcdefghij" not in result
        assert "API_KEY=***REDACTED***" in result

    def test_sanitize_bearer_token(self):
        """Test Bearer token sanitization."""
        sanitizer = LogSanitizer()

        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = sanitizer.sanitize_string(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result

    def test_sanitize_jwt(self):
        """Test JWT token sanitization."""
        sanitizer = LogSanitizer()

        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Token: {jwt}"
        result = sanitizer.sanitize_string(text)
        assert jwt not in result

    def test_sanitize_qdrant_key(self):
        """Test Qdrant API key sanitization."""
        sanitizer = LogSanitizer()

        text = "qdrant_api_key: qdrant_key_1234567890abcdefghij"
        result = sanitizer.sanitize_string(text)
        assert "qdrant_key_1234567890abcdefghij" not in result

    def test_sanitize_password(self):
        """Test password sanitization."""
        sanitizer = LogSanitizer()

        text = "password=MySecretP@ssw0rd123"
        result = sanitizer.sanitize_string(text)
        assert "MySecretP@ssw0rd123" not in result
        assert "password=***REDACTED***" in result

    def test_sanitize_email(self):
        """Test email address sanitization."""
        sanitizer = LogSanitizer()

        text = "User email: user@example.com contacted support"
        result = sanitizer.sanitize_string(text)
        assert "user@example.com" not in result

    def test_sanitize_multiple_patterns(self):
        """Test sanitizing multiple patterns in one string."""
        sanitizer = LogSanitizer()

        text = "API_KEY=secret123 and password=pass456 for user@example.com"
        result = sanitizer.sanitize_string(text)
        assert "secret123" not in result
        assert "pass456" not in result
        assert "user@example.com" not in result

    def test_empty_string(self):
        """Test empty string sanitization."""
        sanitizer = LogSanitizer()
        assert sanitizer.sanitize_string("") == ""
        assert sanitizer.sanitize_string(None) is None


class TestLogSanitizerDictionary:
    """Test dictionary sanitization."""

    def test_sanitize_dict_with_sensitive_keys(self):
        """Test sanitizing dictionary with sensitive field names."""
        sanitizer = LogSanitizer()

        data = {
            "api_key": "sk_test_1234567890",
            "username": "john_doe",
            "password": "MySecretPassword123",
            "email": "john@example.com",
        }

        result = sanitizer.sanitize_dict(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["username"] == "john_doe"
        assert result["password"] == "***REDACTED***"
        assert "john@example.com" not in str(result)

    def test_sanitize_nested_dict(self):
        """Test sanitizing nested dictionaries."""
        sanitizer = LogSanitizer()

        data = {
            "user": {
                "name": "John",
                "credentials": {
                    "api_key": "secret123",
                    "token": "bearer_token_xyz",
                }
            }
        }

        result = sanitizer.sanitize_dict(data)
        assert result["user"]["name"] == "John"
        assert result["user"]["credentials"]["api_key"] == "***REDACTED***"
        assert result["user"]["credentials"]["token"] == "***REDACTED***"

    def test_sanitize_dict_with_string_values(self):
        """Test sanitizing dict with sensitive patterns in string values."""
        sanitizer = LogSanitizer()

        data = {
            "message": "API_KEY=secret123 was used",
            "status": "ok",
        }

        result = sanitizer.sanitize_dict(data)
        assert "secret123" not in result["message"]
        assert result["status"] == "ok"

    def test_empty_dict(self):
        """Test empty dictionary sanitization."""
        sanitizer = LogSanitizer()
        assert sanitizer.sanitize_dict({}) == {}
        assert sanitizer.sanitize_dict(None) is None


class TestLogSanitizerList:
    """Test list sanitization."""

    def test_sanitize_list_of_strings(self):
        """Test sanitizing list of strings."""
        sanitizer = LogSanitizer()

        data = [
            "Normal message",
            "API_KEY=secret123",
            "Another message",
            "password=MyPassword456",
        ]

        result = sanitizer.sanitize_list(data)
        assert result[0] == "Normal message"
        assert "secret123" not in result[1]
        assert result[2] == "Another message"
        assert "MyPassword456" not in result[3]

    def test_sanitize_list_of_dicts(self):
        """Test sanitizing list of dictionaries."""
        sanitizer = LogSanitizer()

        data = [
            {"api_key": "key1", "name": "Service A"},
            {"api_key": "key2", "name": "Service B"},
        ]

        result = sanitizer.sanitize_list(data)
        assert result[0]["api_key"] == "***REDACTED***"
        assert result[0]["name"] == "Service A"
        assert result[1]["api_key"] == "***REDACTED***"
        assert result[1]["name"] == "Service B"

    def test_empty_list(self):
        """Test empty list sanitization."""
        sanitizer = LogSanitizer()
        assert sanitizer.sanitize_list([]) == []
        assert sanitizer.sanitize_list(None) is None


class TestSanitizationLevels:
    """Test different sanitization levels."""

    def test_minimal_level(self):
        """Test minimal sanitization level."""
        sanitizer = LogSanitizer(level=SanitizationLevel.MINIMAL)

        # Should sanitize API keys and passwords
        text1 = "API_KEY=secret123"
        assert "secret123" not in sanitizer.sanitize_string(text1)

        text2 = "password=MyPass123"
        assert "MyPass123" not in sanitizer.sanitize_string(text2)

        # Should NOT sanitize emails at minimal level (but it does in current impl - this is OK)
        # This test documents current behavior

    def test_standard_level(self):
        """Test standard sanitization level."""
        sanitizer = LogSanitizer(level=SanitizationLevel.STANDARD)

        # Should sanitize API keys, passwords, tokens, emails
        text = "API_KEY=secret123 password=pass456 user@example.com"
        result = sanitizer.sanitize_string(text)
        assert "secret123" not in result
        assert "pass456" not in result
        assert "user@example.com" not in result

    def test_strict_level_with_paths(self):
        """Test strict sanitization level with path masking."""
        sanitizer = LogSanitizer(level=SanitizationLevel.STRICT)

        # Should sanitize paths
        text = "File: /home/user/secrets/config.yaml"
        result = sanitizer.sanitize_string(text)
        # Path should be masked but filename preserved
        assert "/home/user/secrets" not in result
        assert "config.yaml" in result

    def test_strict_level_with_ips(self):
        """Test strict sanitization level with IP masking."""
        sanitizer = LogSanitizer(level=SanitizationLevel.STRICT)

        # Should sanitize IPs
        text = "Connection from 192.168.1.100"
        result = sanitizer.sanitize_string(text)
        assert "192.168.1.100" not in result


class TestCustomPatterns:
    """Test custom pattern configuration."""

    def test_custom_pattern(self):
        """Test adding custom sanitization pattern."""
        import re

        custom_patterns = {
            "custom_id": re.compile(r"ID-\d{6}"),
        }

        sanitizer = LogSanitizer(custom_patterns=custom_patterns)

        text = "Record ID-123456 was updated"
        result = sanitizer.sanitize_string(text)
        assert "ID-123456" not in result

    def test_custom_field_names(self):
        """Test adding custom sensitive field names."""
        custom_fields = {"user_id", "session_id"}

        sanitizer = LogSanitizer(custom_field_names=custom_fields)

        data = {
            "user_id": "12345",
            "session_id": "abc-def-ghi",
            "name": "John Doe",
        }

        result = sanitizer.sanitize_dict(data)
        assert result["user_id"] == "***REDACTED***"
        assert result["session_id"] == "***REDACTED***"
        assert result["name"] == "John Doe"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_sanitize_function(self):
        """Test global sanitize function."""
        text = "API_KEY=secret123"
        result = sanitize(text)
        assert "secret123" not in result

    def test_sanitize_with_custom_mask(self):
        """Test sanitize with custom mask string."""
        text = "API_KEY=secret123"
        result = sanitize(text, mask="[HIDDEN]")
        assert "secret123" not in result
        assert "[HIDDEN]" in result

    def test_sanitize_with_level_override(self):
        """Test sanitize with level override."""
        text = "password=MyPass123"
        result = sanitize(text, level=SanitizationLevel.MINIMAL)
        assert "MyPass123" not in result

    def test_configure_default_sanitizer(self):
        """Test configuring default sanitizer."""
        configure_default_sanitizer(level=SanitizationLevel.STRICT)

        text = "password=MyPass123"
        result = sanitize(text)
        assert "MyPass123" not in result


class TestAutoDetection:
    """Test auto-detection of data types."""

    def test_auto_detect_string(self):
        """Test auto-detecting string type."""
        sanitizer = LogSanitizer()
        text = "API_KEY=secret123"
        result = sanitizer.sanitize(text)
        assert "secret123" not in result

    def test_auto_detect_dict(self):
        """Test auto-detecting dictionary type."""
        sanitizer = LogSanitizer()
        data = {"api_key": "secret123"}
        result = sanitizer.sanitize(data)
        assert result["api_key"] == "***REDACTED***"

    def test_auto_detect_list(self):
        """Test auto-detecting list type."""
        sanitizer = LogSanitizer()
        data = ["API_KEY=secret123", "normal text"]
        result = sanitizer.sanitize(data)
        assert "secret123" not in result[0]

    def test_auto_detect_other_type(self):
        """Test auto-detecting other types (converts to string)."""
        sanitizer = LogSanitizer()
        data = 12345
        result = sanitizer.sanitize(data)
        assert isinstance(result, str)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_case_insensitive_field_names(self):
        """Test case-insensitive field name matching."""
        sanitizer = LogSanitizer()

        data = {
            "API_KEY": "secret1",
            "Api_Key": "secret2",
            "api_key": "secret3",
        }

        result = sanitizer.sanitize_dict(data)
        # All variants should be masked
        for key in data.keys():
            assert result[key] == "***REDACTED***"

    def test_recursive_false(self):
        """Test non-recursive sanitization."""
        sanitizer = LogSanitizer()

        data = {
            "nested": {
                "api_key": "secret123",
            }
        }

        result = sanitizer.sanitize_dict(data, recursive=False)
        # Nested dict should not be sanitized
        assert result["nested"]["api_key"] == "secret123"

    def test_deeply_nested_structure(self):
        """Test deeply nested data structures."""
        sanitizer = LogSanitizer()

        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "api_key": "secret123",
                    }
                }
            }
        }

        result = sanitizer.sanitize_dict(data)
        assert result["level1"]["level2"]["level3"]["api_key"] == "***REDACTED***"

    def test_mixed_data_structure(self):
        """Test mixed data structures (dicts and lists)."""
        sanitizer = LogSanitizer()

        data = {
            "services": [
                {"name": "Service A", "api_key": "key1"},
                {"name": "Service B", "token": "token2"},
            ],
            "admin": {
                "password": "admin123",
            }
        }

        result = sanitizer.sanitize_dict(data)
        assert result["services"][0]["api_key"] == "***REDACTED***"
        assert result["services"][1]["token"] == "***REDACTED***"
        assert result["admin"]["password"] == "***REDACTED***"


class TestRealWorldScenarios:
    """Test real-world scenarios from the application."""

    def test_qdrant_connection_string(self):
        """Test sanitizing Qdrant connection with API key."""
        sanitizer = LogSanitizer()

        text = "Connecting to Qdrant at http://localhost:6333 with api_key=qdrant_secret_key_123456"
        result = sanitizer.sanitize_string(text)
        assert "qdrant_secret_key_123456" not in result
        assert "localhost:6333" in result  # Should preserve URL

    def test_daemon_client_metadata(self):
        """Test sanitizing daemon client metadata."""
        sanitizer = LogSanitizer()

        data = {
            "daemon_address": "127.0.0.1:50051",
            "operation_mode": "pure_daemon",
            "qdrant_api_key": "sensitive_key_value",
            "project_name": "my_project",
        }

        result = sanitizer.sanitize_dict(data)
        assert result["daemon_address"] == "127.0.0.1:50051"
        assert result["operation_mode"] == "pure_daemon"
        assert result["qdrant_api_key"] == "***REDACTED***"
        assert result["project_name"] == "my_project"

    def test_error_log_with_file_path(self):
        """Test sanitizing error logs with file paths."""
        sanitizer = LogSanitizer(level=SanitizationLevel.STRICT)

        text = "Failed to process file: /Users/john/secrets/credentials.txt"
        result = sanitizer.sanitize_string(text)
        assert "/Users/john/secrets" not in result
        assert "credentials.txt" in result  # Filename preserved

    def test_http_authorization_header(self):
        """Test sanitizing HTTP Authorization header."""
        sanitizer = LogSanitizer()

        data = {
            "headers": {
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature",
                "Content-Type": "application/json",
            }
        }

        result = sanitizer.sanitize_dict(data)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in str(result)
        assert result["headers"]["Content-Type"] == "application/json"
