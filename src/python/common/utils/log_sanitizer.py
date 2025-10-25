"""Log sanitization utilities to prevent sensitive data leakage.

This module provides utilities to sanitize sensitive information from logs,
preventing accidental exposure of API keys, tokens, file paths, and other
confidential data.

Security Note: This addresses SECURITY_AUDIT.md finding S4.
"""

import re
from typing import Any

# Patterns for sensitive data detection
SENSITIVE_PATTERNS = {
    # API keys and tokens
    "api_key": re.compile(r"(api[-_]?key|apikey)[\s:=]+['\"]?([a-zA-Z0-9_-]{6,})['\"]?", re.IGNORECASE),
    "bearer_token": re.compile(r"bearer\s+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE),
    "jwt": re.compile(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"),

    # Qdrant API keys (specific format)
    "qdrant_key": re.compile(r"qdrant[-_]?(api[-_]?key)?[\s:=]+['\"]?([a-zA-Z0-9_-]{6,})['\"]?", re.IGNORECASE),

    # Generic secrets
    "password": re.compile(r"(password|passwd|pwd)[\s:=]+['\"]?([^'\"\s]{6,})['\"]?", re.IGNORECASE),
    "secret": re.compile(r"(secret|token)[\s:=]+['\"]?([a-zA-Z0-9_-]{6,})['\"]?", re.IGNORECASE),

    # File paths (optional - can expose directory structure)
    "absolute_path": re.compile(r"(/[a-zA-Z0-9_\-./]+|[A-Z]:\\[a-zA-Z0-9_\-\\./]+)"),

    # Email addresses (PII)
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),

    # IP addresses (optional - can reveal network topology)
    "ipv4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
}

# Field names that commonly contain sensitive data
SENSITIVE_FIELD_NAMES: set[str] = {
    "api_key",
    "apikey",
    "api-key",
    "qdrant_api_key",
    "qdrant_key",
    "authorization",
    "auth",
    "bearer",
    "token",
    "access_token",
    "refresh_token",
    "jwt",
    "secret",
    "password",
    "passwd",
    "pwd",
    "credentials",
    "private_key",
    "key",
    "cert",
    "certificate",
}


class SanitizationLevel:
    """Sanitization level configuration."""

    MINIMAL = "minimal"  # Only mask known API keys and passwords
    STANDARD = "standard"  # Mask API keys, passwords, tokens, emails
    STRICT = "strict"  # Mask all sensitive patterns including paths and IPs
    PARANOID = "paranoid"  # Mask all potentially sensitive data


class LogSanitizer:
    """Sanitize sensitive information from log messages and data structures."""

    def __init__(
        self,
        level: str = SanitizationLevel.STANDARD,
        custom_patterns: dict[str, re.Pattern] | None = None,
        custom_field_names: set[str] | None = None,
        mask_paths: bool = False,
        mask_ips: bool = False,
    ):
        """Initialize log sanitizer.

        Args:
            level: Sanitization level (minimal, standard, strict, paranoid)
            custom_patterns: Additional regex patterns to sanitize
            custom_field_names: Additional field names to sanitize
            mask_paths: Whether to mask file system paths
            mask_ips: Whether to mask IP addresses
        """
        self.level = level
        self.mask_paths = mask_paths or level in (SanitizationLevel.STRICT, SanitizationLevel.PARANOID)
        self.mask_ips = mask_ips or level in (SanitizationLevel.STRICT, SanitizationLevel.PARANOID)

        # Build pattern set based on level
        self.patterns = self._build_patterns(custom_patterns)
        self.sensitive_fields = SENSITIVE_FIELD_NAMES.copy()
        if custom_field_names:
            self.sensitive_fields.update(custom_field_names)

    def _build_patterns(self, custom: dict[str, re.Pattern] | None = None) -> dict[str, re.Pattern]:
        """Build sanitization patterns based on level."""
        patterns = {}

        if self.level == SanitizationLevel.MINIMAL:
            patterns["api_key"] = SENSITIVE_PATTERNS["api_key"]
            patterns["password"] = SENSITIVE_PATTERNS["password"]
            patterns["qdrant_key"] = SENSITIVE_PATTERNS["qdrant_key"]

        elif self.level == SanitizationLevel.STANDARD:
            patterns["api_key"] = SENSITIVE_PATTERNS["api_key"]
            patterns["bearer_token"] = SENSITIVE_PATTERNS["bearer_token"]
            patterns["jwt"] = SENSITIVE_PATTERNS["jwt"]
            patterns["qdrant_key"] = SENSITIVE_PATTERNS["qdrant_key"]
            patterns["password"] = SENSITIVE_PATTERNS["password"]
            patterns["secret"] = SENSITIVE_PATTERNS["secret"]
            patterns["email"] = SENSITIVE_PATTERNS["email"]

        else:  # STRICT or PARANOID
            patterns.update(SENSITIVE_PATTERNS)

        # Remove patterns based on flags
        if not self.mask_paths and "absolute_path" in patterns:
            del patterns["absolute_path"]
        if not self.mask_ips and "ipv4" in patterns:
            del patterns["ipv4"]

        # Add custom patterns
        if custom:
            patterns.update(custom)

        return patterns

    def sanitize_string(self, text: str, mask: str = "***REDACTED***") -> str:
        """Sanitize a string by masking sensitive patterns.

        Args:
            text: Text to sanitize
            mask: Replacement string for sensitive data

        Returns:
            Sanitized text with sensitive data masked
        """
        if not text:
            return text

        sanitized = text
        for pattern_name, pattern in self.patterns.items():
            if pattern_name == "absolute_path":
                # For paths, keep just the filename
                sanitized = pattern.sub(lambda m: self._mask_path(m.group(0), mask), sanitized)
            elif pattern_name in ("api_key", "qdrant_key", "password", "secret"):
                # For key-value patterns, mask only the value
                sanitized = pattern.sub(lambda m: self._mask_key_value(m, mask), sanitized)
            else:
                sanitized = pattern.sub(mask, sanitized)

        return sanitized

    def _mask_path(self, path: str, mask: str) -> str:
        """Mask file path while preserving filename."""
        import os
        filename = os.path.basename(path)
        return f"{mask}/{filename}"

    def _mask_key_value(self, match: re.Match, mask: str) -> str:
        """Mask value in key=value pattern while preserving key."""
        groups = match.groups()
        if len(groups) >= 2:
            key = groups[0]
            return f"{key}={mask}"
        return mask

    def sanitize_dict(
        self,
        data: dict[str, Any],
        mask: str = "***REDACTED***",
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Sanitize dictionary by masking sensitive fields.

        Args:
            data: Dictionary to sanitize
            mask: Replacement string for sensitive data
            recursive: Whether to recursively sanitize nested dicts

        Returns:
            New dictionary with sensitive data masked
        """
        if not data:
            return data

        sanitized = {}
        for key, value in data.items():
            # Recursively sanitize nested structures first (takes priority)
            if isinstance(value, dict) and recursive:
                sanitized[key] = self.sanitize_dict(value, mask, recursive)
            elif isinstance(value, list) and recursive:
                sanitized[key] = self.sanitize_list(value, mask, recursive)
            # Check if field name is sensitive (for non-container types)
            elif key.lower() in self.sensitive_fields:
                sanitized[key] = mask
            elif isinstance(value, str):
                sanitized[key] = self.sanitize_string(value, mask)
            else:
                sanitized[key] = value

        return sanitized

    def sanitize_list(
        self,
        data: list[Any],
        mask: str = "***REDACTED***",
        recursive: bool = True,
    ) -> list[Any]:
        """Sanitize list by masking sensitive items.

        Args:
            data: List to sanitize
            mask: Replacement string for sensitive data
            recursive: Whether to recursively sanitize nested structures

        Returns:
            New list with sensitive data masked
        """
        if not data:
            return data

        sanitized = []
        for item in data:
            if isinstance(item, str):
                sanitized.append(self.sanitize_string(item, mask))
            elif isinstance(item, dict) and recursive:
                sanitized.append(self.sanitize_dict(item, mask, recursive))
            elif isinstance(item, list) and recursive:
                sanitized.append(self.sanitize_list(item, mask, recursive))
            else:
                sanitized.append(item)

        return sanitized

    def sanitize(
        self,
        data: str | dict | list | Any,
        mask: str = "***REDACTED***",
    ) -> str | dict | list | Any:
        """Sanitize any data type (auto-detects type).

        Args:
            data: Data to sanitize (string, dict, list, or other)
            mask: Replacement string for sensitive data

        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            return self.sanitize_string(data, mask)
        elif isinstance(data, dict):
            return self.sanitize_dict(data, mask)
        elif isinstance(data, list):
            return self.sanitize_list(data, mask)
        else:
            # For other types, convert to string and sanitize
            return self.sanitize_string(str(data), mask)


# Global sanitizer instance (can be reconfigured)
_default_sanitizer = LogSanitizer(level=SanitizationLevel.STANDARD)


def sanitize(
    data: str | dict | list | Any,
    mask: str = "***REDACTED***",
    level: str | None = None,
) -> str | dict | list | Any:
    """Convenience function to sanitize data using default sanitizer.

    Args:
        data: Data to sanitize
        mask: Replacement string for sensitive data
        level: Optional sanitization level override

    Returns:
        Sanitized data
    """
    if level:
        sanitizer = LogSanitizer(level=level)
        return sanitizer.sanitize(data, mask)
    return _default_sanitizer.sanitize(data, mask)


def configure_default_sanitizer(
    level: str = SanitizationLevel.STANDARD,
    mask_paths: bool = False,
    mask_ips: bool = False,
    **kwargs,
) -> None:
    """Configure the default global sanitizer.

    Args:
        level: Sanitization level
        mask_paths: Whether to mask file paths
        mask_ips: Whether to mask IP addresses
        **kwargs: Additional arguments for LogSanitizer
    """
    global _default_sanitizer
    _default_sanitizer = LogSanitizer(
        level=level,
        mask_paths=mask_paths,
        mask_ips=mask_ips,
        **kwargs,
    )


# Export main API
__all__ = [
    "LogSanitizer",
    "SanitizationLevel",
    "sanitize",
    "configure_default_sanitizer",
]
