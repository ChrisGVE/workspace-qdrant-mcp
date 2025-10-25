"""Input validation and sanitization for enhanced security.

This module provides comprehensive input validation and sanitization including:
- Data type validation and conversion
- String sanitization and encoding validation
- Path traversal prevention
- SQL injection prevention
- Command injection prevention
- Size and length limits enforcement
- Pattern matching and regex validation
- JSON schema validation
"""

import html
import json
import re
import urllib.parse
import uuid
from collections.abc import Callable
from pathlib import Path, PurePath
from re import Pattern
from typing import Any

from loguru import logger


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class SanitizationError(Exception):
    """Raised when input sanitization fails."""
    pass


class InputValidator:
    """Comprehensive input validator with security hardening."""

    # Dangerous characters and patterns
    DANGEROUS_CHARACTERS = {
        'sql_injection': [';', '--', '/*', '*/', 'xp_', 'sp_', 'exec', 'execute'],
        'command_injection': ['&', '|', ';', '$', '`', '$(', '||', '&&'],
        'path_traversal': ['../', '..\\', '%2e%2e', '%2f', '%5c'],
        'script_injection': ['<script', '</script>', 'javascript:', 'vbscript:', 'onload=', 'onerror='],
    }

    # Safe character sets
    ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9]+$')
    ALPHANUMERIC_DASH_UNDERSCORE = re.compile(r'^[a-zA-Z0-9_-]+$')
    SAFE_FILENAME = re.compile(r'^[a-zA-Z0-9._-]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

    def __init__(
        self,
        max_string_length: int = 1024,
        max_collection_size: int = 1000,
        allow_html: bool = False,
        strict_mode: bool = True,
    ):
        """Initialize input validator.

        Args:
            max_string_length: Maximum allowed string length
            max_collection_size: Maximum allowed collection size
            allow_html: Whether to allow HTML content
            strict_mode: Whether to use strict validation
        """
        self.max_string_length = max_string_length
        self.max_collection_size = max_collection_size
        self.allow_html = allow_html
        self.strict_mode = strict_mode

    def validate_string(
        self,
        value: Any,
        min_length: int = 0,
        max_length: int | None = None,
        pattern: Pattern | None = None,
        allowed_chars: set[str] | None = None,
        field_name: str = "string",
    ) -> str:
        """Validate and sanitize string input.

        Args:
            value: Input value to validate
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern to match
            allowed_chars: Set of allowed characters
            field_name: Field name for error messages

        Returns:
            Validated and sanitized string

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (str, bytes)):
            try:
                value = str(value)
            except Exception:
                raise ValidationError(f"{field_name}: Value cannot be converted to string")

        if isinstance(value, bytes):
            try:
                value = value.decode('utf-8')
            except UnicodeDecodeError:
                raise ValidationError(f"{field_name}: Invalid UTF-8 encoding")

        # Length validation
        actual_max = max_length or self.max_string_length
        if len(value) < min_length:
            raise ValidationError(f"{field_name}: String too short (minimum {min_length} characters)")
        if len(value) > actual_max:
            raise ValidationError(f"{field_name}: String too long (maximum {actual_max} characters)")

        # Pattern validation
        if pattern and not pattern.match(value):
            raise ValidationError(f"{field_name}: String does not match required pattern")

        # Character set validation
        if allowed_chars:
            invalid_chars = set(value) - allowed_chars
            if invalid_chars:
                raise ValidationError(f"{field_name}: Contains invalid characters: {invalid_chars}")

        # Security checks
        self._check_security_patterns(value, field_name)

        # Sanitization
        sanitized_value = self._sanitize_string(value)

        return sanitized_value

    def validate_integer(
        self,
        value: Any,
        min_value: int | None = None,
        max_value: int | None = None,
        field_name: str = "integer",
    ) -> int:
        """Validate integer input.

        Args:
            value: Input value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Field name for error messages

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(value, str):
                # Remove whitespace and validate format
                value = value.strip()
                if not re.match(r'^-?\d+$', value):
                    raise ValidationError(f"{field_name}: Invalid integer format")

            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name}: Cannot convert to integer")

        if min_value is not None and int_value < min_value:
            raise ValidationError(f"{field_name}: Value {int_value} below minimum {min_value}")

        if max_value is not None and int_value > max_value:
            raise ValidationError(f"{field_name}: Value {int_value} above maximum {max_value}")

        return int_value

    def validate_float(
        self,
        value: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        field_name: str = "float",
    ) -> float:
        """Validate float input.

        Args:
            value: Input value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Field name for error messages

        Returns:
            Validated float

        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(value, str):
                # Remove whitespace and validate format
                value = value.strip()
                if not re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', value):
                    raise ValidationError(f"{field_name}: Invalid float format")

            float_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name}: Cannot convert to float")

        if min_value is not None and float_value < min_value:
            raise ValidationError(f"{field_name}: Value {float_value} below minimum {min_value}")

        if max_value is not None and float_value > max_value:
            raise ValidationError(f"{field_name}: Value {float_value} above maximum {max_value}")

        return float_value

    def validate_boolean(self, value: Any, field_name: str = "boolean") -> bool:
        """Validate boolean input.

        Args:
            value: Input value to validate
            field_name: Field name for error messages

        Returns:
            Validated boolean

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value_lower = value.lower().strip()
            if value_lower in ('true', '1', 'yes', 'on', 'enabled'):
                return True
            elif value_lower in ('false', '0', 'no', 'off', 'disabled'):
                return False
            else:
                raise ValidationError(f"{field_name}: Invalid boolean value '{value}'")

        if isinstance(value, (int, float)):
            return bool(value)

        raise ValidationError(f"{field_name}: Cannot convert to boolean")

    def validate_list(
        self,
        value: Any,
        min_size: int = 0,
        max_size: int | None = None,
        item_validator: Callable | None = None,
        field_name: str = "list",
    ) -> list[Any]:
        """Validate list input.

        Args:
            value: Input value to validate
            min_size: Minimum list size
            max_size: Maximum list size
            item_validator: Validator function for list items
            field_name: Field name for error messages

        Returns:
            Validated list

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (list, tuple)):
            raise ValidationError(f"{field_name}: Value must be a list or tuple")

        list_value = list(value)
        actual_max = max_size or self.max_collection_size

        if len(list_value) < min_size:
            raise ValidationError(f"{field_name}: List too small (minimum {min_size} items)")
        if len(list_value) > actual_max:
            raise ValidationError(f"{field_name}: List too large (maximum {actual_max} items)")

        # Validate items if validator provided
        if item_validator:
            validated_items = []
            for i, item in enumerate(list_value):
                try:
                    validated_item = item_validator(item)
                    validated_items.append(validated_item)
                except ValidationError as e:
                    raise ValidationError(f"{field_name}[{i}]: {e}")
            return validated_items

        return list_value

    def validate_dict(
        self,
        value: Any,
        required_keys: set[str] | None = None,
        optional_keys: set[str] | None = None,
        key_validator: Callable | None = None,
        value_validator: Callable | None = None,
        field_name: str = "dictionary",
    ) -> dict[str, Any]:
        """Validate dictionary input.

        Args:
            value: Input value to validate
            required_keys: Set of required keys
            optional_keys: Set of optional keys
            key_validator: Validator function for dictionary keys
            value_validator: Validator function for dictionary values
            field_name: Field name for error messages

        Returns:
            Validated dictionary

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, dict):
            raise ValidationError(f"{field_name}: Value must be a dictionary")

        dict_value = dict(value)

        if len(dict_value) > self.max_collection_size:
            raise ValidationError(f"{field_name}: Dictionary too large (maximum {self.max_collection_size} items)")

        # Check required keys
        if required_keys:
            missing_keys = required_keys - set(dict_value.keys())
            if missing_keys:
                raise ValidationError(f"{field_name}: Missing required keys: {missing_keys}")

        # Check for unexpected keys
        if required_keys is not None or optional_keys is not None:
            allowed_keys = (required_keys or set()) | (optional_keys or set())
            unexpected_keys = set(dict_value.keys()) - allowed_keys
            if unexpected_keys:
                raise ValidationError(f"{field_name}: Unexpected keys: {unexpected_keys}")

        # Validate keys and values
        validated_dict = {}
        for key, val in dict_value.items():
            # Validate key
            validated_key = key
            if key_validator:
                try:
                    validated_key = key_validator(key)
                except ValidationError as e:
                    raise ValidationError(f"{field_name} key '{key}': {e}")

            # Validate value
            validated_value = val
            if value_validator:
                try:
                    validated_value = value_validator(val)
                except ValidationError as e:
                    raise ValidationError(f"{field_name}['{key}']: {e}")

            validated_dict[validated_key] = validated_value

        return validated_dict

    def validate_path(
        self,
        value: Any,
        allow_absolute: bool = False,
        allow_parent_traversal: bool = False,
        allowed_extensions: set[str] | None = None,
        field_name: str = "path",
    ) -> Path:
        """Validate file path input.

        Args:
            value: Input value to validate
            allow_absolute: Whether to allow absolute paths
            allow_parent_traversal: Whether to allow parent directory traversal
            allowed_extensions: Set of allowed file extensions
            field_name: Field name for error messages

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (str, Path, PurePath)):
            raise ValidationError(f"{field_name}: Path must be a string or Path object")

        path_str = str(value)

        # Basic string validation
        self.validate_string(path_str, max_length=1024, field_name=f"{field_name} string")

        try:
            path_obj = Path(path_str)
        except Exception:
            raise ValidationError(f"{field_name}: Invalid path format")

        # Security checks
        if not allow_absolute and path_obj.is_absolute():
            raise ValidationError(f"{field_name}: Absolute paths not allowed")

        if not allow_parent_traversal:
            # Check for parent directory traversal
            parts = path_obj.parts
            if '..' in parts:
                raise ValidationError(f"{field_name}: Parent directory traversal not allowed")

        # Check for dangerous patterns
        dangerous_patterns = self.DANGEROUS_CHARACTERS['path_traversal']
        for pattern in dangerous_patterns:
            if pattern in path_str:
                raise ValidationError(f"{field_name}: Contains dangerous path pattern: {pattern}")

        # Validate file extension
        if allowed_extensions and path_obj.suffix:
            if path_obj.suffix.lower() not in allowed_extensions:
                raise ValidationError(f"{field_name}: File extension '{path_obj.suffix}' not allowed")

        return path_obj

    def validate_email(self, value: Any, field_name: str = "email") -> str:
        """Validate email address.

        Args:
            value: Input value to validate
            field_name: Field name for error messages

        Returns:
            Validated email address

        Raises:
            ValidationError: If validation fails
        """
        email_str = self.validate_string(value, min_length=3, max_length=254, field_name=field_name)

        if not self.EMAIL_PATTERN.match(email_str):
            raise ValidationError(f"{field_name}: Invalid email format")

        return email_str.lower()

    def validate_uuid(self, value: Any, field_name: str = "uuid") -> str:
        """Validate UUID string.

        Args:
            value: Input value to validate
            field_name: Field name for error messages

        Returns:
            Validated UUID string

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(value, uuid.UUID):
            return str(value)

        uuid_str = self.validate_string(value, min_length=36, max_length=36, field_name=field_name)

        if not self.UUID_PATTERN.match(uuid_str):
            raise ValidationError(f"{field_name}: Invalid UUID format")

        # Validate using UUID constructor
        try:
            uuid_obj = uuid.UUID(uuid_str)
            return str(uuid_obj)
        except ValueError:
            raise ValidationError(f"{field_name}: Invalid UUID value")

    def validate_json(
        self,
        value: Any,
        schema: dict[str, Any] | None = None,
        field_name: str = "json",
    ) -> dict[str, Any]:
        """Validate JSON input.

        Args:
            value: Input value to validate
            schema: Optional JSON schema for validation
            field_name: Field name for error messages

        Returns:
            Validated JSON object

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(value, dict):
            json_obj = value
        elif isinstance(value, str):
            try:
                json_obj = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValidationError(f"{field_name}: Invalid JSON format: {e}")
        else:
            raise ValidationError(f"{field_name}: Value must be a dictionary or JSON string")

        # TODO: Implement JSON schema validation if needed
        if schema:
            logger.warning("JSON schema validation not implemented yet")

        return json_obj

    def _check_security_patterns(self, value: str, field_name: str) -> None:
        """Check for dangerous security patterns in string."""
        if not self.strict_mode:
            return

        value_lower = value.lower()

        # Check for SQL injection patterns
        for pattern in self.DANGEROUS_CHARACTERS['sql_injection']:
            if pattern.lower() in value_lower:
                logger.warning(f"{field_name}: Potential SQL injection pattern detected: {pattern}")

        # Check for command injection patterns
        for pattern in self.DANGEROUS_CHARACTERS['command_injection']:
            if pattern in value:
                logger.warning(f"{field_name}: Potential command injection pattern detected: {pattern}")

        # Check for script injection patterns
        for pattern in self.DANGEROUS_CHARACTERS['script_injection']:
            if pattern.lower() in value_lower:
                if not self.allow_html:
                    raise ValidationError(f"{field_name}: Script injection pattern detected: {pattern}")
                else:
                    logger.warning(f"{field_name}: Script injection pattern detected but HTML allowed: {pattern}")

    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        if not self.allow_html:
            # Escape HTML characters
            value = html.escape(value, quote=True)

        # Remove null bytes and other control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')

        # URL decode to prevent encoding attacks
        try:
            decoded = urllib.parse.unquote(value)
            if decoded != value:
                # Re-check for dangerous patterns after URL decoding
                self._check_security_patterns(decoded, "url_decoded")
        except Exception:
            pass

        return value


# Convenience functions for common validations
def validate_collection_name(name: Any) -> str:
    """Validate collection name with strict security rules."""
    validator = InputValidator(strict_mode=True)
    return validator.validate_string(
        name,
        min_length=1,
        max_length=63,
        pattern=validator.ALPHANUMERIC_DASH_UNDERSCORE,
        field_name="collection_name",
    )


def validate_document_id(doc_id: Any) -> str:
    """Validate document ID with security checks."""
    validator = InputValidator(strict_mode=True)
    return validator.validate_string(
        doc_id,
        min_length=1,
        max_length=128,
        field_name="document_id",
    )


def validate_search_query(query: Any) -> str:
    """Validate search query with injection protection."""
    validator = InputValidator(strict_mode=True)
    return validator.validate_string(
        query,
        min_length=0,
        max_length=1024,
        field_name="search_query",
    )


def validate_metadata_dict(metadata: Any) -> dict[str, Any]:
    """Validate metadata dictionary with security checks."""
    validator = InputValidator(strict_mode=True)

    def validate_metadata_key(key):
        return validator.validate_string(
            key,
            min_length=1,
            max_length=128,
            pattern=validator.ALPHANUMERIC_DASH_UNDERSCORE,
            field_name="metadata_key",
        )

    def validate_metadata_value(value):
        if isinstance(value, (str, int, float, bool)):
            return value
        elif value is None:
            return None
        else:
            return validator.validate_string(str(value), max_length=1024, field_name="metadata_value")

    return validator.validate_dict(
        metadata,
        key_validator=validate_metadata_key,
        value_validator=validate_metadata_value,
        field_name="metadata",
    )
