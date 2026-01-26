"""Minimal YAML config helpers for compatibility imports."""

from common.core.config import load_config


def save_config(*_args, **_kwargs) -> bool:
    """Placeholder save_config implementation."""
    return True


def validate_config(*_args, **_kwargs) -> list[str]:
    """Placeholder validate_config implementation."""
    return []

