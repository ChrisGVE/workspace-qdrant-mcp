"""Compatibility wrapper for common.core.memory."""

from common.core.memory import *  # noqa: F401,F403

# Legacy alias for older imports.
try:
    DocumentMemory = MemoryManager  # type: ignore[name-defined]
except Exception:
    pass

