"""
Core module compatibility layer.

This module provides backward compatibility by importing common.core modules
into the workspace_qdrant_mcp.core namespace for tests and legacy code.
"""

# Import all core modules from common for backward compatibility
try:
    from common.core.config import *
    from common.core.embeddings import *
    from common.core.client import *
    from common.core.hybrid_search import *
    from common.core.memory import *
    from common.core.collections import *
    from common.core.sparse_vectors import *
except ImportError as e:
    # If common.core is not available, provide a helpful error message
    import warnings
    warnings.warn(
        f"Failed to import common.core modules: {e}. "
        "Make sure src/python is in PYTHONPATH and common package is available.",
        ImportWarning
    )
    # Re-raise to maintain the original error behavior
    raise