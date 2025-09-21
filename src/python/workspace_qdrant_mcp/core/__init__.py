"""
Core module compatibility layer.

This module provides backward compatibility by importing common.core modules
into the workspace_qdrant_mcp.core namespace for tests and legacy code.
"""

# Import all core modules from common for backward compatibility
try:
    from python.common.core.config import *
    from python.common.core.embeddings import *
    from python.common.core.client import *
    from python.common.core.hybrid_search import *
    from python.common.core.memory import *
    from python.common.core.collections import *
    from python.common.core.sparse_vectors import *
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