"""
FastMCP Optimization Module

This module provides enhanced FastMCP framework integration with optimizations:
- Lazy tool loading and efficient registration
- Stdio protocol compression and batching
- Performance monitoring and metrics
- Streaming response support
- Memory usage optimization

The optimizations are designed to improve:
1. Server startup time (lazy loading)
2. Communication efficiency (compression, batching)
3. Memory usage (efficient tool management)
4. Large response handling (streaming)
5. Performance monitoring (built-in metrics)
"""

from .complete_fastmcp_optimization import (
    FastMCPOptimizer,
    FrameworkMetrics,
    MessageCompressor,
    OptimizedFastMCPApp,
    OptimizedToolRegistry,
    OptimizedWorkspaceServer,
    StreamingStdioProtocol,
)

__all__ = [
    "OptimizedFastMCPApp",
    "FastMCPOptimizer",
    "OptimizedWorkspaceServer",
    "StreamingStdioProtocol",
    "FrameworkMetrics",
    "MessageCompressor",
    "OptimizedToolRegistry"
]
