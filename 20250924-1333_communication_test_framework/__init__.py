"""
Comprehensive Inter-Component Communication Testing Framework

This framework provides dummy data simulation and mock services for testing
all communication patterns in the workspace-qdrant-mcp system:

1. MCP-to-daemon communication (Python -> gRPC -> Rust)
2. Daemon-to-MCP communication (Rust -> gRPC -> Python)
3. CLI-to-daemon communication (CLI -> gRPC -> Rust)

Components:
- dummy_data/: Data generators for all message types
- mock_services/: Mock implementations of gRPC services
- fixtures/: Test fixtures for different scenarios
- validators/: Protocol validation utilities
- edge_cases/: Error condition and failure scenario testing
- integration/: Multi-component integration tests

Usage:
    from 20250924-1333_communication_test_framework import (
        DummyDataGenerator,
        MockGrpcServices,
        CommunicationTestSuite
    )
"""

__version__ = "1.0.0"

# Import main components when framework is loaded
from .dummy_data import DummyDataGenerator
from .mock_services import MockGrpcServices
from .validators import ProtocolValidator
from .edge_cases import EdgeCaseSimulator
from .integration import CommunicationTestSuite

__all__ = [
    "DummyDataGenerator",
    "MockGrpcServices",
    "ProtocolValidator",
    "EdgeCaseSimulator",
    "CommunicationTestSuite"
]