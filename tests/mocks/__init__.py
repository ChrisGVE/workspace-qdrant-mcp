"""
Centralized mocking infrastructure for workspace-qdrant-mcp tests.

This module provides comprehensive mocking capabilities for all external dependencies,
enabling reliable testing without requiring external services.

Key Features:
- Enhanced Qdrant client mocking with error scenarios
- File system operation mocking
- gRPC communication mocking
- Network operation mocking
- LSP server communication mocking
- Embedding service mocking
- Configurable failure injection
- Realistic behavior simulation

Usage:
    from tests.mocks import qdrant_mocks, filesystem_mocks, grpc_mocks

    # Use in test fixtures
    @pytest.fixture
    def mock_client():
        return qdrant_mocks.create_enhanced_qdrant_client()
"""

from .qdrant_mocks import *
from .filesystem_mocks import *
from .grpc_mocks import *
from .network_mocks import *
from .lsp_mocks import *
from .embedding_mocks import *
from .external_service_mocks import *
from .error_injection import *

__all__ = [
    # Qdrant mocks
    "EnhancedQdrantClientMock",
    "QdrantErrorInjector",
    "create_enhanced_qdrant_client",

    # Filesystem mocks
    "FileSystemMock",
    "FileWatcherMock",
    "DirectoryOperationMock",
    "create_filesystem_mock",

    # gRPC mocks
    "GRPCClientMock",
    "GRPCServerMock",
    "DaemonCommunicationMock",
    "create_grpc_mock",

    # Network mocks
    "NetworkClientMock",
    "HTTPRequestMock",
    "ConnectionFailureMock",
    "create_network_mock",

    # LSP mocks
    "LSPServerMock",
    "LSPMetadataExtractorMock",
    "LanguageDetectorMock",
    "create_lsp_mock",

    # Embedding mocks
    "EnhancedEmbeddingServiceMock",
    "EmbeddingGeneratorMock",
    "FastEmbedMock",
    "create_embedding_mock",

    # External service mocks
    "ExternalServiceMock",
    "ThirdPartyAPIMock",
    "ServiceUnavailableMock",
    "create_external_service_mock",

    # Error injection
    "ErrorInjector",
    "FailureScenarios",
    "ErrorModeManager",
    "create_error_injector",
]