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

# Import all mock modules
from . import (
    embedding_mocks,
    error_injection,
    external_service_mocks,
    filesystem_mocks,
    grpc_mocks,
    lsp_mocks,
    network_mocks,
    qdrant_mocks,
)
from .embedding_mocks import (
    EmbeddingErrorInjector,
    EmbeddingGeneratorMock,
    EnhancedEmbeddingServiceMock,
    FastEmbedMock,
    create_basic_embedding_service,
    create_embedding_mock,
    create_failing_embedding_service,
    create_realistic_embedding_service,
)

# Import key classes and functions
from .error_injection import (
    ErrorInjector,
    ErrorModeManager,
    FailureScenarios,
    create_error_injector,
    create_error_manager,
)
from .external_service_mocks import (
    ExternalServiceErrorInjector,
    ExternalServiceMock,
    ServiceUnavailableMock,
    ThirdPartyAPIMock,
    create_basic_external_service,
    create_external_service_mock,
    create_failing_external_service,
    create_openai_api_mock,
    create_unavailable_service,
)
from .filesystem_mocks import (
    DirectoryOperationMock,
    FileSystemErrorInjector,
    FileSystemMock,
    create_directory_operation_mock,
    create_filesystem_mock,
)
from .grpc_mocks import (
    DaemonCommunicationMock,
    GRPCClientMock,
    GRPCErrorInjector,
    GRPCServerMock,
    create_basic_grpc_client,
    create_failing_grpc_client,
    create_grpc_mock,
    create_realistic_daemon_communication,
)
from .lsp_mocks import (
    LanguageDetectorMock,
    LSPErrorInjector,
    LSPMetadataExtractorMock,
    LSPServerMock,
    create_basic_lsp_server,
    create_failing_lsp_server,
    create_lsp_mock,
    create_realistic_metadata_extractor,
)
from .network_mocks import (
    ConnectionFailureMock,
    HTTPRequestMock,
    NetworkClientMock,
    NetworkErrorInjector,
    create_basic_network_client,
    create_failing_network_client,
    create_network_mock,
    create_realistic_network_client,
)
from .qdrant_mocks import (
    EnhancedQdrantClientMock,
    QdrantErrorInjector,
    create_basic_qdrant_mock,
    create_enhanced_qdrant_client,
    create_failing_qdrant_mock,
    create_realistic_qdrant_mock,
)

__all__ = [
    # Qdrant mocks
    "EnhancedQdrantClientMock",
    "QdrantErrorInjector",
    "create_enhanced_qdrant_client",

    # Filesystem mocks
    "FileSystemMock",
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
