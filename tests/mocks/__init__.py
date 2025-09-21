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
from . import error_injection
from . import qdrant_mocks
from . import filesystem_mocks
from . import grpc_mocks
from . import network_mocks
from . import lsp_mocks
from . import embedding_mocks
from . import external_service_mocks

# Import key classes and functions
from .error_injection import (
    ErrorInjector,
    FailureScenarios,
    ErrorModeManager,
    create_error_injector,
    create_error_manager,
)

from .qdrant_mocks import (
    EnhancedQdrantClientMock,
    QdrantErrorInjector,
    create_enhanced_qdrant_client,
    create_basic_qdrant_mock,
    create_failing_qdrant_mock,
    create_realistic_qdrant_mock,
)

from .filesystem_mocks import (
    FileSystemMock,
    FileWatcherMock,
    DirectoryOperationMock,
    FileSystemErrorInjector,
    create_filesystem_mock,
    create_file_watcher_mock,
    create_directory_operation_mock,
)

from .grpc_mocks import (
    GRPCClientMock,
    GRPCServerMock,
    DaemonCommunicationMock,
    GRPCErrorInjector,
    create_grpc_mock,
    create_basic_grpc_client,
    create_failing_grpc_client,
    create_realistic_daemon_communication,
)

from .network_mocks import (
    NetworkClientMock,
    HTTPRequestMock,
    ConnectionFailureMock,
    NetworkErrorInjector,
    create_network_mock,
    create_basic_network_client,
    create_failing_network_client,
    create_realistic_network_client,
)

from .lsp_mocks import (
    LSPServerMock,
    LSPMetadataExtractorMock,
    LanguageDetectorMock,
    LSPErrorInjector,
    create_lsp_mock,
    create_basic_lsp_server,
    create_failing_lsp_server,
    create_realistic_metadata_extractor,
)

from .embedding_mocks import (
    EnhancedEmbeddingServiceMock,
    EmbeddingGeneratorMock,
    FastEmbedMock,
    EmbeddingErrorInjector,
    create_embedding_mock,
    create_basic_embedding_service,
    create_failing_embedding_service,
    create_realistic_embedding_service,
)

from .external_service_mocks import (
    ExternalServiceMock,
    ThirdPartyAPIMock,
    ServiceUnavailableMock,
    ExternalServiceErrorInjector,
    create_external_service_mock,
    create_basic_external_service,
    create_failing_external_service,
    create_openai_api_mock,
    create_unavailable_service,
)

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