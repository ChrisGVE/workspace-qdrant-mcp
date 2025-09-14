"""
LSP Communication Protocol Implementation

This module provides a robust asyncio-based LSP client implementation supporting
JSON-RPC communication, request/response handling, and streaming notification processing.
"""

import asyncio
import json
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
from weakref import WeakSet


from .error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from loguru import logger

# logger imported from loguru


class ConnectionState(Enum):
    """LSP client connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    DISCONNECTING = "disconnecting"


class CommunicationMode(Enum):
    """LSP communication modes"""
    STDIO = "stdio"
    TCP = "tcp"
    MANUAL = "manual"  # Manual stream provision


class CircuitBreakerState(Enum):
    """Circuit breaker states for error recovery"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests due to failures  
    HALF_OPEN = "half_open"  # Testing if service recovered


class LspError(WorkspaceError):
    """LSP-specific errors"""

    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        context.update({"server_name": server_name, "method": method})
        
        # Use provided category or default to IPC
        category = kwargs.pop("category", ErrorCategory.IPC)
        
        super().__init__(
            message,
            category=category,
            severity=kwargs.pop("severity", ErrorSeverity.MEDIUM),
            retryable=kwargs.pop("retryable", True),
            context=context,
            **kwargs,
        )


class LspTimeoutError(LspError):
    """LSP request timeout error"""

    def __init__(self, method: str, timeout: float, **kwargs):
        super().__init__(
            f"LSP request '{method}' timed out after {timeout}s",
            method=method,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            context={"timeout": timeout},
            **kwargs,
        )


class LspProtocolError(LspError):
    """LSP protocol violation error"""

    def __init__(self, message: str, raw_data: Optional[str] = None, **kwargs):
        super().__init__(
            f"LSP protocol error: {message}",
            severity=ErrorSeverity.HIGH,
            retryable=False,
            context={"raw_data": raw_data} if raw_data else {},
            **kwargs,
        )


@dataclass
class JsonRpcError:
    """JSON-RPC error object"""
    code: int
    message: str
    data: Optional[Any] = None


@dataclass
class JsonRpcRequest:
    """JSON-RPC request message"""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    method: str = ""
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc, "id": self.id, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        return result


@dataclass
class JsonRpcResponse:
    """JSON-RPC response message"""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.result is not None:
            result["result"] = self.result
        if self.error is not None:
            result["error"] = {
                "code": self.error.code,
                "message": self.error.message,
            }
            if self.error.data is not None:
                result["error"]["data"] = self.error.data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcResponse":
        error = None
        if "error" in data:
            error_data = data["error"]
            error = JsonRpcError(
                code=error_data["code"],
                message=error_data["message"],
                data=error_data.get("data"),
            )
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=error,
        )


@dataclass
class JsonRpcNotification:
    """JSON-RPC notification message"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcNotification":
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data["method"],
            params=data.get("params"),
        )


@dataclass
class PendingRequest:
    """Tracking information for pending LSP requests"""
    future: asyncio.Future
    method: str
    created_at: float
    timeout: float
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker error recovery"""
    failure_threshold: int = 5           # Failures before opening circuit
    success_threshold: int = 3           # Successes to close circuit from half-open
    timeout_seconds: float = 60.0        # Time before trying half-open from open
    health_check_interval: float = 30.0  # Interval for health checks


@dataclass
class RetryConfig:
    """Configuration for request retry logic"""
    max_attempts: int = 3               # Maximum retry attempts
    base_delay: float = 1.0             # Base delay between retries
    max_delay: float = 30.0             # Maximum delay between retries
    exponential_backoff: bool = True    # Use exponential backoff
    jitter: bool = True                 # Add random jitter to delays
    retryable_errors: List[str] = field(default_factory=lambda: [
        "timeout", "connection", "network", "server_error"
    ])


@dataclass
class ClientCapabilities:
    """LSP client capabilities for initialization"""
    workspace: Optional[Dict[str, Any]] = None
    textDocument: Optional[Dict[str, Any]] = None
    window: Optional[Dict[str, Any]] = None
    general: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON-RPC"""
        result = {}
        if self.workspace is not None:
            result["workspace"] = self.workspace
        if self.textDocument is not None:
            result["textDocument"] = self.textDocument
        if self.window is not None:
            result["window"] = self.window
        if self.general is not None:
            result["general"] = self.general
        return result


@dataclass
class InitializeParams:
    """LSP initialize request parameters"""
    processId: Optional[int] = None
    clientInfo: Optional[Dict[str, str]] = None
    locale: Optional[str] = None
    rootPath: Optional[str] = None
    rootUri: Optional[str] = None
    capabilities: Optional[ClientCapabilities] = None
    initializationOptions: Optional[Any] = None
    workspaceFolders: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON-RPC"""
        result = {}
        if self.processId is not None:
            result["processId"] = self.processId
        if self.clientInfo is not None:
            result["clientInfo"] = self.clientInfo
        if self.locale is not None:
            result["locale"] = self.locale
        if self.rootPath is not None:
            result["rootPath"] = self.rootPath
        if self.rootUri is not None:
            result["rootUri"] = self.rootUri
        if self.capabilities is not None:
            result["capabilities"] = self.capabilities.to_dict()
        if self.initializationOptions is not None:
            result["initializationOptions"] = self.initializationOptions
        if self.workspaceFolders is not None:
            result["workspaceFolders"] = self.workspaceFolders
        return result


@dataclass
class ServerCapabilities:
    """LSP server capabilities from initialization response"""
    raw_data: Dict[str, Any]

    def supports_hover(self) -> bool:
        """Check if server supports textDocument/hover"""
        return self.raw_data.get("hoverProvider", False)

    def supports_definition(self) -> bool:
        """Check if server supports textDocument/definition"""
        return self.raw_data.get("definitionProvider", False)

    def supports_references(self) -> bool:
        """Check if server supports textDocument/references"""
        return self.raw_data.get("referencesProvider", False)

    def supports_document_symbol(self) -> bool:
        """Check if server supports textDocument/documentSymbol"""
        return self.raw_data.get("documentSymbolProvider", False)

    def supports_workspace_symbol(self) -> bool:
        """Check if server supports workspace/symbol"""
        return self.raw_data.get("workspaceSymbolProvider", False)

    def supports_completion(self) -> bool:
        """Check if server supports textDocument/completion"""
        return self.raw_data.get("completionProvider") is not None

    def supports_diagnostics(self) -> bool:
        """Check if server provides diagnostics"""
        return self.raw_data.get("textDocumentSync") is not None

    def get_text_document_sync(self) -> Dict[str, Any]:
        """Get textDocumentSync capabilities"""
        sync = self.raw_data.get("textDocumentSync")
        if isinstance(sync, int):
            return {"change": sync}
        elif isinstance(sync, dict):
            return sync
        else:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Get raw capabilities data"""
        return self.raw_data


class AsyncioLspClient:
    """
    Asyncio-based LSP client with JSON-RPC communication support.
    
    This client provides the foundation for LSP communication with:
    - Request correlation and timeout handling
    - Notification routing and handler registration
    - Connection state management
    - Protocol-compliant message parsing
    """

    def __init__(
        self,
        server_name: str = "lsp-server",
        request_timeout: float = 30.0,
        max_pending_requests: int = 100,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self._server_name = server_name
        self._request_timeout = request_timeout
        self._max_pending_requests = max_pending_requests
        
        # Error recovery configuration
        self._circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self._retry_config = retry_config or RetryConfig()
        
        # Connection management
        self._connection_state = ConnectionState.DISCONNECTED
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        
        # Request correlation
        self._pending_requests: Dict[str, PendingRequest] = {}
        self._request_cleanup_task: Optional[asyncio.Task] = None
        
        # Notification handling
        self._notification_handlers: Dict[str, List[Callable]] = {}
        self._global_handlers: WeakSet[Callable] = WeakSet()
        
        # LSP initialization state
        self._initialized = False
        self._server_capabilities: Optional[ServerCapabilities] = None
        
        # Communication mode management
        self._communication_mode = CommunicationMode.MANUAL
        self._server_process: Optional[asyncio.subprocess.Process] = None
        self._tcp_host: Optional[str] = None
        self._tcp_port: Optional[int] = None
        
        # Circuit breaker state
        self._circuit_breaker_state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._circuit_open_time = 0.0
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_successful_request = time.time()
        
        # Background tasks
        self._message_reader_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            "LSP client initialized",
            server_name=self._server_name,
            timeout=self._request_timeout,
        )

    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state"""
        return self._connection_state

    @property
    def server_name(self) -> str:
        """Get server name"""
        return self._server_name

    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._connection_state == ConnectionState.CONNECTED

    @property
    def is_initialized(self) -> bool:
        """Check if LSP client has been initialized"""
        return self._initialized

    @property
    def server_capabilities(self) -> Optional[ServerCapabilities]:
        """Get server capabilities (available after initialization)"""
        return self._server_capabilities

    @property
    def communication_mode(self) -> CommunicationMode:
        """Get current communication mode"""
        return self._communication_mode

    @property
    def circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self._circuit_breaker_state

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (blocking requests)"""
        return self._circuit_breaker_state == CircuitBreakerState.OPEN

    async def connect_stdio(
        self,
        server_command: List[str],
        server_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Connect to an LSP server via stdio subprocess.
        
        Args:
            server_command: Command and arguments to start the LSP server
            server_args: Additional arguments for the server
            cwd: Working directory for the server process
            env: Environment variables for the server process
        """
        if self._connection_state != ConnectionState.DISCONNECTED:
            raise LspError(
                "Cannot connect: client is not in disconnected state",
                server_name=self._server_name,
                context={"current_state": self._connection_state.value},
            )

        command = server_command + (server_args or [])
        logger.info(
            "Starting LSP server via stdio",
            server_name=self._server_name,
            command=command,
        )
        
        self._connection_state = ConnectionState.CONNECTING
        
        try:
            # Start the LSP server process
            self._server_process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            
            if not self._server_process.stdin or not self._server_process.stdout:
                raise LspError(
                    "Failed to create stdio pipes for LSP server",
                    server_name=self._server_name,
                )
            
            # Connect using the process streams
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await asyncio.get_event_loop().connect_read_pipe(
                lambda: protocol, self._server_process.stdout
            )
            
            writer = asyncio.StreamWriter(
                self._server_process.stdin,
                protocol,
                reader,
                asyncio.get_event_loop(),
            )
            
            self._communication_mode = CommunicationMode.STDIO
            await self._connect_streams(reader, writer)
            
            logger.info(
                "Connected to LSP server via stdio",
                server_name=self._server_name,
                pid=self._server_process.pid,
            )
            
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            if self._server_process:
                try:
                    self._server_process.terminate()
                    await self._server_process.wait()
                except:
                    pass
                self._server_process = None
            
            logger.error(
                "Failed to connect to LSP server via stdio",
                server_name=self._server_name,
                command=command,
                error=str(e),
            )
            raise LspError(
                f"Failed to connect to LSP server via stdio: {e}",
                server_name=self._server_name,
                cause=e,
            ) from e

    async def connect_tcp(
        self,
        host: str = "localhost",
        port: int = 9257,
        connect_timeout: float = 10.0,
    ) -> None:
        """
        Connect to an LSP server via TCP socket.
        
        Args:
            host: Server hostname or IP address
            port: Server port number  
            connect_timeout: Connection timeout in seconds
        """
        if self._connection_state != ConnectionState.DISCONNECTED:
            raise LspError(
                "Cannot connect: client is not in disconnected state",
                server_name=self._server_name,
                context={"current_state": self._connection_state.value},
            )

        logger.info(
            "Connecting to LSP server via TCP",
            server_name=self._server_name,
            host=host,
            port=port,
        )
        
        self._connection_state = ConnectionState.CONNECTING
        
        try:
            # Connect to TCP socket
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=connect_timeout,
            )
            
            self._tcp_host = host
            self._tcp_port = port
            self._communication_mode = CommunicationMode.TCP
            
            await self._connect_streams(reader, writer)
            
            logger.info(
                "Connected to LSP server via TCP",
                server_name=self._server_name,
                host=host,
                port=port,
            )
            
        except asyncio.TimeoutError:
            self._connection_state = ConnectionState.ERROR
            raise LspTimeoutError(
                "TCP connection",
                connect_timeout,
                server_name=self._server_name,
            )
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            logger.error(
                "Failed to connect to LSP server via TCP",
                server_name=self._server_name,
                host=host,
                port=port,
                error=str(e),
            )
            raise LspError(
                f"Failed to connect to LSP server via TCP: {e}",
                server_name=self._server_name,
                cause=e,
            ) from e

    async def connect(
        self, 
        reader: asyncio.StreamReader, 
        writer: asyncio.StreamWriter
    ) -> None:
        """
        Connect the LSP client to a server via manually provided streams.
        
        Args:
            reader: StreamReader for receiving messages
            writer: StreamWriter for sending messages
        """
        if self._connection_state != ConnectionState.DISCONNECTED:
            raise LspError(
                "Cannot connect: client is not in disconnected state",
                server_name=self._server_name,
                context={"current_state": self._connection_state.value},
            )

        logger.info("Connecting to LSP server via manual streams", server_name=self._server_name)
        self._connection_state = ConnectionState.CONNECTING
        
        try:
            self._communication_mode = CommunicationMode.MANUAL
            await self._connect_streams(reader, writer)
            logger.info("LSP client connected via manual streams", server_name=self._server_name)
            
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            logger.error(
                "Failed to connect LSP client via manual streams",
                server_name=self._server_name,
                error=str(e),
            )
            raise LspError(
                f"Failed to connect to LSP server via manual streams: {e}",
                server_name=self._server_name,
                cause=e,
            ) from e

    async def _connect_streams(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Internal method to establish connection using provided streams.
        
        Args:
            reader: StreamReader for receiving messages
            writer: StreamWriter for sending messages
        """
        self._reader = reader
        self._writer = writer
        self._shutdown_event.clear()
        
        # Start background tasks
        self._message_reader_task = asyncio.create_task(
            self._message_reader_loop()
        )
        self._request_cleanup_task = asyncio.create_task(
            self._request_cleanup_loop()
        )
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )
        
        # Reset circuit breaker on successful connection
        self._reset_circuit_breaker()
        
        self._connection_state = ConnectionState.CONNECTED

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state"""
        self._circuit_breaker_state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._circuit_open_time = 0.0
        logger.info("Circuit breaker reset to closed state", server_name=self._server_name)

    def _record_success(self) -> None:
        """Record successful request for circuit breaker"""
        self._last_successful_request = time.time()
        
        if self._circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._circuit_breaker_config.success_threshold:
                self._reset_circuit_breaker()
                logger.info(
                    "Circuit breaker closed after successful recovery",
                    server_name=self._server_name,
                )
        elif self._circuit_breaker_state == CircuitBreakerState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)  # Gradually reduce failure count

    def _record_failure(self) -> None:
        """Record failed request for circuit breaker"""
        self._last_failure_time = time.time()
        
        if self._circuit_breaker_state == CircuitBreakerState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._circuit_breaker_config.failure_threshold:
                self._circuit_breaker_state = CircuitBreakerState.OPEN
                self._circuit_open_time = time.time()
                logger.warning(
                    "Circuit breaker opened due to failures",
                    server_name=self._server_name,
                    failure_count=self._failure_count,
                )
        elif self._circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            # Failed during half-open, go back to open
            self._circuit_breaker_state = CircuitBreakerState.OPEN
            self._circuit_open_time = time.time()
            self._success_count = 0
            logger.warning(
                "Circuit breaker reopened after failed recovery attempt",
                server_name=self._server_name,
            )

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit breaker state"""
        if self._circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True
        elif self._circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed to try half-open
            if time.time() - self._circuit_open_time >= self._circuit_breaker_config.timeout_seconds:
                self._circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
                logger.info(
                    "Circuit breaker moved to half-open for recovery test",
                    server_name=self._server_name,
                )
                return True
            return False
        else:  # HALF_OPEN
            return True

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry attempt"""
        if not self._retry_config.exponential_backoff:
            delay = self._retry_config.base_delay
        else:
            delay = min(
                self._retry_config.base_delay * (2 ** attempt),
                self._retry_config.max_delay
            )
        
        # Add jitter if enabled
        if self._retry_config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
            
        return delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable based on configuration"""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        for retryable in self._retry_config.retryable_errors:
            if retryable in error_type or retryable in error_message:
                return True
        
        # Special handling for LSP-specific errors
        if isinstance(error, (LspTimeoutError, LspProtocolError)):
            return True
        if isinstance(error, LspError) and error.retryable:
            return True
            
        return False

    async def disconnect(self) -> None:
        """Disconnect from the LSP server"""
        if self._connection_state == ConnectionState.DISCONNECTED:
            return

        logger.info("Disconnecting LSP client", server_name=self._server_name)
        self._connection_state = ConnectionState.DISCONNECTING
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel pending requests
        for request_id, pending in list(self._pending_requests.items()):
            if not pending.future.done():
                pending.future.cancel()
        self._pending_requests.clear()
        
        # Stop background tasks
        if self._message_reader_task and not self._message_reader_task.done():
            self._message_reader_task.cancel()
            try:
                await self._message_reader_task
            except asyncio.CancelledError:
                pass
                
        if self._request_cleanup_task and not self._request_cleanup_task.done():
            self._request_cleanup_task.cancel()
            try:
                await self._request_cleanup_task
            except asyncio.CancelledError:
                pass
                
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close writer
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass  # Writer might already be closed
        
        self._reader = None
        self._writer = None
        self._connection_state = ConnectionState.DISCONNECTED
        
        # Terminate server process if stdio mode
        if self._communication_mode == CommunicationMode.STDIO and self._server_process:
            logger.debug("Terminating LSP server process", server_name=self._server_name)
            try:
                self._server_process.terminate()
                await asyncio.wait_for(self._server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("LSP server process did not terminate gracefully, killing", 
                             server_name=self._server_name)
                self._server_process.kill()
                await self._server_process.wait()
            except Exception as e:
                logger.warning("Error terminating LSP server process", 
                             server_name=self._server_name, error=str(e))
            finally:
                self._server_process = None
        
        # Reset connection state
        self._communication_mode = CommunicationMode.MANUAL
        self._tcp_host = None
        self._tcp_port = None
        
        # Reset initialization state
        self._initialized = False
        self._server_capabilities = None
        
        logger.info("LSP client disconnected", server_name=self._server_name)

    async def initialize(
        self,
        root_uri: str,
        client_name: str = "workspace-qdrant-mcp",
        client_version: str = "1.0.0",
        workspace_folders: Optional[List[str]] = None,
        initialization_options: Optional[Dict[str, Any]] = None,
    ) -> ServerCapabilities:
        """
        Initialize the LSP client with capability negotiation.
        
        Args:
            root_uri: Root URI of the workspace (file:// format)
            client_name: Name of the client application
            client_version: Version of the client application
            workspace_folders: List of workspace folder paths
            initialization_options: Server-specific initialization options
            
        Returns:
            ServerCapabilities object with negotiated server capabilities
            
        Raises:
            LspError: If initialization fails or client is already initialized
        """
        if self._initialized:
            raise LspError(
                "LSP client is already initialized",
                server_name=self._server_name,
            )

        if not self.is_connected():
            raise LspError(
                "Cannot initialize: client is not connected",
                server_name=self._server_name,
            )

        logger.info(
            "Starting LSP initialization handshake",
            server_name=self._server_name,
            root_uri=root_uri,
        )

        # Build client capabilities
        client_capabilities = ClientCapabilities(
            workspace={
                "workspaceEdit": {
                    "documentChanges": True,
                    "resourceOperations": ["create", "rename", "delete"],
                },
                "workspaceFolders": True,
                "configuration": True,
                "didChangeConfiguration": {"dynamicRegistration": True},
                "didChangeWatchedFiles": {"dynamicRegistration": True},
                "symbol": {
                    "dynamicRegistration": True,
                    "symbolKind": {
                        "valueSet": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
                    },
                },
            },
            textDocument={
                "synchronization": {
                    "dynamicRegistration": True,
                    "willSave": True,
                    "willSaveWaitUntil": True,
                    "didSave": True,
                },
                "hover": {
                    "dynamicRegistration": True,
                    "contentFormat": ["markdown", "plaintext"],
                },
                "definition": {
                    "dynamicRegistration": True,
                    "linkSupport": True,
                },
                "references": {
                    "dynamicRegistration": True,
                    "context": {"includeDeclaration": True},
                },
                "documentSymbol": {
                    "dynamicRegistration": True,
                    "hierarchicalDocumentSymbolSupport": True,
                    "symbolKind": {
                        "valueSet": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
                    },
                },
                "completion": {
                    "dynamicRegistration": True,
                    "completionItem": {
                        "documentationFormat": ["markdown", "plaintext"],
                        "insertReplaceSupport": True,
                    },
                },
                "publishDiagnostics": {
                    "relatedInformation": True,
                    "versionSupport": True,
                    "tagSupport": {"valueSet": [1, 2]},
                },
            },
            window={
                "workDoneProgress": True,
                "showMessage": {
                    "messageActionItem": {"additionalPropertiesSupport": True}
                },
                "showDocument": {"support": True},
            },
            general={
                "regularExpressions": {"engine": "ECMAScript"},
                "markdown": {"parser": "marked"},
            },
        )

        # Prepare workspace folders
        workspace_folder_configs = []
        if workspace_folders:
            for folder_path in workspace_folders:
                folder_path = Path(folder_path).resolve()
                workspace_folder_configs.append({
                    "uri": f"file://{folder_path}",
                    "name": folder_path.name,
                })

        # Build initialize parameters
        initialize_params = InitializeParams(
            processId=os.getpid(),
            clientInfo={"name": client_name, "version": client_version},
            rootUri=root_uri,
            capabilities=client_capabilities,
            workspaceFolders=workspace_folder_configs if workspace_folder_configs else None,
            initializationOptions=initialization_options,
        )

        try:
            # Send initialize request
            logger.debug(
                "Sending initialize request",
                server_name=self._server_name,
                client_name=client_name,
            )

            result = await self.send_request(
                "initialize",
                initialize_params.to_dict(),
                timeout=60.0,  # Initialize can take longer
            )

            # Parse server capabilities
            server_caps_data = result.get("capabilities", {})
            self._server_capabilities = ServerCapabilities(server_caps_data)

            # Send initialized notification
            await self.send_notification("initialized", {})

            # Mark as initialized
            self._initialized = True

            logger.info(
                "LSP client initialization completed",
                server_name=self._server_name,
                hover_support=self._server_capabilities.supports_hover(),
                definition_support=self._server_capabilities.supports_definition(),
                references_support=self._server_capabilities.supports_references(),
                document_symbol_support=self._server_capabilities.supports_document_symbol(),
                workspace_symbol_support=self._server_capabilities.supports_workspace_symbol(),
            )

            return self._server_capabilities

        except Exception as e:
            logger.error(
                "LSP initialization failed",
                server_name=self._server_name,
                error=str(e),
            )
            if isinstance(e, LspError):
                raise
            raise LspError(
                f"LSP initialization failed: {e}",
                server_name=self._server_name,
                cause=e,
            ) from e

    async def hover(
        self,
        file_uri: str,
        line: int,
        character: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get hover information for a symbol at the given position.
        
        Args:
            file_uri: URI of the file (file:// format)
            line: Line number (0-based)
            character: Character position (0-based)
            
        Returns:
            Hover information or None if not available
        """
        if not self._server_capabilities or not self._server_capabilities.supports_hover():
            logger.debug(
                "Hover not supported by server",
                server_name=self._server_name,
            )
            return None

        params = {
            "textDocument": {"uri": file_uri},
            "position": {"line": line, "character": character},
        }
        
        try:
            result = await self.send_request("textDocument/hover", params)
            return result
        except LspError as e:
            logger.warning(
                "Hover request failed",
                server_name=self._server_name,
                file_uri=file_uri,
                line=line,
                character=character,
                error=str(e),
            )
            return None

    async def definition(
        self,
        file_uri: str,
        line: int,
        character: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get definition location(s) for a symbol at the given position.
        
        Args:
            file_uri: URI of the file (file:// format)
            line: Line number (0-based)
            character: Character position (0-based)
            
        Returns:
            List of Location objects or None if not available
        """
        if not self._server_capabilities or not self._server_capabilities.supports_definition():
            logger.debug(
                "Definition not supported by server",
                server_name=self._server_name,
            )
            return None

        params = {
            "textDocument": {"uri": file_uri},
            "position": {"line": line, "character": character},
        }
        
        try:
            result = await self.send_request("textDocument/definition", params)
            
            # Normalize result to always be a list
            if result is None:
                return None
            elif isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return result
            else:
                logger.warning(
                    "Unexpected definition result type",
                    server_name=self._server_name,
                    result_type=type(result).__name__,
                )
                return None
                
        except LspError as e:
            logger.warning(
                "Definition request failed",
                server_name=self._server_name,
                file_uri=file_uri,
                line=line,
                character=character,
                error=str(e),
            )
            return None

    async def references(
        self,
        file_uri: str,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get reference locations for a symbol at the given position.
        
        Args:
            file_uri: URI of the file (file:// format)
            line: Line number (0-based)
            character: Character position (0-based)
            include_declaration: Whether to include the symbol declaration
            
        Returns:
            List of Location objects or None if not available
        """
        if not self._server_capabilities or not self._server_capabilities.supports_references():
            logger.debug(
                "References not supported by server",
                server_name=self._server_name,
            )
            return None

        params = {
            "textDocument": {"uri": file_uri},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_declaration},
        }
        
        try:
            result = await self.send_request("textDocument/references", params)
            return result if isinstance(result, list) else None
        except LspError as e:
            logger.warning(
                "References request failed",
                server_name=self._server_name,
                file_uri=file_uri,
                line=line,
                character=character,
                error=str(e),
            )
            return None

    async def document_symbol(
        self,
        file_uri: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get document symbols (outline) for the given file.
        
        Args:
            file_uri: URI of the file (file:// format)
            
        Returns:
            List of DocumentSymbol or SymbolInformation objects or None if not available
        """
        if not self._server_capabilities or not self._server_capabilities.supports_document_symbol():
            logger.debug(
                "Document symbol not supported by server",
                server_name=self._server_name,
            )
            return None

        params = {
            "textDocument": {"uri": file_uri},
        }
        
        try:
            result = await self.send_request("textDocument/documentSymbol", params)
            return result if isinstance(result, list) else None
        except LspError as e:
            logger.warning(
                "Document symbol request failed",
                server_name=self._server_name,
                file_uri=file_uri,
                error=str(e),
            )
            return None

    async def workspace_symbol(
        self,
        query: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for symbols across the workspace.
        
        Args:
            query: Search query string
            
        Returns:
            List of SymbolInformation objects or None if not available
        """
        if not self._server_capabilities or not self._server_capabilities.supports_workspace_symbol():
            logger.debug(
                "Workspace symbol not supported by server",
                server_name=self._server_name,
            )
            return None

        params = {"query": query}
        
        try:
            result = await self.send_request("workspace/symbol", params)
            return result if isinstance(result, list) else None
        except LspError as e:
            logger.warning(
                "Workspace symbol request failed",
                server_name=self._server_name,
                query=query,
                error=str(e),
            )
            return None

    # Document synchronization methods
    async def did_open(
        self,
        file_uri: str,
        language_id: str,
        version: int,
        text: str,
    ) -> None:
        """
        Notify server that a document was opened.
        
        Args:
            file_uri: URI of the opened file
            language_id: Language identifier (e.g., 'python', 'rust')
            version: Document version number
            text: Complete document content
        """
        params = {
            "textDocument": {
                "uri": file_uri,
                "languageId": language_id,
                "version": version,
                "text": text,
            }
        }
        
        await self.send_notification("textDocument/didOpen", params)
        logger.debug(
            "Sent didOpen notification",
            server_name=self._server_name,
            file_uri=file_uri,
            language_id=language_id,
        )

    async def did_change(
        self,
        file_uri: str,
        version: int,
        changes: List[Dict[str, Any]],
    ) -> None:
        """
        Notify server that a document was changed.
        
        Args:
            file_uri: URI of the changed file
            version: New document version number
            changes: List of content changes
        """
        params = {
            "textDocument": {
                "uri": file_uri,
                "version": version,
            },
            "contentChanges": changes,
        }
        
        await self.send_notification("textDocument/didChange", params)
        logger.debug(
            "Sent didChange notification",
            server_name=self._server_name,
            file_uri=file_uri,
            changes_count=len(changes),
        )

    async def did_save(
        self,
        file_uri: str,
        text: Optional[str] = None,
    ) -> None:
        """
        Notify server that a document was saved.
        
        Args:
            file_uri: URI of the saved file
            text: Complete document content (if server requires it)
        """
        params = {
            "textDocument": {"uri": file_uri}
        }
        if text is not None:
            params["text"] = text
        
        await self.send_notification("textDocument/didSave", params)
        logger.debug(
            "Sent didSave notification",
            server_name=self._server_name,
            file_uri=file_uri,
        )

    async def did_close(
        self,
        file_uri: str,
    ) -> None:
        """
        Notify server that a document was closed.
        
        Args:
            file_uri: URI of the closed file
        """
        params = {
            "textDocument": {"uri": file_uri}
        }
        
        await self.send_notification("textDocument/didClose", params)
        logger.debug(
            "Sent didClose notification",
            server_name=self._server_name,
            file_uri=file_uri,
        )

    # Notification handler management
    def register_diagnostics_handler(
        self,
        handler: Callable[[str, List[Dict[str, Any]]], None],
    ) -> None:
        """
        Register handler for diagnostic notifications.
        
        Args:
            handler: Function that takes (file_uri, diagnostics_list)
        """
        def diagnostics_wrapper(notification: JsonRpcNotification) -> None:
            try:
                params = notification.params or {}
                file_uri = params.get("uri", "")
                diagnostics = params.get("diagnostics", [])
                handler(file_uri, diagnostics)
            except Exception as e:
                logger.error(
                    "Error in diagnostics handler",
                    server_name=self._server_name,
                    error=str(e),
                )
        
        self.register_notification_handler("textDocument/publishDiagnostics", diagnostics_wrapper)
        logger.info(
            "Registered diagnostics handler",
            server_name=self._server_name,
        )

    def register_progress_handler(
        self,
        handler: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """
        Register handler for work progress notifications.
        
        Args:
            handler: Function that takes (token, progress_data)
        """
        def progress_wrapper(notification: JsonRpcNotification) -> None:
            try:
                params = notification.params or {}
                token = params.get("token", "")
                value = params.get("value", {})
                handler(token, value)
            except Exception as e:
                logger.error(
                    "Error in progress handler",
                    server_name=self._server_name,
                    error=str(e),
                )
        
        self.register_notification_handler("$/progress", progress_wrapper)
        logger.info(
            "Registered progress handler",
            server_name=self._server_name,
        )

    def register_log_message_handler(
        self,
        handler: Callable[[int, str], None],
    ) -> None:
        """
        Register handler for server log messages.
        
        Args:
            handler: Function that takes (message_type, message)
        """
        def log_wrapper(notification: JsonRpcNotification) -> None:
            try:
                params = notification.params or {}
                message_type = params.get("type", 1)  # Default to Error
                message = params.get("message", "")
                handler(message_type, message)
            except Exception as e:
                logger.error(
                    "Error in log message handler",
                    server_name=self._server_name,
                    error=str(e),
                )
        
        self.register_notification_handler("window/logMessage", log_wrapper)
        logger.info(
            "Registered log message handler",
            server_name=self._server_name,
        )

    def register_show_message_handler(
        self,
        handler: Callable[[int, str], None],
    ) -> None:
        """
        Register handler for server show message notifications.
        
        Args:
            handler: Function that takes (message_type, message)
        """
        def show_message_wrapper(notification: JsonRpcNotification) -> None:
            try:
                params = notification.params or {}
                message_type = params.get("type", 1)  # Default to Error
                message = params.get("message", "")
                handler(message_type, message)
            except Exception as e:
                logger.error(
                    "Error in show message handler",
                    server_name=self._server_name,
                    error=str(e),
                )
        
        self.register_notification_handler("window/showMessage", show_message_wrapper)
        logger.info(
            "Registered show message handler",
            server_name=self._server_name,
        )

    def register_configuration_handler(
        self,
        handler: Callable[[List[Dict[str, Any]]], List[Any]],
    ) -> None:
        """
        Register handler for workspace configuration requests.
        Note: This is actually a request from server, but handled here for convenience.
        
        Args:
            handler: Function that takes configuration items and returns values
        """
        async def config_wrapper(request_data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                params = request_data.get("params", {})
                items = params.get("items", [])
                result = handler(items)
                return {"result": result}
            except Exception as e:
                logger.error(
                    "Error in configuration handler",
                    server_name=self._server_name,
                    error=str(e),
                )
                return {
                    "error": {
                        "code": -32603,  # InternalError
                        "message": f"Configuration handler error: {e}",
                    }
                }
        
        # Note: This would need request handling support in the base client
        logger.info(
            "Configuration handler registered (requires request handling)",
            server_name=self._server_name,
        )

    # Convenience methods for file watching integration
    async def sync_file_opened(
        self,
        file_path: str,
        content: str,
        language_id: Optional[str] = None,
    ) -> None:
        """
        Convenience method to sync a newly opened file with the LSP server.
        
        Args:
            file_path: Local file path
            content: File content
            language_id: Language identifier (auto-detected if None)
        """
        file_uri = f"file://{Path(file_path).resolve()}"
        
        if language_id is None:
            # Auto-detect language from file extension
            extension = Path(file_path).suffix.lstrip('.')
            language_map = {
                'py': 'python',
                'rs': 'rust',
                'ts': 'typescript',
                'js': 'javascript',
                'json': 'json',
                'go': 'go',
                'java': 'java',
                'c': 'c',
                'cpp': 'cpp',
                'cc': 'cpp',
                'cxx': 'cpp',
                'hpp': 'cpp',
                'rb': 'ruby',
                'php': 'php',
                'sh': 'shellscript',
                'bash': 'shellscript',
                'yaml': 'yaml',
                'yml': 'yaml',
                'toml': 'toml',
                'xml': 'xml',
                'html': 'html',
                'css': 'css',
                'sql': 'sql',
            }
            language_id = language_map.get(extension, 'plaintext')
        
        await self.did_open(file_uri, language_id, 1, content)

    async def sync_file_changed(
        self,
        file_path: str,
        content: str,
        version: int = 1,
    ) -> None:
        """
        Convenience method to sync file changes with the LSP server.
        
        Args:
            file_path: Local file path
            content: New file content
            version: Document version number
        """
        file_uri = f"file://{Path(file_path).resolve()}"
        
        # Send full document change (simpler than incremental changes)
        changes = [{"text": content}]
        
        await self.did_change(file_uri, version, changes)

    async def sync_file_saved(
        self,
        file_path: str,
        content: Optional[str] = None,
    ) -> None:
        """
        Convenience method to notify server of file save.
        
        Args:
            file_path: Local file path
            content: File content (if server requires it)
        """
        file_uri = f"file://{Path(file_path).resolve()}"
        await self.did_save(file_uri, content)

    async def sync_file_closed(
        self,
        file_path: str,
    ) -> None:
        """
        Convenience method to notify server of file close.
        
        Args:
            file_path: Local file path
        """
        file_uri = f"file://{Path(file_path).resolve()}"
        await self.did_close(file_uri)

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Send a JSON-RPC request and wait for response.
        
        Args:
            method: LSP method name
            params: Method parameters
            timeout: Request timeout (defaults to client timeout)
            
        Returns:
            Response result data
            
        Raises:
            LspError: On communication or protocol errors
            LspTimeoutError: On request timeout
        """
        if not self.is_connected():
            raise LspError(
                "Cannot send request: client is not connected",
                server_name=self._server_name,
                method=method,
            )

        if len(self._pending_requests) >= self._max_pending_requests:
            raise LspError(
                "Too many pending requests",
                server_name=self._server_name,
                method=method,
                context={"pending_count": len(self._pending_requests)},
            )

        request_id = str(uuid.uuid4())
        request_timeout = timeout or self._request_timeout
        
        request = JsonRpcRequest(
            id=request_id,
            method=method,
            params=params,
        )
        
        # Create future for response
        response_future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = PendingRequest(
            future=response_future,
            method=method,
            created_at=asyncio.get_event_loop().time(),
            timeout=request_timeout,
        )
        
        try:
            # Send request
            await self._send_message(request.to_dict())
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=request_timeout)
            
            if response.error:
                raise LspError(
                    f"LSP server returned error: {response.error.message}",
                    server_name=self._server_name,
                    method=method,
                    context={
                        "error_code": response.error.code,
                        "error_data": response.error.data,
                    },
                )
            
            logger.debug(
                "LSP request completed",
                server_name=self._server_name,
                method=method,
                request_id=request_id,
            )
            
            return response.result
            
        except asyncio.TimeoutError:
            raise LspTimeoutError(method, request_timeout, server_name=self._server_name)
        except Exception as e:
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
            if isinstance(e, LspError):
                raise
            raise LspError(
                f"Failed to send LSP request: {e}",
                server_name=self._server_name,
                method=method,
                cause=e,
            ) from e
        finally:
            # Clean up pending request
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]

    async def send_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send a JSON-RPC notification (no response expected).
        
        Args:
            method: LSP method name
            params: Method parameters
        """
        if not self.is_connected():
            raise LspError(
                "Cannot send notification: client is not connected",
                server_name=self._server_name,
                method=method,
            )

        notification = JsonRpcNotification(method=method, params=params)
        await self._send_message(notification.to_dict())
        
        logger.debug(
            "LSP notification sent",
            server_name=self._server_name,
            method=method,
        )

    def register_notification_handler(
        self,
        method: str,
        handler: Callable[[JsonRpcNotification], None],
    ) -> None:
        """Register a handler for specific notification method"""
        if method not in self._notification_handlers:
            self._notification_handlers[method] = []
        self._notification_handlers[method].append(handler)
        
        logger.debug(
            "Notification handler registered",
            server_name=self._server_name,
            method=method,
        )

    def register_global_handler(
        self,
        handler: Callable[[JsonRpcNotification], None],
    ) -> None:
        """Register a handler for all notifications"""
        self._global_handlers.add(handler)

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a JSON-RPC message with LSP content-length header"""
        if not self._writer:
            raise LspError(
                "No writer available for sending message",
                server_name=self._server_name,
            )
        
        message_json = json.dumps(message, separators=(',', ':'))
        message_bytes = message_json.encode('utf-8')
        content_length = len(message_bytes)
        
        header = f"Content-Length: {content_length}\r\n\r\n"
        full_message = header.encode('ascii') + message_bytes
        
        try:
            self._writer.write(full_message)
            await self._writer.drain()
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            raise LspError(
                f"Failed to send message: {e}",
                server_name=self._server_name,
                cause=e,
            ) from e

    async def _message_reader_loop(self) -> None:
        """Background task to read and process incoming messages"""
        logger.debug("Starting LSP message reader loop", server_name=self._server_name)
        
        try:
            while not self._shutdown_event.is_set() and self._reader:
                try:
                    # Read message with timeout to allow shutdown
                    message = await asyncio.wait_for(
                        self._read_message(),
                        timeout=1.0
                    )
                    if message:
                        await self._handle_incoming_message(message)
                        
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "Error in message reader loop",
                        server_name=self._server_name,
                        error=str(e),
                    )
                    self._connection_state = ConnectionState.ERROR
                    break
                    
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("LSP message reader loop ended", server_name=self._server_name)

    async def _read_message(self) -> Optional[Dict[str, Any]]:
        """Read a single LSP message with content-length header"""
        if not self._reader:
            return None
        
        try:
            # Read Content-Length header
            header_line = await self._reader.readline()
            if not header_line:
                return None  # EOF
                
            header = header_line.decode('ascii').strip()
            if not header.startswith('Content-Length:'):
                logger.warning(
                    "Invalid LSP header",
                    server_name=self._server_name,
                    header=header,
                )
                return None
                
            content_length = int(header[len('Content-Length:'):].strip())
            
            # Read empty line
            empty_line = await self._reader.readline()
            if empty_line.strip():
                logger.warning(
                    "Expected empty line after Content-Length header",
                    server_name=self._server_name,
                )
            
            # Read message content
            content = await self._reader.readexactly(content_length)
            message_text = content.decode('utf-8')
            
            return json.loads(message_text)
            
        except json.JSONDecodeError as e:
            raise LspProtocolError(
                "Invalid JSON in LSP message",
                raw_data=content.decode('utf-8', errors='replace') if 'content' in locals() else None,
                server_name=self._server_name,
            ) from e
        except Exception as e:
            if self._shutdown_event.is_set():
                return None
            raise LspError(
                f"Failed to read LSP message: {e}",
                server_name=self._server_name,
                cause=e,
            ) from e

    async def _handle_incoming_message(self, message: Dict[str, Any]) -> None:
        """Handle an incoming JSON-RPC message"""
        try:
            # Determine message type
            if "id" in message and ("result" in message or "error" in message):
                # Response
                await self._handle_response(message)
            elif "method" in message and "id" not in message:
                # Notification
                await self._handle_notification(message)
            elif "method" in message and "id" in message:
                # Request from server (not common in LSP)
                logger.warning(
                    "Received request from LSP server - not handled",
                    server_name=self._server_name,
                    method=message.get("method"),
                )
            else:
                raise LspProtocolError(
                    "Invalid JSON-RPC message format",
                    raw_data=json.dumps(message),
                    server_name=self._server_name,
                )
                
        except Exception as e:
            logger.error(
                "Error handling incoming LSP message",
                server_name=self._server_name,
                error=str(e),
                message=message,
            )

    async def _handle_response(self, message: Dict[str, Any]) -> None:
        """Handle a JSON-RPC response"""
        request_id = str(message.get("id", ""))
        
        if request_id in self._pending_requests:
            pending = self._pending_requests[request_id]
            response = JsonRpcResponse.from_dict(message)
            
            if not pending.future.done():
                pending.future.set_result(response)
        else:
            logger.warning(
                "Received response for unknown request",
                server_name=self._server_name,
                request_id=request_id,
            )

    async def _handle_notification(self, message: Dict[str, Any]) -> None:
        """Handle a JSON-RPC notification"""
        notification = JsonRpcNotification.from_dict(message)
        method = notification.method
        
        # Call method-specific handlers
        if method in self._notification_handlers:
            for handler in self._notification_handlers[method]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(notification)
                    else:
                        handler(notification)
                except Exception as e:
                    logger.error(
                        "Error in notification handler",
                        server_name=self._server_name,
                        method=method,
                        error=str(e),
                    )
        
        # Call global handlers
        for handler in list(self._global_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(
                    "Error in global notification handler",
                    server_name=self._server_name,
                    method=method,
                    error=str(e),
                )

    async def _request_cleanup_loop(self) -> None:
        """Background task to clean up expired requests"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                current_time = asyncio.get_event_loop().time()
                expired_requests = []
                
                for request_id, pending in self._pending_requests.items():
                    if current_time - pending.created_at > pending.timeout:
                        expired_requests.append(request_id)
                
                for request_id in expired_requests:
                    pending = self._pending_requests.pop(request_id, None)
                    if pending and not pending.future.done():
                        pending.future.cancel()
                        logger.warning(
                            "Request expired and cancelled",
                            server_name=self._server_name,
                            method=pending.method,
                            request_id=request_id,
                        )
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in request cleanup loop",
                    server_name=self._server_name,
                    error=str(e),
                )

    async def _health_check_loop(self) -> None:
        """Background task for connection health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._circuit_breaker_config.health_check_interval)
                
                # Check if connection is still alive
                if self.is_connected():
                    time_since_last_success = time.time() - self._last_successful_request
                    
                    # If no successful requests for too long, consider it unhealthy
                    if time_since_last_success > self._circuit_breaker_config.timeout_seconds * 2:
                        logger.warning(
                            "LSP server appears unhealthy - no successful requests",
                            server_name=self._server_name,
                            time_since_last_success=time_since_last_success,
                        )
                        self._record_failure()
                
                # Log circuit breaker state periodically
                if self._circuit_breaker_state != CircuitBreakerState.CLOSED:
                    logger.debug(
                        "Circuit breaker health check",
                        server_name=self._server_name,
                        state=self._circuit_breaker_state.value,
                        failure_count=self._failure_count,
                        success_count=self._success_count,
                    )
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in health check loop",
                    server_name=self._server_name,
                    error=str(e),
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = {
            "server_name": self._server_name,
            "connection_state": self._connection_state.value,
            "communication_mode": self._communication_mode.value,
            "initialized": self._initialized,
            "pending_requests": len(self._pending_requests),
            "notification_handlers": {
                method: len(handlers)
                for method, handlers in self._notification_handlers.items()
            },
            "global_handlers": len(self._global_handlers),
        }
        
        # Add communication-specific details
        if self._communication_mode == CommunicationMode.STDIO and self._server_process:
            stats["process_id"] = self._server_process.pid
            stats["process_returncode"] = self._server_process.returncode
        elif self._communication_mode == CommunicationMode.TCP:
            stats["tcp_host"] = self._tcp_host
            stats["tcp_port"] = self._tcp_port
        
        # Add server capabilities if available
        if self._server_capabilities:
            stats["server_capabilities"] = {
                "hover": self._server_capabilities.supports_hover(),
                "definition": self._server_capabilities.supports_definition(),
                "references": self._server_capabilities.supports_references(),
                "document_symbol": self._server_capabilities.supports_document_symbol(),
                "workspace_symbol": self._server_capabilities.supports_workspace_symbol(),
                "completion": self._server_capabilities.supports_completion(),
                "diagnostics": self._server_capabilities.supports_diagnostics(),
            }
        
        return stats

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()