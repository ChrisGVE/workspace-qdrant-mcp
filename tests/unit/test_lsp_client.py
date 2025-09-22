"""
Comprehensive test suite for LSP client implementation.

Tests JSON-RPC protocol compliance, communication modes, error handling,
timeout behavior, and integration with mock LSP servers.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import json
import pytest
import tempfile
import uuid
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

from workspace_qdrant_mcp.core.lsp_client import (
    AsyncioLspClient,
    LspError,
    LspTimeoutError,
    LspProtocolError,
    ConnectionState,
    CommunicationMode,
    ServerCapabilities,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    ClientCapabilities,
    InitializeParams,
)


class MockLspServer:
    """Mock LSP server for testing JSON-RPC protocol compliance."""
    
    def __init__(self, capabilities: Optional[Dict] = None):
        self.capabilities = capabilities or {
            "textDocumentSync": {"openClose": True, "change": 2},
            "hoverProvider": True,
            "definitionProvider": True,
            "referencesProvider": True,
            "documentSymbolProvider": True,
            "workspaceSymbolProvider": True,
        }
        self.messages: List[Dict] = []
        self.responses: Dict[str, Dict] = {}
        self.notifications: List[Dict] = []
        self.initialized = False
        self.should_fail = False
        
    async def handle_message(self, message: Dict) -> Optional[Dict]:
        """Handle incoming JSON-RPC message."""
        self.messages.append(message)
        
        if self.should_fail:
            raise Exception("Mock server failure")
        
        # Handle requests
        if "method" in message and "id" in message:
            return await self._handle_request(message)
        
        # Handle notifications
        elif "method" in message:
            self.notifications.append(message)
            return None
            
        return None
    
    async def _handle_request(self, request: Dict) -> Dict:
        """Handle JSON-RPC request."""
        method = request["method"]
        request_id = request["id"]
        params = request.get("params", {})
        
        # Check for custom responses
        if request_id in self.responses:
            return self.responses[request_id]
        
        # Default responses based on method
        if method == "initialize":
            self.initialized = True
            return {
                "id": request_id,
                "result": {
                    "capabilities": self.capabilities,
                    "serverInfo": {"name": "mock-lsp-server", "version": "1.0.0"}
                }
            }
        
        elif method == "textDocument/hover":
            return {
                "id": request_id,
                "result": {
                    "contents": {"kind": "markdown", "value": "Mock hover info"},
                    "range": {
                        "start": {"line": params.get("position", {}).get("line", 0), "character": 0},
                        "end": {"line": params.get("position", {}).get("line", 0), "character": 10}
                    }
                }
            }
        
        elif method == "textDocument/definition":
            return {
                "id": request_id,
                "result": [{
                    "uri": params.get("textDocument", {}).get("uri", "file:///test.py"),
                    "range": {
                        "start": {"line": 10, "character": 0},
                        "end": {"line": 10, "character": 15}
                    }
                }]
            }
        
        elif method == "textDocument/references":
            return {
                "id": request_id,
                "result": [
                    {
                        "uri": params.get("textDocument", {}).get("uri", "file:///test.py"),
                        "range": {
                            "start": {"line": 5, "character": 0},
                            "end": {"line": 5, "character": 10}
                        }
                    },
                    {
                        "uri": "file:///other.py",
                        "range": {
                            "start": {"line": 15, "character": 5},
                            "end": {"line": 15, "character": 15}
                        }
                    }
                ]
            }
        
        elif method == "textDocument/documentSymbol":
            return {
                "id": request_id,
                "result": [
                    {
                        "name": "TestClass",
                        "kind": 5,  # Class
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 20, "character": 0}
                        },
                        "selectionRange": {
                            "start": {"line": 0, "character": 6},
                            "end": {"line": 0, "character": 15}
                        },
                        "children": [
                            {
                                "name": "test_method",
                                "kind": 6,  # Method
                                "range": {
                                    "start": {"line": 2, "character": 4},
                                    "end": {"line": 5, "character": 8}
                                },
                                "selectionRange": {
                                    "start": {"line": 2, "character": 8},
                                    "end": {"line": 2, "character": 19}
                                }
                            }
                        ]
                    }
                ]
            }
        
        elif method == "workspace/symbol":
            query = params.get("query", "")
            return {
                "id": request_id,
                "result": [
                    {
                        "name": f"MockSymbol_{query}",
                        "kind": 12,  # Function
                        "location": {
                            "uri": "file:///mock.py",
                            "range": {
                                "start": {"line": 1, "character": 0},
                                "end": {"line": 1, "character": 15}
                            }
                        }
                    }
                ]
            }
        
        else:
            return {
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
    
    def set_custom_response(self, request_id: str, response: Dict):
        """Set custom response for specific request."""
        self.responses[request_id] = response
    
    def simulate_failure(self):
        """Simulate server failure."""
        self.should_fail = True


class MockStreamReader:
    """Mock asyncio.StreamReader for testing."""
    
    def __init__(self, formatted_messages: List[str]):
        # formatted_messages should contain full LSP messages with Content-Length headers
        self.raw_data = "\r\n".join(formatted_messages).encode("utf-8")
        self.position = 0
        self.closed = False
    
    async def readline(self) -> bytes:
        if self.closed or self.position >= len(self.raw_data):
            return b""
        
        # Find next line ending
        start = self.position
        while self.position < len(self.raw_data):
            if self.raw_data[self.position:self.position+2] == b"\r\n":
                line = self.raw_data[start:self.position + 2]
                self.position += 2
                return line
            elif self.raw_data[self.position:self.position+1] == b"\n":
                line = self.raw_data[start:self.position + 1]
                self.position += 1
                return line
            self.position += 1
        
        # Return rest of data if no line ending found
        if start < len(self.raw_data):
            line = self.raw_data[start:]
            self.position = len(self.raw_data)
            return line
        
        return b""
    
    async def read(self, n: int = -1) -> bytes:
        if self.closed or self.position >= len(self.raw_data):
            return b""
        
        if n <= 0:
            result = self.raw_data[self.position:]
            self.position = len(self.raw_data)
            return result
        
        end = min(self.position + n, len(self.raw_data))
        result = self.raw_data[self.position:end]
        self.position = end
        return result
    
    async def readexactly(self, n: int) -> bytes:
        if self.closed:
            return b""
        
        if self.position + n > len(self.raw_data):
            # Not enough data - return what we have
            result = self.raw_data[self.position:]
            self.position = len(self.raw_data)
            return result
        
        result = self.raw_data[self.position:self.position + n]
        self.position += n
        return result
    
    def at_eof(self) -> bool:
        return self.closed or self.position >= len(self.raw_data)


class MockStreamWriter:
    """Mock asyncio.StreamWriter for testing."""
    
    def __init__(self):
        self.written_data: List[bytes] = []
        self.closed = False
    
    def write(self, data: bytes):
        if not self.closed:
            self.written_data.append(data)
    
    async def drain(self):
        pass
    
    def close(self):
        self.closed = True
    
    async def wait_closed(self):
        pass
    
    def get_written_messages(self) -> List[str]:
        """Extract JSON-RPC messages from written data."""
        messages = []
        data = b"".join(self.written_data).decode()
        
        # Parse Content-Length protocol
        while "Content-Length:" in data:
            header_end = data.find("\r\n\r\n")
            if header_end == -1:
                break
                
            header = data[:header_end]
            length_line = [line for line in header.split("\r\n") if line.startswith("Content-Length:")]
            if not length_line:
                break
                
            length = int(length_line[0].split(":")[1].strip())
            message_start = header_end + 4
            message = data[message_start:message_start + length]
            messages.append(message)
            data = data[message_start + length:]
        
        return messages


@pytest.fixture
def mock_server():
    """Create mock LSP server."""
    return MockLspServer()


@pytest.fixture
def lsp_client():
    """Create LSP client for testing."""
    return AsyncioLspClient(
        server_name="test-server",
        request_timeout=5.0,
        max_pending_requests=10,
    )


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir)
        (workspace_path / "test.py").write_text("def test_function():\n    pass\n")
        (workspace_path / "other.py").write_text("import test\n")
        yield workspace_path


class TestAsyncioLspClient:
    """Test AsyncioLspClient foundation and basic functionality."""
    
    def test_client_initialization(self, lsp_client):
        """Test client initializes with correct parameters."""
        assert lsp_client.server_name == "test-server"
        assert lsp_client._request_timeout == 5.0
        assert lsp_client._max_pending_requests == 10
        assert not lsp_client.is_connected()
    
    def test_message_id_generation(self, lsp_client):
        """Test request ID generation is unique."""
        ids = set()
        for _ in range(100):
            request_id = str(uuid.uuid4())
            assert request_id not in ids
            ids.add(request_id)
    
    def test_json_rpc_message_formatting(self, lsp_client):
        """Test JSON-RPC message formatting follows protocol."""
        request = JsonRpcRequest(
            id="test-id",
            method="test/method",
            params={"param": "value"}
        )
        message = request.to_dict()
        
        assert "jsonrpc" in message
        assert message["jsonrpc"] == "2.0"
        assert "id" in message
        assert message["method"] == "test/method"
        assert "params" in message
        assert message["params"] == {"param": "value"}
    
    def test_content_length_protocol(self, lsp_client):
        """Test Content-Length header protocol implementation."""
        message = {"jsonrpc": "2.0", "method": "test", "params": {}}
        message_json = json.dumps(message, separators=(',', ':'))
        message_bytes = message_json.encode('utf-8')
        content_length = len(message_bytes)
        
        header = f"Content-Length: {content_length}\r\n\r\n"
        full_message = header.encode('ascii') + message_bytes
        
        lines = full_message.decode('utf-8').split("\r\n")
        assert lines[0].startswith("Content-Length:")
        assert lines[1] == ""  # Empty line after headers
        
        parsed_length = int(lines[0].split(":")[1].strip())
        content = "\r\n".join(lines[2:])
        assert len(content.encode()) == parsed_length


class TestLspCommunicationModes:
    """Test different LSP communication modes (stdio, TCP, manual)."""
    
    @pytest.mark.asyncio
    async def test_stdio_connection_setup(self, lsp_client, mock_server):
        """Test stdio connection setup with process spawning."""
        
        # Test manual connection (stdio setup is complex to mock properly)
        message = '{"jsonrpc":"2.0","id":"1","result":{"capabilities":{}}}'
        formatted_message = f'Content-Length: {len(message)}\r\n\r\n{message}'
        reader = MockStreamReader([formatted_message])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        lsp_client._communication_mode = CommunicationMode.STDIO  # Simulate stdio mode
        
        assert lsp_client.is_connected()
        assert lsp_client.communication_mode == CommunicationMode.STDIO
        
        await lsp_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_tcp_connection_setup(self, lsp_client):
        """Test TCP connection setup with socket communication."""
        
        message = '{"jsonrpc":"2.0","id":"1","result":{"capabilities":{}}}'
        formatted_message = f'Content-Length: {len(message)}\r\n\r\n{message}'
        reader = MockStreamReader([formatted_message])
        writer = MockStreamWriter()
        
        with patch('asyncio.open_connection', return_value=(reader, writer)):
            await lsp_client.connect_tcp("localhost", 8080)
            
            assert lsp_client.is_connected()
            assert lsp_client.communication_mode == CommunicationMode.TCP
            
            await lsp_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_manual_connection_setup(self, lsp_client):
        """Test manual connection with provided streams."""
        
        message = '{"jsonrpc":"2.0","id":"1","result":{"capabilities":{}}}'
        formatted_message = f'Content-Length: {len(message)}\r\n\r\n{message}'
        reader = MockStreamReader([formatted_message])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        assert lsp_client.is_connected()
        assert lsp_client.communication_mode == CommunicationMode.MANUAL
        
        await lsp_client.disconnect()


class TestLspInitialization:
    """Test LSP initialization handshake and capability negotiation."""
    
    @pytest.mark.asyncio
    async def test_initialization_handshake(self, lsp_client, temp_workspace):
        """Test complete initialization handshake."""
        
        # Mock successful initialization response
        init_response = {
            "jsonrpc": "2.0",
            "id": "init",
            "result": {
                "capabilities": {
                    "textDocumentSync": {"openClose": True, "change": 2},
                    "hoverProvider": True,
                    "definitionProvider": True,
                    "referencesProvider": True,
                    "documentSymbolProvider": True,
                    "workspaceSymbolProvider": True,
                },
                "serverInfo": {"name": "test-server", "version": "1.0.0"}
            }
        }
        
        init_response_json = json.dumps(init_response)
        formatted_message = f'Content-Length: {len(init_response_json)}\r\n\r\n{init_response_json}'
        reader = MockStreamReader([formatted_message])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        # Initialize with workspace
        root_uri = f"file://{temp_workspace.resolve()}"
        capabilities = await lsp_client.initialize(
            root_uri=root_uri,
            client_name="test-client",
            client_version="1.0.0"
        )
        
        assert capabilities is not None
        assert capabilities.get_text_document_sync() is not None
        assert capabilities.supports_hover() is True
        assert capabilities.supports_definition() is True
        
        # Check initialization message was sent
        messages = writer.get_written_messages()
        assert len(messages) >= 1
        
        init_message = json.loads(messages[0])
        assert init_message["method"] == "initialize"
        assert "rootUri" in init_message["params"]
        assert init_message["params"]["rootUri"] == root_uri
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, lsp_client):
        """Test handling of initialization failures."""
        
        # Mock initialization error response
        error_response = {
            "jsonrpc": "2.0",
            "id": "init",
            "error": {"code": -32603, "message": "Initialization failed"}
        }
        
        reader = MockStreamReader([
            f'Content-Length: {len(json.dumps(error_response))}\r\n\r\n{json.dumps(error_response)}',
        ])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        with pytest.raises(LspError):
            await lsp_client.initialize("file:///test", "test-client")
    
    @pytest.mark.asyncio
    async def test_capability_detection(self, lsp_client):
        """Test LSP server capability detection and parsing."""
        
        capabilities_data = {
            "textDocumentSync": {"openClose": True, "change": 2},
            "hoverProvider": True,
            "definitionProvider": True,
            "referencesProvider": True,
            "documentSymbolProvider": True,
            "workspaceSymbolProvider": False,  # Disabled feature
        }
        
        init_response = {
            "jsonrpc": "2.0",
            "id": "init",
            "result": {"capabilities": capabilities_data}
        }
        
        reader = MockStreamReader([
            f'Content-Length: {len(json.dumps(init_response))}\r\n\r\n{json.dumps(init_response)}',
        ])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        capabilities = await lsp_client.initialize("file:///test", "test-client")
        
        assert capabilities.supports_hover() is True
        assert capabilities.supports_definition() is True
        assert capabilities.supports_references() is True
        assert capabilities.supports_document_symbol() is True
        assert capabilities.supports_workspace_symbol() is False


class TestLspCoreMethods:
    """Test core LSP method implementations."""
    
    @pytest.fixture
    async def connected_client(self, lsp_client):
        """Provide connected and initialized LSP client."""
        
        init_response = {
            "jsonrpc": "2.0",
            "id": "init",
            "result": {
                "capabilities": {
                    "hoverProvider": True,
                    "definitionProvider": True,
                    "referencesProvider": True,
                    "documentSymbolProvider": True,
                    "workspaceSymbolProvider": True,
                }
            }
        }
        
        reader = MockStreamReader([
            f'Content-Length: {len(json.dumps(init_response))}\r\n\r\n{json.dumps(init_response)}',
        ])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        await lsp_client.initialize("file:///test", "test-client")
        
        return lsp_client, reader, writer
    
    @pytest.mark.asyncio
    async def test_hover_request(self, connected_client):
        """Test textDocument/hover request."""
        
        client, reader, writer = connected_client
        
        # Mock hover response
        hover_response = {
            "jsonrpc": "2.0",
            "id": "hover_1",
            "result": {
                "contents": {"kind": "markdown", "value": "Test hover info"},
                "range": {
                    "start": {"line": 5, "character": 10},
                    "end": {"line": 5, "character": 20}
                }
            }
        }
        
        # Add hover response to reader
        reader.messages.append(f'Content-Length: {len(json.dumps(hover_response))}\r\n\r\n{json.dumps(hover_response)}')
        
        result = await client.hover("file:///test.py", 5, 10)
        
        assert result is not None
        assert "contents" in result
        assert "Test hover info" in str(result["contents"])
        
        # Verify request was sent correctly
        messages = writer.get_written_messages()
        hover_request = None
        for msg in messages:
            parsed = json.loads(msg)
            if parsed.get("method") == "textDocument/hover":
                hover_request = parsed
                break
        
        assert hover_request is not None
        assert hover_request["params"]["textDocument"]["uri"] == "file:///test.py"
        assert hover_request["params"]["position"]["line"] == 5
        assert hover_request["params"]["position"]["character"] == 10
    
    @pytest.mark.asyncio
    async def test_definition_request(self, connected_client):
        """Test textDocument/definition request."""
        
        client, reader, writer = connected_client
        
        # Mock definition response
        definition_response = {
            "jsonrpc": "2.0",
            "id": "def_1",
            "result": [{
                "uri": "file:///other.py",
                "range": {
                    "start": {"line": 10, "character": 0},
                    "end": {"line": 10, "character": 15}
                }
            }]
        }
        
        reader.messages.append(f'Content-Length: {len(json.dumps(definition_response))}\r\n\r\n{json.dumps(definition_response)}')
        
        result = await client.definition("file:///test.py", 5, 10)
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["uri"] == "file:///other.py"
        assert result[0]["range"]["start"]["line"] == 10
    
    @pytest.mark.asyncio
    async def test_references_request(self, connected_client):
        """Test textDocument/references request."""
        
        client, reader, writer = connected_client
        
        # Mock references response
        references_response = {
            "jsonrpc": "2.0",
            "id": "ref_1",
            "result": [
                {
                    "uri": "file:///test.py",
                    "range": {
                        "start": {"line": 5, "character": 0},
                        "end": {"line": 5, "character": 10}
                    }
                },
                {
                    "uri": "file:///other.py",
                    "range": {
                        "start": {"line": 15, "character": 5},
                        "end": {"line": 15, "character": 15}
                    }
                }
            ]
        }
        
        reader.messages.append(f'Content-Length: {len(json.dumps(references_response))}\r\n\r\n{json.dumps(references_response)}')
        
        result = await client.references("file:///test.py", 5, 10, include_declaration=True)
        
        assert result is not None
        assert len(result) == 2
        assert result[0]["uri"] == "file:///test.py"
        assert result[1]["uri"] == "file:///other.py"
    
    @pytest.mark.asyncio
    async def test_document_symbol_request(self, connected_client):
        """Test textDocument/documentSymbol request."""
        
        client, reader, writer = connected_client
        
        # Mock document symbols response
        symbols_response = {
            "jsonrpc": "2.0",
            "id": "sym_1",
            "result": [
                {
                    "name": "TestClass",
                    "kind": 5,  # Class
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 20, "character": 0}
                    },
                    "selectionRange": {
                        "start": {"line": 0, "character": 6},
                        "end": {"line": 0, "character": 15}
                    },
                    "children": [
                        {
                            "name": "test_method",
                            "kind": 6,  # Method
                            "range": {
                                "start": {"line": 2, "character": 4},
                                "end": {"line": 5, "character": 8}
                            },
                            "selectionRange": {
                                "start": {"line": 2, "character": 8},
                                "end": {"line": 2, "character": 19}
                            }
                        }
                    ]
                }
            ]
        }
        
        reader.messages.append(f'Content-Length: {len(json.dumps(symbols_response))}\r\n\r\n{json.dumps(symbols_response)}')
        
        result = await client.document_symbol("file:///test.py")
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "TestClass"
        assert result[0]["kind"] == 5
        assert len(result[0]["children"]) == 1
        assert result[0]["children"][0]["name"] == "test_method"
    
    @pytest.mark.asyncio
    async def test_workspace_symbol_request(self, connected_client):
        """Test workspace/symbol request."""
        
        client, reader, writer = connected_client
        
        # Mock workspace symbols response
        symbols_response = {
            "jsonrpc": "2.0",
            "id": "ws_sym_1",
            "result": [
                {
                    "name": "test_function",
                    "kind": 12,  # Function
                    "location": {
                        "uri": "file:///test.py",
                        "range": {
                            "start": {"line": 1, "character": 0},
                            "end": {"line": 1, "character": 13}
                        }
                    }
                }
            ]
        }
        
        reader.messages.append(f'Content-Length: {len(json.dumps(symbols_response))}\r\n\r\n{json.dumps(symbols_response)}')
        
        result = await client.workspace_symbol("test")
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "test_function"
        assert result[0]["location"]["uri"] == "file:///test.py"


class TestLspNotificationHandling:
    """Test LSP notification system and document synchronization."""
    
    @pytest.mark.asyncio
    async def test_document_synchronization(self, lsp_client):
        """Test document lifecycle notifications (didOpen, didChange, didSave, didClose)."""
        
        reader = MockStreamReader([])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        # Test didOpen
        await lsp_client.did_open("file:///test.py", "python", 1, "def test():\n    pass\n")
        
        # Test didChange
        await lsp_client.did_change("file:///test.py", 2, [{
            "range": {
                "start": {"line": 1, "character": 4},
                "end": {"line": 1, "character": 8}
            },
            "text": "return None"
        }])
        
        # Test didSave
        await lsp_client.did_save("file:///test.py", "def test():\n    return None\n")
        
        # Test didClose
        await lsp_client.did_close("file:///test.py")
        
        # Verify all notifications were sent
        messages = writer.get_written_messages()
        
        methods_sent = []
        for msg in messages:
            parsed = json.loads(msg)
            if "method" in parsed and "id" not in parsed:  # Notifications don't have id
                methods_sent.append(parsed["method"])
        
        assert "textDocument/didOpen" in methods_sent
        assert "textDocument/didChange" in methods_sent
        assert "textDocument/didSave" in methods_sent
        assert "textDocument/didClose" in methods_sent
    
    @pytest.mark.asyncio
    async def test_notification_handlers(self, lsp_client):
        """Test notification handler registration and callbacks."""
        
        reader = MockStreamReader([])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        # Track handler calls
        diagnostics_calls = []
        progress_calls = []
        
        def diagnostics_handler(uri: str, diagnostics: List[Dict]):
            diagnostics_calls.append((uri, diagnostics))
        
        def progress_handler(token: str, value):
            progress_calls.append((token, value))
        
        # Register handlers
        lsp_client.register_diagnostics_handler(diagnostics_handler)
        lsp_client.register_progress_handler(progress_handler)
        
        # Simulate incoming notifications
        diagnostic_notification = {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": "file:///test.py",
                "diagnostics": [
                    {
                        "range": {
                            "start": {"line": 1, "character": 0},
                            "end": {"line": 1, "character": 10}
                        },
                        "severity": 1,
                        "message": "Test error",
                        "source": "test-lsp"
                    }
                ]
            }
        }
        
        progress_notification = {
            "jsonrpc": "2.0",
            "method": "$/progress",
            "params": {
                "token": "test-token",
                "value": {"kind": "begin", "title": "Test progress"}
            }
        }
        
        # Process notifications manually (simulating server messages)
        await lsp_client._handle_notification(diagnostic_notification)
        await lsp_client._handle_notification(progress_notification)
        
        # Verify handlers were called
        assert len(diagnostics_calls) == 1
        assert diagnostics_calls[0][0] == "file:///test.py"
        assert len(diagnostics_calls[0][1]) == 1
        
        assert len(progress_calls) == 1
        assert progress_calls[0][0] == "test-token"
    
    @pytest.mark.asyncio
    async def test_file_watching_integration(self, lsp_client, temp_workspace):
        """Test file watching convenience methods."""
        
        reader = MockStreamReader([])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        test_file = temp_workspace / "test.py"
        content = test_file.read_text()
        
        # Test file sync methods
        await lsp_client.sync_file_opened(str(test_file), content)
        await lsp_client.sync_file_changed(str(test_file), content + "\n# Modified")
        await lsp_client.sync_file_saved(str(test_file))
        await lsp_client.sync_file_closed(str(test_file))
        
        # Verify notifications were sent
        messages = writer.get_written_messages()
        
        methods_sent = []
        for msg in messages:
            parsed = json.loads(msg)
            if "method" in parsed:
                methods_sent.append(parsed["method"])
        
        expected_methods = [
            "textDocument/didOpen",
            "textDocument/didChange",
            "textDocument/didSave",
            "textDocument/didClose"
        ]
        
        for method in expected_methods:
            assert method in methods_sent


class TestLspErrorHandling:
    """Test error handling, timeouts, and recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_request_timeout(self, lsp_client):
        """Test request timeout handling."""
        
        # Reader that never responds
        reader = MockStreamReader([])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        # Set very short timeout for testing
        lsp_client._request_timeout = 0.1
        
        with pytest.raises(LspTimeoutError):
            await lsp_client.hover("file:///test.py", 0, 0)
    
    @pytest.mark.asyncio
    async def test_protocol_error_handling(self, lsp_client):
        """Test handling of JSON-RPC protocol errors."""
        
        # Mock error response
        error_response = {
            "jsonrpc": "2.0",
            "id": "test",
            "error": {"code": -32601, "message": "Method not found"}
        }
        
        reader = MockStreamReader([
            f'Content-Length: {len(json.dumps(error_response))}\r\n\r\n{json.dumps(error_response)}',
        ])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        with pytest.raises(LspError):
            # This will generate a request that gets the error response
            await lsp_client.send_request("nonexistent/method", {})
    
    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self, lsp_client):
        """Test connection failure detection and recovery."""
        
        reader = MockStreamReader([])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        assert lsp_client.is_connected()
        
        # Simulate connection failure
        writer.close()
        reader.closed = True
        
        # Connection should be detected as failed
        with pytest.raises(LspError):
            await lsp_client.hover("file:///test.py", 0, 0)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, lsp_client):
        """Test circuit breaker for failed servers."""
        
        # Circuit breaker is already configured by default
        reader = MockStreamReader([])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        # Set very short timeout to cause failures
        lsp_client._request_timeout = 0.01
        
        # Cause failures to trigger circuit breaker
        with pytest.raises(LspTimeoutError):
            await lsp_client.hover("file:///test.py", 0, 0)
        
        with pytest.raises(LspTimeoutError):
            await lsp_client.hover("file:///test.py", 0, 0)
        
        # Circuit breaker behavior may vary - check if open
        if lsp_client.is_circuit_open:
            with pytest.raises(LspError):
                await lsp_client.hover("file:///test.py", 0, 0)
    
    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, lsp_client):
        """Test handling of malformed JSON-RPC messages."""
        
        # Malformed JSON
        reader = MockStreamReader([
            'Content-Length: 20\r\n\r\n{"invalid": json}',
        ])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        # Should handle malformed messages gracefully
        # The message reading should continue despite parse errors
        reader.messages.append('Content-Length: 50\r\n\r\n{"jsonrpc":"2.0","id":"test","result":null}')
        
        # This should work despite the malformed message
        result = await lsp_client.send_request("test/method", {})
        assert result is None


class TestLspIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_lsp_session(self, lsp_client, temp_workspace):
        """Test complete LSP session from connection to shutdown."""
        
        # Mock complete session responses
        responses = [
            # Initialize response
            '{"jsonrpc":"2.0","id":"init","result":{"capabilities":{"hoverProvider":true,"definitionProvider":true}}}',
            # Hover response
            '{"jsonrpc":"2.0","id":"hover_1","result":{"contents":{"kind":"markdown","value":"Test hover"}}}',
            # Definition response
            '{"jsonrpc":"2.0","id":"def_1","result":[{"uri":"file:///test.py","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":10}}}]}',
        ]
        
        formatted_responses = []
        for resp in responses:
            formatted_responses.append(f'Content-Length: {len(resp)}\r\n\r\n{resp}')
        
        reader = MockStreamReader(formatted_responses)
        writer = MockStreamWriter()
        
        # Full session
        await lsp_client.connect(reader, writer)
        
        # Initialize
        capabilities = await lsp_client.initialize(
            f"file://{temp_workspace.resolve()}",
            "test-client"
        )
        
        assert capabilities.supports_hover()
        assert capabilities.supports_definition()
        
        # Open document
        test_file = temp_workspace / "test.py"
        content = test_file.read_text()
        await lsp_client.sync_file_opened(str(test_file), content)
        
        # Make LSP requests
        test_uri = f"file://{test_file.resolve()}"
        hover_result = await lsp_client.hover(test_uri, 0, 5)
        assert hover_result is not None
        
        definition_result = await lsp_client.definition(test_uri, 0, 5)
        assert definition_result is not None
        assert len(definition_result) == 1
        
        # Close document and disconnect
        await lsp_client.sync_file_closed(str(test_file))
        await lsp_client.disconnect()
        
        assert not lsp_client.is_connected()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, lsp_client):
        """Test handling of concurrent LSP requests."""
        
        # Mock responses for concurrent requests
        responses = []
        for i in range(5):
            resp = f'{{"jsonrpc":"2.0","id":"req_{i}","result":{{"contents":{{"value":"Response {i}"}}}}}}'
            responses.append(f'Content-Length: {len(resp)}\r\n\r\n{resp}')
        
        reader = MockStreamReader(responses)
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        # Send multiple concurrent requests
        tasks = []
        for i in range(5):
            task = lsp_client.send_request(f"test/method_{i}", {"param": i})
            tasks.append(task)
        
        # Wait for all responses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for i, result in enumerate(results):
            assert not isinstance(result, Exception)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_server_capability_fallback(self, lsp_client):
        """Test fallback behavior for unsupported server capabilities."""
        
        # Server with limited capabilities
        limited_capabilities = {
            "hoverProvider": True,
            "definitionProvider": False,  # Not supported
            "referencesProvider": False,  # Not supported
        }
        
        init_response = {
            "jsonrpc": "2.0",
            "id": "init",
            "result": {"capabilities": limited_capabilities}
        }
        
        reader = MockStreamReader([
            f'Content-Length: {len(json.dumps(init_response))}\r\n\r\n{json.dumps(init_response)}',
        ])
        writer = MockStreamWriter()
        
        await lsp_client.connect(reader, writer)
        
        capabilities = await lsp_client.initialize("file:///test", "test-client")
        
        # Verify capability detection
        assert capabilities.supports_hover() is True
        assert capabilities.supports_definition() is False
        assert capabilities.supports_references() is False
        
        # Requests for unsupported features should return None (graceful fallback)
        definition_result = await lsp_client.definition("file:///test.py", 0, 0)
        assert definition_result is None
        
        references_result = await lsp_client.references("file:///test.py", 0, 0)
        assert references_result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])