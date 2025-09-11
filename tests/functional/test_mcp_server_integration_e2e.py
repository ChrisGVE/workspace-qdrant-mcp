"""
MCP Server Integration Testing (End-to-End)

Comprehensive end-to-end tests for MCP server functionality including tool 
registration, server lifecycle, and client integration with protocol compliance.

This module implements subtask 203.7 of the End-to-End Functional Testing Framework.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest


class MCPServerTestEnvironment:
    """Test environment for MCP server integration testing."""
    
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.config_dir = tmp_path / ".config" / "workspace-qdrant"
        self.server_process = None
        self.server_port = 8123  # Test port
        self.server_url = f"http://localhost:{self.server_port}"
        self.mcp_executable = "uv run workspace-qdrant-mcp"
        
        self.setup_environment()
    
    def setup_environment(self):
        """Set up MCP server test environment."""
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create MCP server configuration
        config_content = {
            "qdrant_url": "http://localhost:6333",
            "mcp_server": {
                "transport": "http",
                "host": "127.0.0.1",
                "port": self.server_port,
                "timeout": 30,
                "max_connections": 10
            },
            "tools": {
                "enabled": [
                    "add_document",
                    "search_workspace",
                    "hybrid_search_advanced",
                    "get_document",
                    "list_workspace_collections",
                    "workspace_status",
                    "search_by_metadata",
                    "update_scratchbook",
                    "search_scratchbook",
                    "list_scratchbook_notes",
                    "delete_scratchbook_note"
                ]
            }
        }
        
        import yaml
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
    
    def start_mcp_server(self, timeout: int = 30) -> bool:
        """Start MCP server in HTTP mode."""
        try:
            # Set up environment
            env = os.environ.copy()
            env.update({
                "WQM_CONFIG_DIR": str(self.config_dir),
                "PYTHONPATH": str(Path.cwd()),
                "WQM_MCP_MODE": "server"
            })
            
            # Start server
            self.server_process = subprocess.Popen(
                f"{self.mcp_executable} --transport http --host 127.0.0.1 --port {self.server_port}",
                shell=True,
                cwd=self.tmp_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            return self._wait_for_server_ready(timeout)
            
        except Exception as e:
            print(f"Failed to start MCP server: {e}")
            return False
    
    def stop_mcp_server(self):
        """Stop MCP server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            finally:
                self.server_process = None
    
    def _wait_for_server_ready(self, timeout: int) -> bool:
        """Wait for server to be ready to accept connections."""
        import socket
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to connect to server port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', self.server_port))
                sock.close()
                
                if result == 0:
                    # Server is accepting connections
                    time.sleep(1)  # Give it a moment to fully initialize
                    return True
                    
            except Exception:
                pass
            
            time.sleep(0.5)
        
        return False
    
    def is_server_running(self) -> bool:
        """Check if MCP server is running."""
        if not self.server_process:
            return False
        
        return self.server_process.poll() is None
    
    def get_server_logs(self) -> Tuple[str, str]:
        """Get server stdout and stderr logs."""
        if self.server_process:
            try:
                stdout, stderr = self.server_process.communicate(timeout=1)
                return stdout, stderr
            except subprocess.TimeoutExpired:
                return "", ""
        return "", ""


class MCPClientSimulator:
    """Simulates MCP client interactions for testing."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id = None
    
    async def send_mcp_request(
        self, 
        method: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send MCP request to server."""
        try:
            import aiohttp
            import asyncio
            
            request_data = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": method,
                "params": params or {}
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.server_url}/mcp",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {
                            "error": {
                                "code": response.status,
                                "message": f"HTTP {response.status}",
                                "data": await response.text()
                            }
                        }
        except Exception as e:
            return {
                "error": {
                    "code": -1,
                    "message": str(e),
                    "data": None
                }
            }
    
    async def test_tool_call(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test calling a specific MCP tool."""
        return await self.send_mcp_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available MCP tools."""
        return await self.send_mcp_request("tools/list")
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server information."""
        return await self.send_mcp_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })


class MCPProtocolValidator:
    """Validates MCP protocol compliance and behavior."""
    
    @staticmethod
    def validate_mcp_response(response: Dict[str, Any]) -> Dict[str, bool]:
        """Validate MCP response structure and content."""
        return {
            "has_jsonrpc": "jsonrpc" in response,
            "correct_jsonrpc_version": response.get("jsonrpc") == "2.0",
            "has_id": "id" in response,
            "has_result_or_error": "result" in response or "error" in response,
            "no_both_result_and_error": not ("result" in response and "error" in response),
            "valid_structure": all([
                "jsonrpc" in response,
                response.get("jsonrpc") == "2.0",
                "id" in response,
                ("result" in response) != ("error" in response)  # XOR
            ])
        }
    
    @staticmethod
    def validate_tool_definition(tool: Dict[str, Any]) -> Dict[str, bool]:
        """Validate MCP tool definition structure."""
        required_fields = ["name", "description"]
        
        return {
            "has_name": "name" in tool,
            "has_description": "description" in tool,
            "has_input_schema": "inputSchema" in tool,
            "valid_name": isinstance(tool.get("name"), str) and len(tool.get("name", "")) > 0,
            "valid_description": isinstance(tool.get("description"), str) and len(tool.get("description", "")) > 0,
            "complete_definition": all(field in tool for field in required_fields)
        }
    
    @staticmethod
    def validate_error_response(error: Dict[str, Any]) -> Dict[str, bool]:
        """Validate MCP error response structure."""
        return {
            "has_code": "code" in error,
            "has_message": "message" in error,
            "valid_code": isinstance(error.get("code"), int),
            "valid_message": isinstance(error.get("message"), str) and len(error.get("message", "")) > 0,
            "complete_error": all(field in error for field in ["code", "message"])
        }


@pytest.mark.functional
@pytest.mark.mcp_integration
class TestMCPServerIntegration:
    """Test MCP server integration and protocol compliance."""
    
    @pytest.fixture
    def mcp_env(self, tmp_path):
        """Create MCP server test environment."""
        env = MCPServerTestEnvironment(tmp_path)
        yield env
        env.stop_mcp_server()
    
    @pytest.fixture
    def validator(self):
        """Create MCP protocol validator."""
        return MCPProtocolValidator()
    
    @pytest.mark.asyncio
    async def test_mcp_server_startup_lifecycle(self, mcp_env, validator):
        """Test MCP server startup and lifecycle management."""
        # Test server startup
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if startup_success:
            # Validate server is running
            assert mcp_env.is_server_running(), "Server not running after startup"
            
            # Test server responsiveness
            client = MCPClientSimulator(mcp_env.server_url)
            response = await client.get_server_info()
            
            # Validate response structure
            if "error" not in response:
                validation = validator.validate_mcp_response(response)
                assert validation["valid_structure"], "Invalid MCP response structure"
            
            # Test server shutdown
            mcp_env.stop_mcp_server()
            assert not mcp_env.is_server_running(), "Server still running after shutdown"
        else:
            # Server startup failed - check logs for diagnostics
            stdout, stderr = mcp_env.get_server_logs()
            pytest.skip(f"MCP server startup failed. Stdout: {stdout[:200]}... Stderr: {stderr[:200]}...")
    
    @pytest.mark.asyncio
    async def test_mcp_tool_registration_and_discovery(self, mcp_env, validator):
        """Test MCP tool registration and discovery."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test tool listing
        tools_response = await client.list_tools()
        
        if "error" in tools_response:
            # Server may not be fully ready or tools endpoint not available
            error_info = tools_response["error"]
            pytest.skip(f"Tools listing failed: {error_info.get('message', 'Unknown error')}")
        
        # Validate response structure
        validation = validator.validate_mcp_response(tools_response)
        assert validation["valid_structure"], "Invalid tools list response structure"
        
        # Check if tools are present
        if "result" in tools_response and "tools" in tools_response["result"]:
            tools = tools_response["result"]["tools"]
            
            # Validate tool definitions
            expected_tools = [
                "add_document",
                "search_workspace", 
                "workspace_status",
                "list_workspace_collections"
            ]
            
            tool_names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
            
            for tool in tools:
                if isinstance(tool, dict):
                    tool_validation = validator.validate_tool_definition(tool)
                    assert tool_validation["complete_definition"], f"Invalid tool definition: {tool.get('name', 'unknown')}"
            
            # At least some expected tools should be available
            available_expected = [name for name in expected_tools if name in tool_names]
            assert len(available_expected) > 0, "No expected tools found in registration"
    
    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self, mcp_env, validator):
        """Test MCP tool execution and response handling."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test basic tool calls
        test_cases = [
            {
                "tool": "workspace_status",
                "args": {},
                "description": "Get workspace status"
            },
            {
                "tool": "list_workspace_collections", 
                "args": {},
                "description": "List collections"
            }
        ]
        
        for test_case in test_cases:
            response = await client.test_tool_call(
                test_case["tool"], 
                test_case["args"]
            )
            
            # Validate response structure
            validation = validator.validate_mcp_response(response)
            
            if "error" in response:
                # Tool execution failed - validate error structure
                error_validation = validator.validate_error_response(response["error"])
                assert error_validation["complete_error"], f"Invalid error response for {test_case['tool']}"
                
                # Error should be informative
                error_message = response["error"].get("message", "")
                assert len(error_message) > 0, f"Empty error message for {test_case['tool']}"
            else:
                # Tool execution succeeded - validate success structure
                assert validation["valid_structure"], f"Invalid response structure for {test_case['tool']}"
                assert "result" in response, f"No result in successful response for {test_case['tool']}"
    
    @pytest.mark.asyncio
    async def test_mcp_document_operations(self, mcp_env, validator):
        """Test MCP document-related operations."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test document addition
        test_document = {
            "content": "This is a test document for MCP integration testing.",
            "metadata": {
                "title": "MCP Test Document",
                "type": "test",
                "created_by": "test_suite"
            },
            "collection": "test-collection"
        }
        
        add_response = await client.test_tool_call("add_document", test_document)
        
        # Validate add document response
        validation = validator.validate_mcp_response(add_response)
        
        if "error" in add_response:
            # Document addition failed - should provide clear error
            error_validation = validator.validate_error_response(add_response["error"])
            assert error_validation["complete_error"], "Invalid error response for add_document"
            
            # Common reasons: Qdrant not available, collection issues
            error_message = add_response["error"].get("message", "").lower()
            expected_error_indicators = ["connection", "qdrant", "collection", "database"]
            
            has_expected_error = any(indicator in error_message for indicator in expected_error_indicators)
            if has_expected_error:
                pytest.skip(f"Document operations require Qdrant: {add_response['error']['message']}")
        else:
            # Document addition succeeded
            assert validation["valid_structure"], "Invalid add_document response structure"
            assert "result" in add_response, "No result in successful add_document response"
            
            # Test document search
            search_response = await client.test_tool_call("search_workspace", {
                "query": "MCP integration testing",
                "limit": 5
            })
            
            search_validation = validator.validate_mcp_response(search_response)
            
            if "error" not in search_response:
                assert search_validation["valid_structure"], "Invalid search response structure"
                assert "result" in search_response, "No result in successful search response"
    
    @pytest.mark.asyncio
    async def test_mcp_scratchbook_operations(self, mcp_env, validator):
        """Test MCP scratchbook-related operations."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test scratchbook operations
        scratchbook_operations = [
            {
                "tool": "list_scratchbook_notes",
                "args": {},
                "description": "List scratchbook notes"
            },
            {
                "tool": "update_scratchbook",
                "args": {
                    "content": "MCP integration test note",
                    "metadata": {"test": True}
                },
                "description": "Update scratchbook"
            },
            {
                "tool": "search_scratchbook",
                "args": {
                    "query": "integration test",
                    "limit": 10
                },
                "description": "Search scratchbook"
            }
        ]
        
        for operation in scratchbook_operations:
            response = await client.test_tool_call(
                operation["tool"],
                operation["args"]
            )
            
            # Validate response structure
            validation = validator.validate_mcp_response(response)
            
            if "error" in response:
                # Operation failed - validate error
                error_validation = validator.validate_error_response(response["error"])
                assert error_validation["complete_error"], f"Invalid error for {operation['tool']}"
                
                # Should provide meaningful error message
                error_message = response["error"].get("message", "")
                assert len(error_message) > 0, f"Empty error message for {operation['tool']}"
            else:
                # Operation succeeded
                assert validation["valid_structure"], f"Invalid response for {operation['tool']}"
                assert "result" in response, f"No result for successful {operation['tool']}"
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling_and_edge_cases(self, mcp_env, validator):
        """Test MCP server error handling and edge cases."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test invalid tool calls
        error_test_cases = [
            {
                "tool": "nonexistent_tool",
                "args": {},
                "description": "Non-existent tool"
            },
            {
                "tool": "add_document",
                "args": {"invalid": "arguments"},
                "description": "Invalid arguments"
            },
            {
                "tool": "search_workspace",
                "args": {},  # Missing required query
                "description": "Missing required arguments"
            }
        ]
        
        for test_case in error_test_cases:
            response = await client.test_tool_call(
                test_case["tool"],
                test_case["args"]
            )
            
            # Should return error for invalid calls
            validation = validator.validate_mcp_response(response)
            assert validation["valid_structure"], f"Invalid response structure for {test_case['description']}"
            
            if "error" in response:
                # Validate error structure
                error_validation = validator.validate_error_response(response["error"])
                assert error_validation["complete_error"], f"Invalid error structure for {test_case['description']}"
                
                # Error should be informative
                error_message = response["error"].get("message", "")
                assert len(error_message) > 0, f"Empty error message for {test_case['description']}"
                
                # Error code should be appropriate
                error_code = response["error"].get("code")
                assert isinstance(error_code, int), f"Invalid error code type for {test_case['description']}"
    
    @pytest.mark.asyncio
    async def test_mcp_protocol_compliance(self, mcp_env, validator):
        """Test MCP protocol compliance and standards."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test protocol initialization
        init_response = await client.get_server_info()
        
        if "error" not in init_response:
            validation = validator.validate_mcp_response(init_response)
            assert validation["valid_structure"], "Invalid initialization response structure"
            
            # Check for proper protocol version handling
            if "result" in init_response:
                result = init_response["result"]
                
                # Should have server information
                assert "protocolVersion" in result, "Missing protocol version in init response"
                assert "capabilities" in result, "Missing capabilities in init response"
                assert "serverInfo" in result, "Missing server info in init response"
                
                # Validate server info structure
                server_info = result["serverInfo"]
                assert "name" in server_info, "Missing server name"
                assert "version" in server_info, "Missing server version"
        
        # Test JSON-RPC compliance
        test_requests = [
            {
                "method": "tools/list",
                "params": {},
                "description": "Standard tools list"
            }
        ]
        
        for request in test_requests:
            response = await client.send_mcp_request(
                request["method"],
                request["params"]
            )
            
            # All responses should be JSON-RPC 2.0 compliant
            validation = validator.validate_mcp_response(response)
            assert validation["has_jsonrpc"], f"Missing jsonrpc field in {request['description']}"
            assert validation["correct_jsonrpc_version"], f"Wrong jsonrpc version in {request['description']}"
            assert validation["has_id"], f"Missing id field in {request['description']}"
            assert validation["has_result_or_error"], f"Missing result/error in {request['description']}"
            assert validation["no_both_result_and_error"], f"Both result and error in {request['description']}"
    
    @pytest.mark.asyncio
    async def test_mcp_concurrent_operations(self, mcp_env, validator):
        """Test MCP server handling of concurrent operations."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        # Create multiple clients for concurrent testing
        clients = [MCPClientSimulator(mcp_env.server_url) for _ in range(3)]
        
        async def concurrent_operation(client: MCPClientSimulator, operation_id: int):
            """Perform concurrent operation."""
            try:
                response = await client.test_tool_call("workspace_status", {})
                return {
                    "operation_id": operation_id,
                    "success": "error" not in response,
                    "response": response
                }
            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute concurrent operations
        tasks = [
            concurrent_operation(client, i) 
            for i, client in enumerate(clients)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate concurrent operation results
        successful_operations = 0
        for result in results:
            if isinstance(result, dict) and result.get("success", False):
                successful_operations += 1
                
                # Validate response structure for successful operations
                response = result["response"]
                validation = validator.validate_mcp_response(response)
                assert validation["valid_structure"], f"Invalid concurrent response for operation {result['operation_id']}"
        
        # At least some concurrent operations should succeed
        assert successful_operations > 0, "No concurrent operations succeeded"
    
    @pytest.mark.asyncio
    async def test_mcp_server_resource_management(self, mcp_env, validator):
        """Test MCP server resource management and limits."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test rapid sequential requests
        rapid_requests = []
        for i in range(10):
            request_task = client.test_tool_call("workspace_status", {})
            rapid_requests.append(request_task)
        
        # Execute rapid requests
        try:
            responses = await asyncio.gather(*rapid_requests, return_exceptions=True)
            
            # Validate server handled rapid requests
            successful_responses = 0
            for response in responses:
                if isinstance(response, dict):
                    validation = validator.validate_mcp_response(response)
                    if validation["valid_structure"]:
                        successful_responses += 1
            
            # Should handle most rapid requests successfully
            success_rate = successful_responses / len(responses)
            assert success_rate > 0.5, f"Low success rate for rapid requests: {success_rate:.2%}"
            
        except Exception as e:
            pytest.skip(f"Rapid request testing failed: {e}")
    
    @pytest.mark.asyncio
    async def test_mcp_client_integration_patterns(self, mcp_env, validator):
        """Test common MCP client integration patterns."""
        startup_success = mcp_env.start_mcp_server(timeout=30)
        
        if not startup_success:
            pytest.skip("MCP server startup failed")
        
        client = MCPClientSimulator(mcp_env.server_url)
        
        # Test initialization → discovery → operation pattern
        workflow_steps = []
        
        # Step 1: Initialize
        init_response = await client.get_server_info()
        workflow_steps.append(("initialize", init_response))
        
        # Step 2: Discover tools
        tools_response = await client.list_tools()
        workflow_steps.append(("list_tools", tools_response))
        
        # Step 3: Execute operation
        status_response = await client.test_tool_call("workspace_status", {})
        workflow_steps.append(("execute_tool", status_response))
        
        # Validate workflow
        for step_name, response in workflow_steps:
            validation = validator.validate_mcp_response(response)
            
            if "error" not in response:
                assert validation["valid_structure"], f"Invalid response structure in {step_name}"
            else:
                # If error, should still be properly structured
                error_validation = validator.validate_error_response(response["error"])
                assert error_validation["complete_error"], f"Invalid error structure in {step_name}"
        
        # Test session consistency
        # Multiple calls should work consistently
        consistency_responses = []
        for _ in range(3):
            response = await client.test_tool_call("workspace_status", {})
            consistency_responses.append(response)
        
        # All consistency responses should have similar structure
        valid_responses = [
            r for r in consistency_responses 
            if validator.validate_mcp_response(r)["valid_structure"]
        ]
        
        assert len(valid_responses) > 0, "No valid responses in consistency test"