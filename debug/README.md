# MCP Inspector Debugging Tools

This directory contains debugging tools and configurations for the workspace-qdrant-mcp server using the official MCP Inspector React-based debugging tool.

## Overview

The MCP Inspector provides real-time protocol monitoring and debugging capabilities for MCP servers. It includes:

- **Web UI**: React-based interface for interactive debugging at `http://localhost:6274`
- **CLI Interface**: Command-line tools for automated testing and inspection
- **Multiple Transports**: Support for STDIO, HTTP, and SSE connections
- **Real-time Monitoring**: Live protocol message inspection and analysis

## Quick Start

### 1. Web UI Debugging (Recommended)

Start the interactive debugging interface:

```bash
# Debug using STDIO transport (most common)
./debug-stdio.sh

# Debug using HTTP transport (requires server running on port 8000)
./debug-http.sh
```

Open your browser to `http://localhost:6274` to access the debugging interface.

### 2. Command-Line Testing

Test server connectivity and tools:

```bash
# Run comprehensive tool testing
./test-tools.sh

# List all available MCP tools
npm run list-tools

# List all available resources
npm run list-resources

# Test specific tool (example)
npx @modelcontextprotocol/inspector --cli --config mcp-inspector-config.json --server workspace-qdrant-stdio --method tools/call --tool-name workspace_status
```

## Transport Modes

### STDIO Mode (Default)
- **Use case**: Direct server debugging, development
- **Command**: `./debug-stdio.sh`
- **Configuration**: Automatically starts the server with STDIO transport

### HTTP Mode
- **Use case**: Web client debugging, production testing
- **Setup**:
  1. Start HTTP server: `uv run python -m workspace_qdrant_mcp.server --transport http --port 8000`
  2. Run debugger: `./debug-http.sh`
- **Configuration**: Connects to running HTTP server on port 8000

### SSE Mode
- **Use case**: Server-Sent Events debugging
- **Setup**: Similar to HTTP mode but with SSE transport
- **Configuration**: Uses SSE endpoint for real-time updates

## Configuration Files

### `mcp-inspector-config.json`
Main configuration file defining server connections:

```json
{
  "mcpServers": {
    "workspace-qdrant-stdio": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "workspace_qdrant_mcp.server"],
      "env": {
        "WQM_STDIO_MODE": "true",
        "WQM_LOG_LEVEL": "DEBUG",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

### `package.json`
NPM configuration with debugging scripts:

- `debug-stdio`: Start UI with STDIO transport
- `debug-http`: Start UI with HTTP transport
- `debug-cli-stdio`: CLI mode with STDIO
- `list-tools`: List all available MCP tools
- `test-connection`: Test server connectivity

## Available Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `debug-stdio.sh` | Web UI with STDIO transport | `./debug-stdio.sh` |
| `debug-http.sh` | Web UI with HTTP transport | `./debug-http.sh` |
| `test-tools.sh` | Comprehensive CLI testing | `./test-tools.sh` |

## MCP Tools Available for Debugging

The workspace-qdrant-mcp server provides 30+ MCP tools that can be debugged:

### Core Document Management
- `add_document` - Add documents to collections
- `get_document` - Retrieve specific documents
- `add_document_with_project_context` - Add with project metadata

### Search Operations
- `search_workspace` - Basic workspace search
- `search_workspace_by_project` - Project-scoped search
- `hybrid_search_advanced` - Advanced hybrid search with filtering

### Collection Management
- `list_collections` - List all available collections
- `create_workspace_collection` - Create new collections
- `get_workspace_collection_info` - Get collection metadata

### Multi-tenant Operations
- `initialize_project_workspace_collections` - Set up project workspaces
- `search_memory_collections` - Search across memory collections

### System Tools
- `workspace_status` - Get server status and health
- `get_server_info` - Detailed server information

## Debugging Workflow

### 1. Start Debugging Session

```bash
# Start the web interface
./debug-stdio.sh
```

### 2. Explore Server Capabilities

1. Open `http://localhost:6274` in your browser
2. Click "Connect" to establish connection
3. Explore the "Tools" tab to see all available MCP tools
4. Check the "Resources" tab for available resources

### 3. Test Individual Tools

1. Select a tool from the Tools tab
2. Fill in required parameters
3. Click "Call Tool" to execute
4. View the response and any errors in real-time

### 4. Monitor Protocol Messages

1. Switch to the "Messages" tab
2. Watch real-time MCP protocol messages
3. Analyze request/response patterns
4. Debug any protocol-level issues

### 5. CLI Testing

```bash
# Test specific functionality
npm run list-tools
npm run test-connection

# Call specific tools
npx @modelcontextprotocol/inspector --cli \
  --config mcp-inspector-config.json \
  --server workspace-qdrant-stdio \
  --method tools/call \
  --tool-name workspace_status
```

## Troubleshooting

### Connection Issues

1. **STDIO Mode**: Ensure Python environment is activated and dependencies installed
2. **HTTP Mode**: Verify the HTTP server is running on the correct port
3. **Port Conflicts**: Change ports using environment variables:
   ```bash
   CLIENT_PORT=8080 SERVER_PORT=9000 ./debug-stdio.sh
   ```

### Common Problems

- **Server not starting**: Check Python path and virtual environment
- **Tool failures**: Verify Qdrant server is running on `localhost:6333`
- **Protocol errors**: Check server logs for detailed error messages

### Environment Variables

- `WQM_STDIO_MODE=true`: Enable STDIO mode
- `WQM_LOG_LEVEL=DEBUG`: Increase logging verbosity
- `QDRANT_URL`: Qdrant server URL (default: `http://localhost:6333`)
- `CLIENT_PORT`: Inspector UI port (default: 6274)
- `SERVER_PORT`: Inspector proxy port (default: 6277)

## Integration with Testing

The MCP Inspector can be integrated with the existing test suite:

```bash
# Run in CI/CD pipelines for protocol validation
npm run test-connection

# Automated tool testing
npm run list-tools | grep -q "workspace_status" && echo "Tools available"

# Integration with pytest
cd .. && uv run pytest tests/integration/test_mcp_inspector.py
```

## Advanced Usage

### Custom Server Configuration

Modify `mcp-inspector-config.json` to add custom servers:

```json
{
  "mcpServers": {
    "custom-server": {
      "type": "stdio",
      "command": "python",
      "args": ["custom_server.py"],
      "env": {
        "CUSTOM_VAR": "value"
      }
    }
  }
}
```

### Remote Debugging

For remote server debugging:

```bash
npx @modelcontextprotocol/inspector --cli https://your-server.com --transport http --method tools/list
```

### Performance Monitoring

Use the Inspector to monitor performance:

1. Enable detailed logging in server configuration
2. Monitor message timing in the Messages tab
3. Analyze tool execution times
4. Identify bottlenecks in protocol communication

## Security Notes

- The Inspector proxy runs on localhost by default for security
- For remote access, use SSH tunneling or proper authentication
- Never expose the Inspector to untrusted networks
- Review server configurations before sharing

## Support

For issues with:
- **MCP Inspector**: [Official Repository](https://github.com/modelcontextprotocol/inspector)
- **workspace-qdrant-mcp**: Check project documentation and logs
- **Protocol errors**: Enable debug logging and check message patterns