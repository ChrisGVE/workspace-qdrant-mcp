# Configuration Examples

This directory contains example configurations for using workspace-qdrant-mcp with Claude Desktop and other MCP clients.

## Claude Desktop Configuration

### Production Setup (`claude_desktop_config.json`)

For production use with UV tool installation:

```bash
# Install the server as a UV tool
uv tool install workspace-qdrant-mcp
```

Then add the configuration to your Claude Desktop config file (usually at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS).

### Development Setup (`claude_desktop_dev_config.json`)

For development when working with the source code:

1. Clone the repository
2. Set up the development environment:
   ```bash
   cd workspace-qdrant-mcp
   uv sync
   ```
3. Update the `cwd` and `PYTHONPATH` in the config to point to your local repository
4. Add the configuration to your Claude Desktop config file

## Important Notes

### Stdio Transport (Default)

The workspace-qdrant-mcp server now correctly uses **stdio transport** by default, which is the proper way to communicate with Claude Desktop and other MCP clients. The server reads JSON-RPC messages from stdin and writes responses to stdout.

**No port configuration is needed for MCP communication** - ports are only used when explicitly running in HTTP mode for web-based clients.

### Environment Variables

Required environment variables:
- `QDRANT_URL`: URL of your Qdrant server (e.g., `http://localhost:6333`)
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings (if using OpenAI models)

Optional environment variables:
- `LOG_LEVEL`: Set to `DEBUG` for verbose logging
- `CONFIG_FILE`: Path to custom configuration file

### Testing the Configuration

After adding the configuration to Claude Desktop:

1. Restart Claude Desktop
2. Start a new conversation
3. You should see the workspace-qdrant-mcp tools available in the tool panel
4. Try using tools like "workspace_status" to verify the connection

### Troubleshooting

If the server doesn't appear in Claude Desktop:

1. Check the Claude Desktop logs for error messages
2. Verify your Qdrant server is running and accessible
3. Ensure all required environment variables are set
4. Test the server manually:
   ```bash
   # For UV installation
   workspace-qdrant-mcp --transport stdio
   
   # For development
   cd workspace-qdrant-mcp
   python -m workspace_qdrant_mcp.server --transport stdio
   ```

The server should start and wait for JSON-RPC input on stdin. You can test it by sending:
```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "0.1.0", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
```
