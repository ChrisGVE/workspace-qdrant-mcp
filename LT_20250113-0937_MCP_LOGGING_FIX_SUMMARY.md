# MCP Logging Interference Fix - Implementation Summary

## Problem

The workspace-qdrant-mcp server was outputting JSON log messages to stderr even in stdio mode with LOG_LEVEL=CRITICAL, which interfered with the MCP JSON-RPC protocol communication that Claude Code expects on clean stdin/stdout channels.

## Root Cause

- Console logging was enabled by default in all transport modes
- Even when using stderr in stdio mode, log messages were still being output
- No mechanism existed to completely disable console output for MCP protocol compliance

## Solution Implemented

### 1. Environment Variables Added

- **`MCP_QUIET_MODE`**: Primary control for MCP console logging
  - Default: `true` when `transport=stdio`, `false` otherwise
  - Set to `false` to force enable console logging in stdio mode

- **`DISABLE_MCP_CONSOLE_LOGS`**: Alternative explicit disable control
  - Default: `false`
  - Set to `true` to disable console logging in any mode

- **`WQM_STDIO_MODE`**: Internal flag set by server when in stdio transport
  - Used by other modules to detect MCP mode

### 2. Logging Configuration Changes

**File: `src/python/common/observability/logger.py`**

- Modified `configure_logging()` function to respect MCP quiet mode
- Added conditional console handler creation logic
- Console output disabled in stdio mode unless explicitly overridden
- Preserved stderr usage in stdio mode when logging is enabled

**File: `src/python/workspace_qdrant_mcp/server.py`**

- Set `WQM_STDIO_MODE=true` when transport is stdio
- Set `MCP_QUIET_MODE=true` by default for stdio transport
- Added automatic log file creation in `~/.workspace-qdrant-mcp/logs/` for stdio mode
- Silenced startup log messages in stdio mode unless quiet mode is disabled

### 3. Behavior Matrix

| Transport | MCP_QUIET_MODE | DISABLE_MCP_CONSOLE_LOGS | Console Output | Log File |
|-----------|----------------|--------------------------|----------------|----------|
| stdio     | true (default) | false (default)          | ❌ Disabled     | ✅ Auto-created |
| stdio     | false          | false                    | ✅ Enabled (stderr) | ✅ Auto-created |
| stdio     | any            | true                     | ❌ Disabled     | ✅ Auto-created |
| http      | any            | false (default)          | ✅ Enabled (stdout) | ⚠️ If configured |
| http      | any            | true                     | ❌ Disabled     | ⚠️ If configured |

## Files Modified

1. **`src/python/common/observability/logger.py`**
   - Added MCP quiet mode detection logic
   - Modified console handler creation conditions
   - Updated environment variable handling

2. **`src/python/workspace_qdrant_mcp/server.py`**
   - Added stdio mode environment variable setup
   - Added automatic log file configuration for stdio mode
   - Silenced startup messages in quiet mode
   - Updated documentation with new environment variables

## Testing Results

✅ **stdio mode (default)**: No console output, clean MCP protocol communication
✅ **stdio mode with MCP_QUIET_MODE=false**: Console logging enabled to stderr
✅ **HTTP mode**: Normal console output as expected
✅ **Log files**: Automatically created in stdio mode for debugging

## Usage Examples

### Claude Code (Default - Clean MCP Protocol)
```bash
workspace-qdrant-mcp --transport stdio
# No console output, logs to ~/.workspace-qdrant-mcp/logs/server.log
```

### Debug Mode (Enable Console Logging)
```bash
MCP_QUIET_MODE=false workspace-qdrant-mcp --transport stdio
# Console logs to stderr, MCP JSON-RPC on stdout/stdin
```

### HTTP Web Client
```bash
workspace-qdrant-mcp --transport http --port 8000
# Normal console logging as before
```

### Explicit Console Disable
```bash
DISABLE_MCP_CONSOLE_LOGS=true workspace-qdrant-mcp --transport http
# Disable console even in HTTP mode
```

## Benefits

1. **MCP Protocol Compliance**: Clean stdin/stdout for JSON-RPC communication
2. **Debugging Preserved**: File logging automatically enabled in stdio mode
3. **Flexible Control**: Multiple environment variables for different use cases
4. **Backward Compatibility**: HTTP mode behavior unchanged by default
5. **Troubleshooting Support**: Can re-enable console logging when needed

## Migration Notes

- **No breaking changes**: Existing configurations continue to work
- **Default behavior**: stdio mode is now quiet by default (as expected for MCP)
- **Log file location**: `~/.workspace-qdrant-mcp/logs/server.log` for stdio mode
- **Environment variables**: New variables are optional with sensible defaults