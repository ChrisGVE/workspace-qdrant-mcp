# Common Issues

## Overview
This troubleshooting guide covers the most frequent issues users encounter with workspace-qdrant-mcp and their solutions.

## Installation Issues

### Package Installation Problems
**Issue**: Installation fails with permission errors
**Solution**: 
```bash
# Use user installation
pip install --user workspace-qdrant-mcp
# Or with uv
uv tool install --user workspace-qdrant-mcp
```

### Command Not Found
**Issue**: `workspace-qdrant-mcp` command not found after installation
**Solution**:
```bash
# Check PATH includes installation directory
which workspace-qdrant-mcp
# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

## Connection Issues

### Qdrant Server Unreachable
**Issue**: `Connection refused to localhost:6333`
**Diagnosis**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections
# Start Qdrant if needed
docker run -p 6333:6333 qdrant/qdrant
```

### MCP Integration Not Working
**Issue**: Claude doesn't recognize MCP tools
**Solution**:
```bash
# Restart Claude Desktop completely
# Verify configuration
cat ~/.config/claude-desktop/claude_desktop_config.json
# Test connection
workspace-qdrant-test
```

## Search and Performance Issues

### No Search Results
**Issue**: Searches return empty results despite having content
**Diagnosis**:
```bash
wqutil list-collections
wqutil workspace-status
workspace-qdrant-test --component search
```

### Slow Performance
**Issue**: Long response times during searches
**Solution**:
```bash
workspace-qdrant-health --analyze
wqutil optimize-collection my-project-scratchbook
```

## Collection Issues

### Collections Not Created
**Issue**: Expected collections don't exist
**Diagnosis**:
```bash
# Check project detection
wqutil workspace-status
# Verify environment variables
echo $COLLECTIONS $GLOBAL_COLLECTIONS
```

### Wrong Project Name
**Issue**: Collections have unexpected names
**Solution**: Collections use sanitized directory names. Check current directory and Git repository name.

## Quick Diagnostic Commands

```bash
# Full system check
workspace-qdrant-test

# Health monitoring
workspace-qdrant-health

# Project status
wqutil workspace-status

# Collection status
wqutil list-collections
```

For detailed troubleshooting, see the other guides in this section.