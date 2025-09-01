# Installation and Setup

## Objectives
- Install workspace-qdrant-mcp successfully
- Set up Qdrant server (local or cloud)
- Complete initial configuration
- Verify installation works correctly

## Prerequisites
- Python 3.10+ installed
- Basic command line knowledge
- Administrative access to install software

## Overview
This tutorial will take you from a fresh system to a fully working workspace-qdrant-mcp installation. We'll use the interactive setup wizard to handle most configuration automatically.

**Estimated time**: 15-20 minutes

## Step 1: Install Qdrant Server

workspace-qdrant-mcp requires a running Qdrant server. Choose your preferred method:

### Option A: Local Docker Installation (Recommended)

```bash
# Start Qdrant with Docker
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Verify Qdrant is running
curl http://localhost:6333/collections
```

**Expected output**: Empty JSON array `[]`

### Option B: Qdrant Cloud (For production/team use)

1. Visit [Qdrant Cloud](https://cloud.qdrant.io)
2. Create an account and cluster
3. Note your cluster URL and API key
4. Set environment variable: `export QDRANT_API_KEY="your-api-key-here"`

### Option C: Local Binary Installation

```bash
# Download and install Qdrant binary (Linux/macOS)
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant
chmod +x qdrant
./qdrant
```

## Step 2: Install workspace-qdrant-mcp

Choose your preferred package manager:

### Using uv (Recommended)

```bash
# Install globally with uv
uv tool install workspace-qdrant-mcp

# Verify installation
workspace-qdrant-mcp --version
```

### Using pip

```bash
# Install with pip
pip install workspace-qdrant-mcp

# Verify installation
workspace-qdrant-mcp --version
```

**Expected output**: Version number like `workspace-qdrant-mcp 1.0.0`

## Step 3: Run the Setup Wizard

The interactive setup wizard will configure everything automatically:

```bash
workspace-qdrant-setup
```

The wizard will:

1. **Test Qdrant Connection**
   ```
   🔍 Testing Qdrant connection...
   ✅ Qdrant server accessible at http://localhost:6333
   ✅ Server version: 1.7.0
   ```

2. **Configure Embedding Model**
   ```
   📐 Choose embedding model:
   1) sentence-transformers/all-MiniLM-L6-v2 (lightweight, 384D)
   2) BAAI/bge-base-en-v1.5 (balanced, 768D)
   3) BAAI/bge-large-en-v1.5 (high quality, 1024D)
   
   Enter choice [1-3]: 1
   ✅ Selected: sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Project Detection Setup**
   ```
   🔍 Configure project detection:
   GitHub username (optional): your-username
   Collection types: project,docs
   Global collections: references
   ✅ Configuration saved
   ```

4. **Claude Desktop Integration**
   ```
   🤖 Configure Claude Desktop integration?
   [y/N]: y
   
   ✅ Added to Claude Desktop config:
   ~/.config/claude-desktop/claude_desktop_config.json
   ```

5. **Test Sample Operations**
   ```
   🧪 Testing sample operations...
   ✅ Created test collection
   ✅ Stored test document
   ✅ Search functionality working
   ✅ Cleanup completed
   ```

## Step 4: Verify Installation

Test your setup with diagnostic tools:

```bash
# Run comprehensive diagnostics
workspace-qdrant-test
```

**Expected output**:
```
🔍 Running Workspace-Qdrant-MCP Diagnostics...

✅ Qdrant Connection
   - Server: http://localhost:6333
   - Version: 1.7.0
   - Status: Healthy

✅ Embedding System
   - Model: sentence-transformers/all-MiniLM-L6-v2
   - Dimensions: 384
   - Load time: 2.1s

✅ Collection Operations
   - Create: Working
   - Store: Working
   - Search: Working
   - Delete: Working

✅ MCP Integration
   - Tools registered: 5
   - Ready for Claude

🎉 All systems operational!
```

## Step 5: Verify Claude Integration

### For Claude Desktop

1. Restart Claude Desktop
2. Start a new conversation
3. Type: "Can you list my available collections?"

**Expected response**: Claude should list available MCP tools including qdrant-find and qdrant-store.

### For Claude Code

```bash
# Verify MCP integration
claude mcp list

# Should include workspace-qdrant-mcp
```

## Troubleshooting

### Common Issues

**Issue**: `Connection refused to localhost:6333`
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it:
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Issue**: `Permission denied` installing globally
```bash
# Use user installation instead
pip install --user workspace-qdrant-mcp

# Or use uv with user flag
uv tool install --user workspace-qdrant-mcp
```

**Issue**: `Module not found` when running commands
```bash
# Check PATH includes installation directory
which workspace-qdrant-mcp

# Add to PATH if needed (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

## Configuration Files Created

The setup wizard creates these configuration files:

```
~/.config/claude-desktop/claude_desktop_config.json  # Claude Desktop MCP config
~/.workspace-qdrant-mcp/config.json                  # Application config
~/.workspace-qdrant-mcp/projects.json                # Project detection cache
```

## Next Steps

🎉 **Congratulations!** You've successfully installed and configured workspace-qdrant-mcp.

**What's next:**
- [First Steps with Collections](02-first-collections.md) - Learn how collections work
- [Basic Search Operations](03-basic-search.md) - Perform your first searches
- [Verification and Testing](04-verification.md) - Thoroughly test your setup

## Environment Variables Reference

For future reference, here are the key environment variables you can customize:

```bash
# Qdrant configuration
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your-api-key"  # For cloud only

# Collection configuration  
export COLLECTIONS="project,docs,tests"
export GLOBAL_COLLECTIONS="references,standards"

# Project filtering
export GITHUB_USER="your-username"

# Embedding model
export FASTEMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

---

**Need help?** Check the [Troubleshooting Guide](../troubleshooting/01-common-issues.md) or [open an issue](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues).