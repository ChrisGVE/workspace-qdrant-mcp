# GitHub Issues to Create

This document lists the GitHub issues that need to be created for tracking current problems and improvements.

## Critical Issues

### 1. Daemon hardcoded log paths prevent user-level service execution
**Priority:** High
**Labels:** bug, daemon, service-management

**Description:**
The Rust memexd daemon has hardcoded log paths that point to system-level directories (`/var/log/memexd.log`), causing permission errors when running as a user service.

**Current Behavior:**
When the daemon starts as a user service (via `wqm service install`), it immediately fails with:
```
Error: FileSystem { message: "Failed to open log file: Permission denied (os error 13)", path: "/var/log/memexd.log", operation: "open_file", source: None }
```

**Expected Behavior:**
The daemon should:
1. Respect log paths specified in plist configuration (`StandardOutPath` and `StandardErrorPath`)
2. Default to user-writable directories when running as user service
3. Support configuration-based log path overrides

**Proposed Solution:**
1. Modify Rust daemon to accept log paths via:
   - Command line arguments (`--log-file`, `--error-log-file`)
   - Environment variables (`MEMEXD_LOG_FILE`, `MEMEXD_ERROR_LOG_FILE`)
   - Configuration file settings

2. Default to user directories when no system paths are writable:
   - `~/Library/Logs/memexd.log` (macOS)
   - `~/.local/share/logs/memexd.log` (Linux)

**Files Affected:**
- `core/src/bin/memexd.rs`
- Logging initialization code
- Service configuration generation

---

### 2. UV tool install not picking up source code changes
**Priority:** Medium
**Labels:** bug, build-system, development

**Description:**
When using `uv tool install .` to install the workspace-qdrant-mcp package, changes to source code are not being picked up, requiring manual workarounds.

**Current Behavior:**
- Source code changes made to service.py and other files
- `uv tool uninstall` + `uv tool install .` doesn't pick up changes
- Old code persists in installed version

**Workaround:**
- Use `uv run python -m workspace_qdrant_mcp.cli.main` to run from source
- Create wrapper scripts in ~/.local/bin

**Expected Behavior:**
- `uv tool install . --force` should install current source code
- Package should be built fresh from current working directory

---

### 3. Auto-ingestion not processing workspace files
**Priority:** High  
**Labels:** bug, auto-ingestion, core-functionality

**Description:**
Despite service installation success, the auto-ingestion system is not detecting and processing files from the workspace directories (workspace-qdrant-mcp and workspace-qdrant-web-ui projects).

**Expected Behavior:**
- Daemon should watch configured directories
- Python and Rust files should be automatically ingested
- Vector embeddings should be created and stored

**Investigation Needed:**
- Check if daemon is actually running and monitoring directories
- Verify auto-ingestion configuration is correct
- Test file detection and processing pipeline

---

### 4. Service installation XML formatting issues
**Priority:** Low (Fixed)
**Labels:** bug, service-management, fixed

**Description:**
The generated plist files had XML formatting issues causing launchctl load failures.

**Status:** ✅ RESOLVED
- Fixed in commit 6388140
- ProgramArguments array elements now properly formatted with newlines
- Service loads successfully via launchctl

---

## Enhancement Requests

### 5. Add configuration management with restart notifications
**Priority:** Medium
**Labels:** enhancement, configuration

**Description:**
Users need a way to modify daemon configuration and be notified when restart is required.

**Requirements:**
- `wqm config set key=value` command
- `wqm config get [key]` command  
- `wqm config list` command
- Automatic restart notification (no hot reload)
- Validate configuration before applying

---

### 6. Improve error handling and user feedback
**Priority:** Medium
**Labels:** enhancement, user-experience

**Description:**
Better error messages and user guidance when things go wrong.

**Features Needed:**
- Descriptive error messages instead of raw Python tracebacks
- Suggestions for common problems
- Health check command to diagnose issues
- Better logging levels and output

---

### 7. Multi-language LSP support for auto-ingestion
**Priority:** Medium
**Labels:** enhancement, auto-ingestion

**Description:**
Support different LSPs for different file types to improve code understanding and embedding quality.

**Requirements:**
- Python LSP integration for .py files
- Rust Analyzer integration for .rs files  
- Configurable LSP per file type
- Fallback to simple text processing when LSP unavailable

---

## Documentation Issues

### 8. Create comprehensive installation guide
**Priority:** Medium
**Labels:** documentation

**Description:**
Users need clear installation and setup instructions.

**Content Needed:**
- Prerequisites and dependencies
- Step-by-step installation
- Configuration examples
- Troubleshooting common issues
- Platform-specific notes (macOS, Linux, Windows)

---

### 9. API documentation for MCP integration
**Priority:** Low
**Labels:** documentation, api

**Description:**
Document the MCP server API and available tools/resources.

**Content Needed:**
- Available MCP tools and their parameters
- Resource types and schemas
- Integration examples with different MCP clients
- Authentication and configuration

---

## Process Improvements

### 10. Automated issue management workflow
**Priority:** Low
**Labels:** process, automation

**Description:**
Set up GitHub Actions to automatically manage issues based on commits and PR activity.

**Features:**
- Auto-close issues when referenced in commit messages
- Auto-label issues based on files changed
- Milestone management
- Stale issue cleanup

---

## Instructions for Creating Issues

1. Copy each issue section above
2. Create new GitHub issue with the title
3. Use the description content as the issue body
4. Apply the specified labels
5. Set appropriate milestone if available
6. Assign to appropriate team member if known

## Automation

Once GitHub MCP integration is working, these issues can be created automatically using:
```bash
# Example command (when MCP is working)
mcp__github__create_issue --title="Daemon hardcoded log paths" --body="..." --labels="bug,daemon"
```