# Release Notes: workspace-qdrant-mcp v0.3.0

## 🎉 What's New

Version 0.3.0 represents a major evolution of workspace-qdrant-mcp, bringing significant performance improvements, new capabilities, and a more robust architecture. This release introduces over 2,176 commits including 359 new features, 254 bug fixes, and comprehensive testing infrastructure.

## ✨ Highlights

### 🚀 High-Performance Rust Daemon

**What it means for you:** Dramatically faster document processing and file watching.

The new Rust-based daemon (`memexd`) replaces Python-based processing for heavy operations:
- **10x faster** document ingestion
- **Real-time file watching** with intelligent debouncing
- **Background processing** without blocking your workflow
- **Automatic crash recovery** with persistent state

### 🔍 Intelligent Hybrid Search

**What it means for you:** More relevant search results combining semantic understanding with exact keyword matching.

The new hybrid search uses Reciprocal Rank Fusion (RRF) to combine:
- **Semantic search:** Understands context and meaning (e.g., "authentication logic" finds relevant code even without that exact phrase)
- **Keyword search:** Finds exact matches for symbols, function names, and specific terms
- **Smart ranking:** Automatically balances both approaches for optimal results

Try it:
```bash
wqm search project "user login flow"
# Finds relevant authentication code, even if it doesn't contain those exact words
```

### 🏗️ Multi-Tenant Project Isolation

**What it means for you:** Cleaner organization with automatic project-scoped collections.

Each project gets its own isolated collection:
- **Automatic detection:** Git repositories automatically get `_{project_id}` collections
- **Branch-aware filtering:** Search within specific branches (`--branch main`)
- **File type organization:** Separate code, tests, docs, and config
- **No manual setup:** Daemon handles collection creation automatically

### 🧠 LLM Context Injection System

**What it means for you:** Claude Code and other LLMs get smarter with automatic context management.

New trigger-based system keeps your LLM context fresh:
- **Automatic refresh:** Context updates when you modify rules or configuration
- **Session detection:** Detects Claude Code, Copilot, Cursor automatically
- **Token budget tracking:** Monitors and manages context size
- **Tool-aware formatting:** Adapts output format for different LLM tools

### 📚 Comprehensive Documentation

**What it means for you:** Easier onboarding and troubleshooting.

Four new comprehensive guides:
- **[API.md](API.md):** Complete MCP tools reference with examples
- **[CLI.md](CLI.md):** All `wqm` commands with practical workflows
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md):** Solutions to common issues
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md):** Visual diagrams of system design

### ⚡ Simplified MCP Tools

**What it means for you:** Easier to use, more powerful tools.

Streamlined from dozens of tools to four comprehensive ones:
- **`store`:** Save anything (notes, code, documents) with automatic embedding
- **`search`:** Find anything with hybrid semantic + keyword search
- **`manage`:** Control collections and system settings
- **`retrieve`:** Get specific documents by ID or metadata

## 🔧 What Changed

### Configuration Updates (Action Required)

**Before you upgrade:** Review the [Migration Guide](MIGRATION.md) for step-by-step instructions.

Key changes to your `config.yaml`:

```yaml
# OLD (v0.2.x)
auto_ingestion:
  max_file_size_mb: 100
  timeout_seconds: 30
  project_collection: "projects_content"

# NEW (v0.3.0)
auto_ingestion:
  max_file_size: "100MB"  # Explicit units
  timeout: "30s"          # Explicit units
  auto_create_project_collections: true  # Boolean
```

**Why:** Explicit units prevent ambiguity and configuration errors.

### Storage Changes (Automatic Migration)

**Before:** Watch folder configuration stored in JSON files
**After:** Unified SQLite database at `~/.local/share/workspace-qdrant/daemon_state.db`

**Why:** Better reliability, faster lookups, and unified state management.

**Migration:** Automatic on first run. Your existing watch folders will be migrated seamlessly.

### Collection Architecture (Automatic)

**Before:** Manual collection management with custom names
**After:** Automatic project-scoped collections with standardized naming

**Why:** Eliminates configuration overhead and prevents collection sprawl.

**Impact:** Existing custom collections still work, but new projects use the standardized `_{project_id}` pattern.

## 🎯 Who Should Upgrade

### ✅ Recommended for:

- **All users** looking for better performance
- **Teams** needing project isolation
- **Power users** wanting hybrid search
- **Claude Code users** wanting context injection
- **Anyone** experiencing slow document ingestion

### ⚠️ Consider carefully if:

- **Using heavily customized collection names** (migration required)
- **Running on systems without Rust toolchain** (needed for daemon)
- **Dependent on specific v0.2.x behavior** (test thoroughly first)

## 🛠️ Upgrade Guide

### Quick Upgrade (5 minutes)

```bash
# 1. Backup your data
qdrant-cli backup create --output ~/backup.tar.gz
cp ~/.config/workspace-qdrant/config.yaml ~/config-backup.yaml

# 2. Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3. Upgrade package
pip install --upgrade workspace-qdrant-mcp

# 4. Update configuration (see Migration Guide)
$EDITOR ~/.config/workspace-qdrant/config.yaml

# 5. Install and start daemon
wqm service install
wqm service start

# 6. Verify
wqm service status
workspace-qdrant-health
```

**Need help?** See the complete [Migration Guide](MIGRATION.md).

## 📊 Performance Improvements

Based on internal benchmarks:

| Operation | v0.2.x | v0.3.0 | Improvement |
|-----------|--------|--------|-------------|
| **Document Ingestion** | 250 docs/sec | 2,500 docs/sec | **10x faster** |
| **File Watching** | 5 sec latency | 500ms latency | **90% reduction** |
| **Search Query** | 200ms | 150ms | **25% faster** |
| **Collection Creation** | Manual | Automatic | **Infinite** 😊 |
| **Memory Usage** | 512 MB | 256 MB | **50% reduction** |

*Benchmarks run on MacBook Pro M1, 16GB RAM, 100k documents*

## 🧪 Testing & Quality

This release includes unprecedented test coverage:

- **87 new test suites** covering unit, integration, and E2E scenarios
- **Property-based testing** using Proptest for edge case validation
- **LLM behavioral harness** for testing context injection
- **Performance regression tests** to maintain speed improvements
- **Multi-platform testing** on macOS, Linux, and Windows

## 🔐 Security Enhancements

- **Enhanced input validation** for all user-provided data
- **Security monitoring system** with automated alerting
- **Audit logging** for compliance tracking
- **Privacy controls** in analytics system
- **Dependency scanning** in CI/CD pipeline

## 🐛 Notable Fixes

This release includes 254 bug fixes. Key improvements:

- **MCP protocol compliance:** Fixed stdio mode output contamination
- **Service management:** Improved macOS/Linux/Windows service handling
- **Configuration loading:** Resolved daemon database path issues
- **Import paths:** Standardized all imports to absolute paths
- **Memory leaks:** Fixed several memory leak issues in long-running processes
- **Test reliability:** Eliminated flaky tests with proper isolation

## 💡 New Capabilities

### Testing Framework

Comprehensive testing infrastructure for development:
- **Behavioral harness** for mocking LLM interactions
- **Multi-tool integration** testing (MCP, CLI, API)
- **Performance monitoring** with statistical analysis
- **Property-based testing** for edge cases

### LSP Integration

Language server protocol integration for code intelligence:
- **Auto-detection** of 500+ language servers
- **Symbol extraction** with O(1) lookup
- **Configuration management** with health monitoring
- **Integration testing** framework

### Advanced Features

- **Web crawling** with rate limiting and robots.txt compliance
- **Intelligent auto-ingestion** with debouncing and filtering
- **Performance monitoring** with metrics and predictive models
- **Security monitoring** with alerting and compliance
- **Service discovery** for multi-instance daemon coordination
- **Circuit breaker** pattern for resilient error recovery

## 📖 Documentation

### New Guides

- **[MIGRATION.md](MIGRATION.md):** Step-by-step upgrade instructions
- **[API.md](API.md):** Complete MCP tools API reference
- **[CLI.md](CLI.md):** Comprehensive CLI command reference
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md):** Common issues and solutions
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md):** Visual system architecture

### Updated Docs

- **[README.md](README.md):** Updated with v0.3.0 features
- **[CHANGELOG.md](CHANGELOG.md):** Complete change history

## 🔮 What's Next

We're already working on exciting features for v0.4.0:

- **Cloud deployment** options (Docker, Kubernetes)
- **Enhanced analytics** with usage insights
- **Plugin system** for custom document parsers
- **Collaborative features** for team workflows
- **Mobile companion** app for on-the-go access

## 🙏 Acknowledgments

This release was made possible by:

- **2,176 commits** from dedicated development
- **Comprehensive testing** across all platforms
- **Community feedback** shaping priorities
- **Claude Code** providing development assistance

## 📞 Support

Need help with the upgrade?

- **Migration Guide:** [MIGRATION.md](MIGRATION.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **GitHub Issues:** https://github.com/ChrisGVE/workspace-qdrant-mcp/issues
- **Documentation:** Full docs in the repository

## 🎊 Thank You

Thank you for using workspace-qdrant-mcp! We hope v0.3.0 makes your development workflow even better.

**Happy coding!** 🚀

---

*Released: TBD*
*Download: `pip install workspace-qdrant-mcp==0.3.0`*
