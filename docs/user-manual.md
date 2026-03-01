# User Manual

## Chapter 1: Installation

### Prerequisites

- **Qdrant** vector database (local Docker or Qdrant Cloud)
- **macOS** (arm64 or x86_64) or **Linux** (x86_64 or arm64)

### Install Qdrant

**Local (Docker):**

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Qdrant Cloud:** Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io), then set the URL and API key:

```bash
export QDRANT_URL="https://your-cluster.qdrant.io:6333"
export QDRANT_API_KEY="your-api-key"
```

### Install workspace-qdrant

**Pre-built binaries (recommended):**

```bash
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash
```

This installs `wqm` (CLI) and `memexd` (daemon) to `~/.local/bin`. Add to your PATH if not already there:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

**Build from source:**

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp
./install.sh
```

Requires Rust 1.75+ and ONNX Runtime. See [Installation Reference](reference/installation.md) for platform-specific details.

### Start the daemon

```bash
wqm service install    # Install as system service (one-time)
wqm service start      # Start the daemon
```

Verify everything is running:

```bash
wqm --version          # CLI version
wqm admin health       # Daemon and Qdrant connectivity
```

---

## Chapter 2: Configuration

### Config file

All components share one config file at `~/.workspace-qdrant/config.yaml`. You only need to create this file to override defaults — workspace-qdrant works out of the box with local Qdrant.

**Qdrant Cloud example:**

```yaml
qdrant:
  url: https://your-cluster.qdrant.io:6333
  api_key: your-api-key
```

**Custom database path:**

```yaml
database:
  path: /custom/path/state.db
```

### Environment variables

Environment variables override config file values:

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | — | API key for Qdrant Cloud |
| `WQM_DATABASE_PATH` | `~/.workspace-qdrant/state.db` | SQLite database location |
| `WQM_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARN, ERROR) |
| `WQM_DAEMON_ADDR` | `http://127.0.0.1:50051` | Daemon gRPC address |

### Log locations

| OS | Path |
|----|------|
| macOS | `~/Library/Logs/workspace-qdrant/` |
| Linux | `~/.local/state/workspace-qdrant/logs/` |

View logs:

```bash
wqm service logs           # Recent daemon logs
wqm service logs --follow  # Follow live
```

### Configuration reference

For the complete list of all configuration options, see [Configuration Reference](reference/configuration.md).

---

## Chapter 3: Project Management

### Registering a project

Register a directory so the daemon watches it for changes and indexes all files:

```bash
wqm project register /path/to/your/project
```

The daemon automatically:
1. Detects the Git repository (if any)
2. Scans all source files matching the allowlist
3. Generates embeddings and indexes them in Qdrant
4. Watches for file changes going forward

For the current directory:

```bash
cd /path/to/project
wqm project register .
```

### Checking ingestion progress

After registration, files are queued for processing. Monitor progress:

```bash
wqm queue stats            # Pending/processing/done counts
wqm project check          # Compare tracked vs filesystem files
```

### Listing projects

```bash
wqm project list           # All registered projects
wqm project info           # Detailed info for current directory's project
wqm project status         # Status summary
```

### Project lifecycle

```bash
wqm project activate       # Mark project as active (higher processing priority)
wqm project deactivate     # Mark as inactive (still watched, lower priority)
wqm project delete         # Remove project and all indexed data
```

Active projects receive:
- Higher queue processing priority
- LSP code intelligence enrichment (for supported languages)
- More frequent file change processing

### Watch folder management

Each registered project or library has a watch configuration:

```bash
wqm watch list             # All watch configurations
wqm watch show <watch-id>  # Details for a specific watch
wqm watch disable <id>     # Stop watching (keeps data)
wqm watch enable <id>      # Resume watching
wqm watch archive <id>     # Archive (stop watching, data stays searchable)
wqm watch pause            # Pause ALL watchers temporarily
wqm watch resume           # Resume all paused watchers
```

### Library ingestion

Libraries are reference documentation — books, papers, API docs, manuals. They're stored in the `libraries` collection, separate from project code.

**Ingest a single document:**

```bash
wqm library ingest /path/to/document.pdf --library my-docs
```

Supported formats: PDF, DOCX, PPTX, ODT, ODS, RTF, EPUB, HTML, Markdown, plain text.

**Watch a library folder:**

```bash
wqm library watch /path/to/docs-folder --library reference-docs
```

New or changed documents in the folder are automatically re-ingested.

**List libraries:**

```bash
wqm library list           # All libraries with document counts
wqm library info my-docs   # Details for a specific library
```

### Ingesting other content

```bash
wqm ingest file path/to/file.py --library snippets   # Single file to library
wqm ingest folder path/to/docs/ --library manuals     # Folder to library
wqm ingest url https://example.com/docs --library web  # Web page
wqm ingest text "Some content" --library notes         # Raw text
```

---

## Chapter 4: Searching

### Semantic search

Search using natural language — finds conceptually related results even without exact keyword matches:

```bash
wqm search project "authentication middleware"
wqm search project "how does error handling work"
```

### Search scopes

```bash
wqm search project "query"      # Current project only
wqm search collection "query" --collection libraries  # Specific collection
wqm search global "query"       # All projects
wqm search rules "query"        # Behavioral rules
```

### Filters

Narrow results with filters:

```bash
# By file type
wqm search project "database" --file-type rust

# By branch
wqm search project "feature" --branch develop

# By limit
wqm search project "query" --limit 5
```

### Output formats

```bash
wqm search project "query"                    # Table (default)
wqm search project "query" --format json       # JSON for scripting
wqm search project "query" --format plain      # Plain text
```

### MCP search tools

When using workspace-qdrant through an AI assistant (Claude Desktop, Claude Code), searches are available as MCP tools:

**search** — hybrid semantic + keyword search:
```
search(query="authentication", scope="project")
search(query="JWT implementation", collection="libraries")
search(query="error handling", tags=["security"])
```

**grep** — exact substring or regex search (faster for known strings):
```
grep(pattern="handleRequest")
grep(pattern="fn process_.*event", regex=true)
grep(pattern="TODO", pathGlob="**/*.rs")
```

**list** — browse project file structure:
```
list(format="summary")              # Overview of project layout
list(path="src/", format="tree")    # Tree view of src/ directory
list(extension="rs", depth=5)       # All Rust files
```

### When to use which tool

| Goal | Tool | Example |
|------|------|---------|
| Find code by concept | `search` | "authentication flow" |
| Find exact string | `grep` | "handleRequest" |
| Find by regex pattern | `grep` | "fn.*Error" |
| Browse file structure | `list` | See what files exist |
| Get specific document | `retrieve` | Known document ID |
| Find library docs | `search` with `collection="libraries"` | "React hooks" |

### Tags and keywords

Documents are automatically tagged with extracted concepts. Use tags to filter results:

```bash
wqm tags search "authentication"   # Find tags matching a term
```

In MCP:
```
search(query="error handling", tag="security")
search(query="database", tags=["orm", "migration"])
```

### Search quality tips

1. **Be specific** — "JWT token validation middleware" finds better results than "auth"
2. **Use technical terms** — include function names, class names, or domain terms
3. **Start narrow, then widen** — search `scope="project"` first, then `scope="all"`
4. **Use grep for known code** — if you know the function name, `grep` is faster and more precise
5. **Combine search + list** — use `list` to understand project layout, then `search` within relevant areas

---

## Chapter 5: Rules and Behavioral Memory

Rules are persistent preferences that carry across sessions. They're stored in the `rules` collection and automatically injected into AI assistant context.

### Creating rules

**Via CLI:**

```bash
wqm rules add --label "prefer-uv" --content "Always use uv instead of pip for Python package management"
wqm rules add --label "use-pytest" --content "Use pytest for all Python tests, not unittest" --scope project
```

**Via MCP:**
```
rules(action="add", label="prefer-uv", content="Always use uv instead of pip")
rules(action="add", label="test-style", content="Use pytest with fixtures", scope="project")
```

### Rule scoping

| Scope | Behavior |
|-------|----------|
| `global` | Applies to all projects (default) |
| `project` | Applies only to the current project |

Project-scoped rules require a `projectId` (auto-detected from the current directory in MCP).

### Managing rules

```bash
wqm rules list                          # List all rules
wqm rules list --scope project          # Project rules only
wqm rules remove --label "prefer-uv"   # Remove a rule
```

**Via MCP:**
```
rules(action="list")                           # All rules
rules(action="list", scope="project")          # Project rules
rules(action="update", label="prefer-uv", content="Updated content")
rules(action="remove", label="prefer-uv")
```

### Best practices for rules

- **Labels:** Use kebab-case, max 15 characters: `prefer-uv`, `use-pytest`, `no-any-types`
- **Content:** Write in imperative voice — "Use X" not "X should be used"
- **Specificity:** Be precise — "Use ruff for Python linting and formatting" not "Use linting"
- **Scope:** Use project scope for project-specific conventions, global for personal preferences

### Rule examples

| Label | Content | Scope |
|-------|---------|-------|
| `prefer-uv` | Always use uv instead of pip for Python packages | global |
| `use-pytest` | Use pytest with fixtures for all tests | global |
| `no-any-types` | Never use `any` type in TypeScript — use proper types or `unknown` | project |
| `commit-style` | Use conventional commits: feat/fix/docs/refactor prefix | global |
| `error-handling` | Return Result types, never panic in library code | project |

---

## Chapter 6: CLI Reference

This chapter covers the most commonly used `wqm` commands. For the complete reference with all flags, see [CLI Reference](reference/cli.md).

### Service management

```bash
wqm service start       # Start daemon
wqm service stop        # Stop daemon
wqm service restart     # Restart daemon
wqm service status      # Check if running
wqm service install     # Install as system service
wqm service uninstall   # Remove system service
wqm service logs        # View daemon logs
```

### Search

```bash
wqm search project "query"                    # Search current project
wqm search project "query" --limit 5          # Limit results
wqm search project "query" --format json      # JSON output
wqm search collection "query" -c libraries    # Search libraries
wqm search global "query"                     # All projects
wqm search rules "query"                      # Search rules
```

### Project management

```bash
wqm project list                    # List all projects
wqm project register /path          # Register new project
wqm project info                    # Info for current dir
wqm project status                  # Status summary
wqm project activate                # Set active (auto-detects CWD)
wqm project deactivate              # Set inactive
wqm project delete                  # Remove project + data
wqm project check                   # Verify ingestion completeness
wqm project branch list             # List branches
```

### Library management

```bash
wqm library list                          # List all libraries
wqm library add my-docs                   # Create library (metadata only)
wqm library watch /path --library my-docs # Watch a folder
wqm library ingest file.pdf --library docs # Ingest single document
wqm library info my-docs                  # Library details
wqm library remove my-docs               # Remove library + data
wqm library rescan my-docs               # Re-ingest all documents
```

### Queue and monitoring

```bash
wqm queue stats          # Pending/processing/done counts
wqm queue list           # List queue items
wqm queue show <id>      # Item details
wqm queue retry          # Retry failed items
wqm queue clean          # Remove old completed items
wqm status health        # System health overview
```

### Tags and keywords

```bash
wqm tags search "auth"               # Search tags
wqm tags list --tenant <id>          # Tags for a project
wqm tags tree --tenant <id>          # Tag hierarchy
wqm tags stats                       # Extraction statistics
```

### Code graph

```bash
wqm graph stats --tenant <id>                          # Node/edge counts
wqm graph query --node-id <id> --tenant <id> --hops 2  # Related nodes
wqm graph impact --symbol "MyClass" --tenant <id>      # Impact analysis
wqm graph pagerank --tenant <id> --top-k 20            # Most important nodes
wqm graph communities --tenant <id>                    # Code communities
wqm graph betweenness --tenant <id> --top-k 20         # Bridge nodes
```

### Administrative

```bash
wqm admin health                     # Health diagnostics
wqm admin cleanup-orphans            # Find orphaned tenants
wqm admin recover-state              # Rebuild state.db from Qdrant
wqm collections list                 # List Qdrant collections
wqm rebuild all                      # Rebuild all indexes
```

---

## Chapter 7: MCP Server Setup

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "node",
      "args": ["/path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

For Qdrant Cloud, add `"QDRANT_API_KEY": "your-key"` to the `env` block.

### Claude Code

```bash
claude mcp add workspace-qdrant -- node /path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js
```

### CLAUDE.md integration

Add the following to your project's `CLAUDE.md` (or global `~/.claude/CLAUDE.md`) so Claude uses workspace-qdrant proactively:

````markdown
## workspace-qdrant

The `workspace-qdrant` MCP server provides codebase-aware search, library retrieval, and persistent behavioral rules.

### Project Registration

At session start, check whether the current project is registered with workspace-qdrant. If not registered, ask the user before registering.

### Codebase Intelligence

Use `search` as the primary tool for understanding project code:
- Search for symbols, functions, and patterns across the codebase
- Use `scope="project"` for current project, `scope="all"` for broader search
- Use `grep` for exact string matching and `list` for browsing file structure

### Behavioral Rules

Use `rules(action="list")` at session start to load active rules. Only add rules when the user explicitly asks.

### Issue Reporting

Report workspace-qdrant issues at https://github.com/ChrisGVE/workspace-qdrant-mcp/issues
````

### Verifying MCP tools

After configuration, verify the tools are available in your AI assistant:

1. Start a new conversation
2. Ask: "What workspace-qdrant tools do you have?"
3. You should see 6 tools: search, retrieve, rules, store, grep, list

### Troubleshooting MCP connections

| Issue | Solution |
|-------|----------|
| Tools not appearing | Restart Claude Desktop/Code, check config path |
| "Connection refused" | Verify daemon is running: `wqm service status` |
| "Qdrant unreachable" | Verify Qdrant is running: `curl http://localhost:6333/collections` |
| Timeout errors | Check `QDRANT_URL` in env block matches your setup |
| Permission errors | Ensure `node` binary is in PATH |

---

## Chapter 8: Troubleshooting

### Common issues

**Daemon won't start:**

```bash
wqm service status       # Check current state
wqm debug logs           # View error logs
wqm admin health         # Diagnose connectivity
```

Common causes:
- Port 50051 already in use (another daemon instance)
- Qdrant not reachable (Docker not running, wrong URL)
- Missing ONNX Runtime library

**Files not being indexed:**

```bash
wqm project check        # Compare tracked vs filesystem
wqm queue stats          # Check if queue is processing
wqm queue list --status failed  # Check for failed items
```

Common causes:
- File type not in allowlist (only source code and documents are indexed)
- File in excluded directory (node_modules, .git, target, etc.)
- File exceeds size limit
- Queue processor is paused: `wqm watch resume`

**Search returns no results:**

```bash
wqm project list         # Verify project is registered
wqm queue stats          # Verify ingestion completed (pending = 0)
wqm collections list     # Verify collections exist
```

Common causes:
- Project not registered or still being ingested
- Searching wrong collection (projects vs libraries)
- Query too specific — try broader terms

### Health checks

```bash
wqm admin health         # Full health diagnostic
```

This checks:
- Daemon connectivity (gRPC port 50051)
- Qdrant connectivity and collection status
- SQLite database accessibility
- Queue processor state

### Log locations

| OS | Path |
|----|------|
| macOS | `~/Library/Logs/workspace-qdrant/` |
| Linux | `~/.local/state/workspace-qdrant/logs/` |

```bash
wqm service logs                  # Recent logs
wqm debug logs --errors-only      # Errors only
```

### Queue inspection

When things seem stuck:

```bash
wqm queue stats                   # Overview
wqm queue list --status failed    # Failed items
wqm queue show <queue-id>         # Details of specific item
wqm queue retry                   # Retry all failed items
wqm queue clean --days 7          # Clean old completed items
```

### Rebuilding indexes

If search results seem stale or incomplete:

```bash
wqm rebuild all                   # Rebuild everything
wqm rebuild search                # Rebuild FTS5 search index only
wqm rebuild tags                  # Rebuild tag hierarchy only
wqm rebuild vocabulary            # Rebuild BM25 vocabulary
```

### Database recovery

If the SQLite database is corrupted:

```bash
wqm admin recover-state           # Rebuild state.db from Qdrant
```

This reconstructs the database from Qdrant collection data. Watch folders need to be re-registered.

### Getting help

- [GitHub Issues](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues) — bug reports and feature requests
- [Quick Start Guide](quick-start.md) — getting started
- [CLI Reference](reference/cli.md) — complete command documentation
- [MCP Tools Reference](reference/mcp-tools.md) — MCP tool parameters
