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
