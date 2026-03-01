# wqm CLI Reference

`wqm` is the Workspace Qdrant MCP command-line interface. It provides control over the daemon service, project and library management, queue inspection, code graph queries, and system administration.

## Global Options

These options apply to every command.

| Option | Default | Description |
|--------|---------|-------------|
| `--format <FORMAT>` | `table` | Output format: `table`, `json`, or `plain` |
| `-v, --verbose` | off | Enable verbose output |
| `--daemon-addr <ADDR>` | `http://127.0.0.1:50051` | Daemon gRPC address. Also reads from `WQM_DAEMON_ADDR` environment variable |

## Command Groups

---

## Search & Content

### `wqm search`

Query indexed content using semantic or keyword search. Search commands guide you to the appropriate MCP tool call since embedding generation runs through the MCP server.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `project <query>` | Search within the current project |
| `collection <name> <query>` | Search a named collection |
| `global <query>` | Search across all projects |
| `rules <query>` | Search behavioral rules |

**Global flag:** `-n, --limit <N>` (default: `10`)

**`wqm search project`**

| Flag | Description |
|------|-------------|
| `--include-libs` | Include library content in results |
| `-f, --file-type <TYPE>` | Filter by file type: `code`, `doc`, `test`, `config` |
| `-b, --branch <BRANCH>` | Filter by branch name |

**`wqm search collection`**

| Flag | Description |
|------|-------------|
| `-f, --filter <JSON>` | Metadata filter in JSON format |

**`wqm search global`**

| Flag | Description |
|------|-------------|
| `--exclude <PROJECT>` | Exclude a project (repeatable) |

**`wqm search rules`**

| Flag | Description |
|------|-------------|
| `-s, --scope <SCOPE>` | Filter by scope: `global`, `project`, `language` |

**Examples**

```sh
# Search current project for error handling patterns
wqm search project "error handling retry logic"

# Search with file type filter
wqm search project "database connection" --file-type code

# Search across all projects
wqm search global "authentication middleware" -n 20

# Search rules for testing preferences
wqm search rules "unit test"
```

---

### `wqm ingest`

Enqueue documents for ingestion into the vector store.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `file <path>` | Ingest a single file |
| `folder <path>` | Ingest a folder recursively |
| `text <content>` | Ingest raw text content |
| `url <url>` | Fetch and ingest a web page |
| `status` | Show ingestion queue status |

**`wqm ingest file`**

| Flag | Description |
|------|-------------|
| `-c, --collection <NAME>` | Target collection (auto-detected if omitted) |
| `-t, --tag <TAG>` | Library tag for library ingestion |

**`wqm ingest folder`**

| Flag | Description |
|------|-------------|
| `-c, --collection <NAME>` | Target collection (auto-detected if omitted) |
| `-t, --tag <TAG>` | Library tag for library ingestion |
| `-p, --patterns <GLOB>` | File patterns to include (repeatable) |
| `-n, --limit <N>` | Maximum number of files to process |

**`wqm ingest text`**

| Flag | Description |
|------|-------------|
| `-c, --collection <NAME>` | Target collection (required) |
| `-t, --title <TITLE>` | Document title or identifier |

**`wqm ingest url`**

| Flag | Description |
|------|-------------|
| `-c, --collection <NAME>` | Target collection |
| `-l, --library <NAME>` | Library name (stores in libraries collection) |
| `-t, --title <TITLE>` | Document title (auto-extracted from HTML if omitted) |

**`wqm ingest status`**

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Show detailed queue items |

**Examples**

```sh
# Ingest a single document into a library
wqm ingest file /path/to/doc.pdf --tag rust-docs

# Ingest a web page into a library
wqm ingest url https://doc.rust-lang.org/book/ --library rust-book

# Ingest raw text
wqm ingest text "API documentation content" --collection projects --title "API Notes"

# Check ingestion queue
wqm ingest status --verbose
```

---

### `wqm rules`

Manage behavioral rules stored in the `rules` collection. Rules are loaded by the MCP server at session start and persist across sessions.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List behavioral rules |
| `add` | Add a new rule |
| `remove` | Remove a rule |
| `search <query>` | Search rules by content |
| `scope` | Manage rule scopes |

**`wqm rules list`**

| Flag | Description |
|------|-------------|
| `--global` | Show only global rules |
| `--project <PATH_OR_ID>` | Show only rules for a specific project |
| `-t, --rule-type <TYPE>` | Filter by type: `preference`, `behavior`, `constraint`, `pattern` |
| `-v, --verbose` | Show full rule content |
| `-f, --format <FORMAT>` | Output format: `table` (default) or `json` |

**`wqm rules add`**

| Flag | Description |
|------|-------------|
| `--label <LABEL>` | Rule label/identifier (required) |
| `--content <CONTENT>` | Rule content (required) |
| `--global` | Apply to all projects |
| `--project <PATH_OR_ID>` | Apply to a specific project |
| `-t, --rule-type <TYPE>` | Rule type (default: `preference`) |
| `-p, --priority <N>` | Priority 1-10, higher = more important (default: `5`) |

**`wqm rules remove`**

| Flag | Description |
|------|-------------|
| `--label <LABEL>` | Rule label to remove (required) |
| `--global` | Remove from global scope |
| `--project <PATH_OR_ID>` | Remove from a specific project |

**`wqm rules search`**

| Flag | Description |
|------|-------------|
| `--global` | Search only global rules |
| `--project <PATH_OR_ID>` | Search only rules for a specific project |
| `-n, --limit <N>` | Maximum results (default: `10`) |

**`wqm rules scope`**

| Flag | Description |
|------|-------------|
| `--list` | List all available scopes |
| `--show <SCOPE>` | Show rules for a specific scope |
| `-v, --verbose` | Show verbose scope information |

**Examples**

```sh
# List all global rules
wqm rules list --global

# Add a project-specific rule
wqm rules add --label "use-tokio" --content "Prefer tokio async runtime" --project /path/to/project

# Add a global preference
wqm rules add --label "prefer-rust" --content "Prefer Rust for performance-critical code" --global

# Search rules
wqm rules search "testing"

# Remove a rule
wqm rules remove --label "use-tokio" --global
```

---

### `wqm scratch`

Manage scratchpad entries — short-lived notes stored in the `scratchpad` collection.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `add <content>` | Add a new scratchpad entry |
| `list` | List scratchpad entries |

---

## Project & Library

### `wqm project`

Manage project registration and lifecycle within the daemon.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List all registered projects |
| `status [path]` | Show project status |
| `register [path]` | Register a project for tracking |
| `info [project]` | Show detailed project information |
| `delete [project]` | Delete a project and its data |
| `priority [project] <level>` | Set project priority |
| `activate [project]` | Activate a project |
| `deactivate [project]` | Deactivate a project |
| `check [project]` | Check ingestion status vs filesystem |
| `branch <action>` | Branch management |

All `[project]` arguments accept a project ID or path. When omitted, the current working directory is used.

**`wqm project list`**

| Flag | Description |
|------|-------------|
| `-a, --active` | Show only active projects |
| `-p, --priority <LEVEL>` | Filter by priority: `high`, `normal`, `low` |

**`wqm project register`**

| Flag | Description |
|------|-------------|
| `-n, --name <NAME>` | Human-readable project name |
| `-y, --yes` | Skip confirmation prompt |

**`wqm project delete`**

| Flag | Description |
|------|-------------|
| `-y, --yes` | Skip confirmation prompt |
| `--keep-data` | Remove from SQLite only; preserve Qdrant vectors |

**`wqm project priority`**

| Value | Description |
|-------|-------------|
| `high` | High priority — processed first in queue |
| `normal` | Normal priority |

**`wqm project check`**

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Show per-file status |
| `--json` | Output as JSON |

**`wqm project branch`**

| Action | Description |
|--------|-------------|
| `list` | List branches for current project |
| `info` | Show current branch info |
| `switch <branch>` | Switch active branch for indexing |

**Examples**

```sh
# Register current directory as a project
wqm project register

# Register a specific path with a name
wqm project register /path/to/project --name "My Project"

# List all active high-priority projects
wqm project list --active --priority high

# Check what files are missing from the index
wqm project check --verbose

# Set project to high priority
wqm project priority high

# Switch to a feature branch
wqm project branch switch feature/new-api
```

---

### `wqm library`

Manage reference libraries — external documentation, PDFs, and other content ingested into the `libraries` collection.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List all libraries |
| `add <tag> <path>` | Add a library (metadata only, no watching) |
| `watch <tag> <path>` | Watch a library path for changes |
| `unwatch <tag>` | Stop watching a library |
| `remove <tag>` | Remove a library and all its vectors |
| `rescan <tag>` | Rescan and re-ingest a library |
| `info [tag]` | Show library information |
| `status` | Show watch status for all libraries |
| `ingest <file>` | Ingest a single document into a library |
| `config <tag>` | Configure library settings |
| `set-incremental <files...>` | Set or clear the incremental flag on tracked files |

**`wqm library list`**

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Show detailed information |

**`wqm library add`**

| Flag | Description |
|------|-------------|
| `-m, --mode <MODE>` | Sync mode: `sync` or `incremental` (default: `incremental`) |

**`wqm library watch`**

| Flag | Description |
|------|-------------|
| `-p, --patterns <GLOB>` | File patterns to include (repeatable) |
| `-m, --mode <MODE>` | Sync mode: `sync` or `incremental` (default: `incremental`) |

**`wqm library remove`**

| Flag | Description |
|------|-------------|
| `-y, --yes` | Skip confirmation prompt |

**`wqm library rescan`**

| Flag | Description |
|------|-------------|
| `-f, --force` | Force re-ingestion of all files |

**`wqm library ingest`**

| Flag | Description |
|------|-------------|
| `-l, --library <TAG>` | Library tag to ingest into (required) |
| `--chunk-tokens <N>` | Target tokens per chunk (default: `105`) |
| `--overlap-tokens <N>` | Overlap tokens between chunks (default: `12`) |

**`wqm library config`**

| Flag | Description |
|------|-------------|
| `--mode <MODE>` | Set sync mode: `sync` or `incremental` |
| `--patterns <GLOBS>` | Set file patterns (comma-separated) |
| `--enable` | Enable watching |
| `--disable` | Disable watching |
| `--show` | Show current configuration |

**`wqm library set-incremental`**

| Flag | Description |
|------|-------------|
| `--clear` | Clear the incremental flag (allow deletions) |

**Sync modes**

| Mode | Behavior |
|------|----------|
| `incremental` | Append-only; files removed from disk are not deleted from Qdrant |
| `sync` | Delete vectors for files removed from disk |

**Examples**

```sh
# Watch a directory of PDFs as a library
wqm library watch rust-docs /path/to/rust/docs --patterns "*.pdf,*.md"

# Ingest a single document
wqm library ingest /path/to/reference.pdf --library rust-docs

# Show library details
wqm library info rust-docs

# Rescan after adding new files
wqm library rescan rust-docs

# Remove a library and all its vectors
wqm library remove rust-docs --yes
```

---

### `wqm watch`

Inspect and manage watch folder configurations directly. Watch folders drive the daemon's file event processing.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List all watch configurations |
| `enable <watch-id>` | Enable a watch configuration |
| `disable <watch-id>` | Disable a watch configuration |
| `show <watch-id>` | Show details for a specific watch |
| `archive <watch-id>` | Archive a watch (data remains searchable; watching stops) |
| `unarchive <watch-id>` | Unarchive a watch (resumes watching) |
| `pause` | Pause all enabled watchers |
| `resume` | Resume all paused watchers |

**`wqm watch list`**

| Flag | Description |
|------|-------------|
| `--enabled` | Show only enabled watches |
| `--disabled` | Show only disabled watches |
| `-c, --collection <NAME>` | Filter by collection name |
| `--json` | Output as JSON |
| `-v, --verbose` | Show more columns |
| `--show-archived` | Include archived watch folders |

**`wqm watch show`**

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |

**Examples**

```sh
# List all active watches
wqm watch list --enabled

# Pause all watchers during a large refactor
wqm watch pause

# Resume watching after refactor is complete
wqm watch resume

# Archive a watch folder to keep data but stop watching
wqm watch archive abc123
```

---

### `wqm tags`

Inspect keyword and tag extraction results stored in SQLite.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list --doc <ID>` | List tags for a specific document |
| `keywords --doc <ID>` | List keywords for a specific document |
| `tree --tenant <ID>` | Show canonical tag hierarchy |
| `stats` | Show extraction statistics |
| `search <query>` | Search tags by name |
| `baskets --doc <ID>` | Show keyword baskets for a document |

**`wqm tags list`**

| Flag | Description |
|------|-------------|
| `--doc <ID>` | Document ID (required) |
| `--tag-type <TYPE>` | Filter by type: `concept` or `structural` |
| `--json` | Output as JSON |

**`wqm tags keywords`**

| Flag | Description |
|------|-------------|
| `--doc <ID>` | Document ID (required) |
| `--json` | Output as JSON |

**`wqm tags tree`**

| Flag | Description |
|------|-------------|
| `--tenant <ID>` | Tenant ID (required) |
| `--collection <NAME>` | Collection (default: `projects`) |

**`wqm tags stats`**

| Flag | Description |
|------|-------------|
| `--tenant <ID>` | Tenant ID (optional; all tenants if omitted) |
| `--collection <NAME>` | Collection (default: `projects`) |

**`wqm tags search`**

| Flag | Description |
|------|-------------|
| `--collection <NAME>` | Collection (default: `projects`) |
| `--json` | Output as JSON |

**`wqm tags baskets`**

| Flag | Description |
|------|-------------|
| `--doc <ID>` | Document ID (required) |
| `--json` | Output as JSON |

---

## Queue & Analytics

### `wqm queue`

Inspect and manage the unified processing queue backed by SQLite.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List queue items |
| `show <queue-id>` | Show details for a specific item |
| `stats` | Show queue statistics |
| `retry [queue-id]` | Retry failed items |
| `clean` | Remove old completed or failed items |
| `remove <queue-id>` | Remove a specific item |

**`wqm queue list`**

| Flag | Description |
|------|-------------|
| `-s, --status <STATUS>` | Filter by status: `pending`, `in_progress`, `done`, `failed` |
| `-c, --collection <NAME>` | Filter by collection name |
| `-t, --item-type <TYPE>` | Filter by item type: `file`, `folder`, `content`, `project`, `library` |
| `-n, --limit <N>` | Maximum items to show (default: `50`) |
| `--offset <N>` | Skip first N items (default: `0`) |
| `-o, --order-by <FIELD>` | Sort field: `created_at`, `priority`, `status` (default: `created_at`) |
| `-d, --desc` | Descending order |
| `--json` | Output as JSON |
| `-v, --verbose` | Show more columns |

**`wqm queue show`**

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |

The `<queue-id>` argument accepts a full queue ID or an idempotency key prefix.

**`wqm queue stats`**

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |
| `-t, --by-type` | Show breakdown by item type |
| `-o, --by-op` | Show breakdown by operation |
| `-c, --by-collection` | Show breakdown by collection |

**`wqm queue retry`**

| Flag | Description |
|------|-------------|
| `--all` | Retry all failed items |

**`wqm queue clean`**

| Flag | Description |
|------|-------------|
| `--days <N>` | Remove items older than N days (default: `7`) |
| `--status <STATUS>` | Only clean items with this status: `done` or `failed` |
| `-y, --yes` | Skip confirmation prompt |

**Examples**

```sh
# Show all pending items
wqm queue list --status pending

# Show failed items with breakdown
wqm queue stats --by-type --by-op

# Retry all failed items
wqm queue retry --all

# Clean up completed items older than 3 days
wqm queue clean --days 3 --status done --yes

# Inspect a specific item
wqm queue show abc123
```

---

### `wqm stats`

Search instrumentation analytics derived from `search_events` and related tables.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `overview` | Search instrumentation overview |
| `processing` | Processing timing stats with per-phase breakdown |
| `log-search` | Log a search event (used by wrapper scripts) |

**`wqm stats overview` and `wqm stats processing`**

| Flag | Description |
|------|-------------|
| `-p, --period <PERIOD>` | Time period: `day`, `week` (default), `month`, `all` |

**`wqm stats processing`** (additional flags)

| Flag | Description |
|------|-------------|
| `--op <OP>` | Filter by operation: `add`, `update`, `delete`, `scan` |
| `--item-type <TYPE>` | Filter by item type: `file`, `text`, `folder`, `tenant` |
| `--json` | Output as JSON |

**`wqm stats log-search`**

| Flag | Description |
|------|-------------|
| `--tool <NAME>` | Tool name (e.g., `rg`, `grep`) (required) |
| `--query <TEXT>` | Search query text (required) |
| `--actor <ACTOR>` | Actor: `claude` (default) or `user` |
| `--session-id <ID>` | Session ID (optional) |

---

## Service & Admin

### `wqm service`

Manage the `memexd` daemon process and system service registration.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `start` | Start the daemon |
| `stop` | Stop the daemon |
| `restart` | Restart the daemon |
| `status` | Show daemon status |
| `install` | Install the daemon as a system service |
| `uninstall` | Uninstall the daemon system service |
| `logs` | View daemon logs |

**`wqm service status`**

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |

**`wqm service install`**

| Flag | Description |
|------|-------------|
| `--binary <PATH>` | Path to `memexd` binary (auto-detected if omitted) |

**`wqm service uninstall`**

| Flag | Description |
|------|-------------|
| `--remove-data` | Also remove `~/.workspace-qdrant/` data directory |

**`wqm service logs`**

| Flag | Description |
|------|-------------|
| `-n, --lines <N>` | Number of lines to show (default: `50`) |
| `-f, --follow` | Follow log output (like `tail -f`) |
| `-e, --errors-only` | Show only ERROR and WARN level entries |

**Examples**

```sh
# Install and start the daemon
wqm service install
wqm service start

# Check daemon health
wqm service status

# Follow logs in real time
wqm service logs --follow

# Show only errors from the last 100 lines
wqm service logs -n 100 --errors-only

# Restart after configuration change
wqm service restart
```

---

### `wqm status`

Consolidated system status monitoring.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `queue` | Queue status summary |
| `watch` | Watch folder status |
| `performance` | Performance metrics |
| `health` | System health check |

---

### `wqm admin`

Administrative operations that affect persistent state.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `rename-tenant <old-id> <new-id>` | Rename a tenant ID across all SQLite tables |
| `idle-history` | Show idle/active state transition history |
| `prune-logs` | Remove old log files |
| `cleanup-orphans` | Detect and optionally delete orphaned tenants |
| `recover-state` | Rebuild `state.db` from Qdrant collections |

**`wqm admin rename-tenant`**

| Flag | Description |
|------|-------------|
| `-y, --yes` | Skip confirmation prompt |

**`wqm admin idle-history`**

| Flag | Description |
|------|-------------|
| `-H, --hours <N>` | Hours of history to analyze (default: `24`) |

**`wqm admin prune-logs`**

| Flag | Description |
|------|-------------|
| `--dry-run` | List files that would be deleted without deleting |
| `--retention-hours <N>` | Retention period in hours (default: `36`) |

**`wqm admin cleanup-orphans`**

Orphans are tenant IDs that exist in Qdrant but have no corresponding entry in SQLite `watch_folders` or `tracked_files`.

| Flag | Description |
|------|-------------|
| `--delete` | Actually delete orphaned points (default is dry-run report) |
| `--collection <NAME>` | Limit to a specific collection (default: all 4 canonical collections) |

**`wqm admin recover-state`**

Scrolls all Qdrant collections and reconstructs `watch_folders`, `tracked_files`, `qdrant_chunks`, and `rules_mirror`. Backs up the existing `state.db` before proceeding.

| Flag | Description |
|------|-------------|
| `--confirm` | Actually perform recovery (default is dry-run) |

---

### `wqm config`

Configuration management for the daemon and CLI.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `generate` | Generate a default configuration file |
| `default` | Show default configuration values |
| `xdg` | Show XDG base directory paths |
| `show` | Show current active configuration |
| `path` | Show configuration file path |

---

### `wqm collections`

Manage Qdrant collections.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List all collections |
| `reset` | Reset a collection (deletes all vectors) |

---

### `wqm language`

Language support tools for LSP servers and Tree-sitter grammars.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List supported languages |
| `ts-install <lang>` | Install a Tree-sitter grammar |
| `ts-remove <lang>` | Remove a Tree-sitter grammar |
| `lsp-install <lang>` | Install an LSP server |
| `lsp-remove <lang>` | Remove an LSP server |
| `status` | Show language support status |

---

### `wqm update`

Update `wqm` and `memexd` binaries from GitHub releases.

---

## Maintenance

### `wqm rebuild`

Trigger index rebuilds via the daemon. Requires the daemon to be running.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `tags` | Rebuild canonical tag hierarchy |
| `search` | Rebuild FTS5 code search index |
| `vocabulary` | Rebuild BM25 sparse vocabulary |
| `keywords` | Re-extract keywords/tags for all documents |
| `rules` | Diagnose and reconcile rules between Qdrant and SQLite |
| `projects` | Rescan all project watch folders |
| `libraries` | Rescan all library watch folders |
| `all` | Rebuild all computed indexes in sequence |

**Common flags** (where applicable)

| Flag | Description |
|------|-------------|
| `--tenant <ID>` | Limit to a specific tenant (all tenants if omitted) |
| `--collection <NAME>` | Target collection (default: `projects`) |

**Examples**

```sh
# Full rebuild after schema migration
wqm rebuild all

# Rebuild only the FTS5 search index
wqm rebuild search

# Rebuild tag hierarchy for one tenant
wqm rebuild tags --tenant abc123def456
```

---

## Data Management

### `wqm backup`

Create Qdrant collection snapshots.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `create` | Create snapshots of all collections |

---

### `wqm restore`

Restore Qdrant collections from snapshots.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `list` | List available snapshots |
| `from <path>` | Restore from a snapshot file |

---

## Code Graph

### `wqm graph`

Query the code relationship graph built from Tree-sitter analysis. All subcommands require a `--tenant` argument.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `query` | Query nodes related to a symbol within N hops |
| `impact` | Find nodes affected by changing a symbol |
| `stats` | Show node and edge counts |
| `pagerank` | Compute PageRank scores |
| `communities` | Detect code communities |
| `betweenness` | Compute betweenness centrality scores |
| `migrate` | Migrate graph data between backends |

**`wqm graph query`**

| Flag | Description |
|------|-------------|
| `--node-id <ID>` | Node ID to query from (required) |
| `--tenant <ID>` | Project tenant ID (required) |
| `--hops <N>` | Maximum traversal depth 1–5 (default: `2`) |
| `--edge-types <TYPES>` | Edge type filter, comma-separated: `CALLS`, `IMPORTS`, `CONTAINS`, `USES_TYPE`, `EXTENDS`, `IMPLEMENTS` |

**`wqm graph impact`**

| Flag | Description |
|------|-------------|
| `--symbol <NAME>` | Symbol name to analyze (required) |
| `--tenant <ID>` | Project tenant ID (required) |
| `--file <PATH>` | Narrow to a specific file path |

**`wqm graph stats`**

| Flag | Description |
|------|-------------|
| `--tenant <ID>` | Project tenant ID (optional; all tenants if omitted) |

**`wqm graph pagerank`**

| Flag | Description |
|------|-------------|
| `--tenant <ID>` | Project tenant ID (required) |
| `--damping <F>` | Damping factor (default: `0.85`) |
| `--max-iterations <N>` | Maximum iterations (default: `100`) |
| `--tolerance <F>` | Convergence tolerance (default: `1e-6`) |
| `--top-k <N>` | Return only top K results |
| `--edge-types <TYPES>` | Edge type filter, comma-separated |

**`wqm graph communities`**

| Flag | Description |
|------|-------------|
| `--tenant <ID>` | Project tenant ID (required) |
| `--max-iterations <N>` | Max label propagation iterations (default: `50`) |
| `--min-size <N>` | Minimum community size to include (default: `2`) |
| `--edge-types <TYPES>` | Edge type filter, comma-separated |

**`wqm graph betweenness`**

| Flag | Description |
|------|-------------|
| `--tenant <ID>` | Project tenant ID (required) |
| `--top-k <N>` | Return only top K results |
| `--max-samples <N>` | Sample N source nodes for large graphs (`0` = all) |
| `--edge-types <TYPES>` | Edge type filter, comma-separated |

**`wqm graph migrate`**

| Flag | Description |
|------|-------------|
| `--from <BACKEND>` | Source backend: `sqlite` or `ladybug` (default: `sqlite`) |
| `--to <BACKEND>` | Target backend: `sqlite` or `ladybug` (default: `ladybug`) |
| `--tenant <ID>` | Migrate a specific tenant (all tenants if omitted) |
| `--batch-size <N>` | Import batch size (default: `500`) |

**Examples**

```sh
# Show graph statistics for all tenants
wqm graph stats

# Find all callers of a function within 2 hops
wqm graph query --node-id "src/auth.rs::validate_token" --tenant abc123 --hops 2

# Identify what breaks if a function changes
wqm graph impact --symbol "validate_token" --tenant abc123

# Find the most central nodes (top 20 by PageRank)
wqm graph pagerank --tenant abc123 --top-k 20

# Detect code communities
wqm graph communities --tenant abc123 --min-size 3

# Find structural bridge nodes
wqm graph betweenness --tenant abc123 --top-k 10
```

---

## Setup & Diagnostics

### `wqm init`

Install shell completions.

**Usage**

```sh
wqm init bash   # Bash completions
wqm init zsh    # Zsh completions
wqm init fish   # Fish completions
```

---

### `wqm man`

Generate and install man pages.

---

### `wqm hooks`

Manage Claude Code hooks for real-time ingestion during active coding sessions.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `install` | Install Claude Code hooks |
| `uninstall` | Remove Claude Code hooks |
| `status` | Show hook installation status |

---

### `wqm debug`

Diagnostic tools for troubleshooting.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `logs` | Show daemon logs |
| `errors` | Show recent errors |
| `queue-errors` | Show queue processing errors |
| `language` | Show language support diagnostics |

---

## Benchmarking

### `wqm benchmark`

Internal benchmarking tools for performance testing.

**Subcommands**

| Subcommand | Description |
|------------|-------------|
| `sparse-vectors` | Benchmark sparse vector generation |
| `search-engines` | Benchmark search engine performance |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `WQM_DAEMON_ADDR` | Override daemon gRPC address (default: `http://127.0.0.1:50051`) |
| `WQM_DATABASE_PATH` | Override SQLite state database path |
| `WQM_LOG_LEVEL` | Log level: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `QDRANT_URL` | Qdrant server URL (default: `http://localhost:6333`) |
| `QDRANT_API_KEY` | Qdrant API key (required for Qdrant Cloud) |
| `ORT_LIB_LOCATION` | Path to ONNX Runtime static library |
