## Code Intelligence

### Architecture: Tree-sitter Baseline + LSP Enhancement

| Component       | When                     | What it provides                                 |
| --------------- | ------------------------ | ------------------------------------------------ |
| **Tree-sitter** | Always, during ingestion | Symbol definitions, language, syntax structure   |
| **LSP**         | Active projects only     | References (where used), types, resolved imports |

**Rationale:** Tree-sitter is fast and always available. LSP provides richer data but requires spawning language servers, so it's reserved for active projects.

### Tree-sitter (Baseline)

Runs on every code file during ingestion:

- **Symbol definitions:** Function, class, method, struct names and locations
- **Language detection:** From grammar matching
- **Syntax structure:** Imports, exports, declarations
- **Semantic chunking:** Split files into meaningful units (see below)

**Dynamic grammar support (semantic chunking):**

Grammars are downloaded automatically on first use (`auto_download: true` by default) and cached in `~/.workspace-qdrant/grammars/`. Pre-download with `wqm language ts-install <lang>`. The daemon checks for grammar updates when the queue is idle (configurable interval, default: weekly).

| Language   | Grammar       | Chunk Types                                      |
| ---------- | ------------- | ------------------------------------------------ |
| C          | Auto-download | function, struct, preamble                       |
| C++        | Auto-download | function, class, method, struct, preamble        |
| Go         | Auto-download | function, struct, method, preamble               |
| Java       | Auto-download | class, method, interface, preamble               |
| JavaScript | Auto-download | function, class, method, preamble                |
| JSX        | Auto-download | function, class, method, preamble                |
| Python     | Auto-download | function, class, method, preamble                |
| Rust       | Auto-download | function, struct, impl, trait, method, preamble  |
| TSX        | Auto-download | function, class, method, interface, preamble     |
| TypeScript | Auto-download | function, class, method, interface, preamble     |

**All other languages** fall back to text-based overlap chunking (384 chars target, 58 chars overlap).

**Grammars:** 10 languages have known grammars available for auto-download (see table above). Optional static compilation is available via `--features static-grammars` for environments without internet access.

### LSP (Enhancement for Active Projects)

Runs when project is active:

```
1. Project activated (RegisterProject received)
2. Daemon spawns language server(s) for detected languages
3. LSP queries enrich existing Qdrant entries:
   - Symbol references (where used)
   - Type information
   - Resolved imports
4. Server kept alive until:
   - Project deactivated, AND
   - All queued items for project processed
```

**One server per language per project.** Multi-target projects (e.g., Cargo workspace with multiple crates) are handled by single language server.

**Default LSP server configurations:**

| Language       | Server Binary                  | Notes                        |
| -------------- | ------------------------------ | ---------------------------- |
| Python         | `ruff` (ruff-lsp)             | Primary; fallback to pylsp/pyright |
| Rust           | `rust-analyzer`               |                              |
| TypeScript/JS  | `typescript-language-server`   | Handles both TS and JS       |
| Go             | `gopls`                       |                              |
| C/C++          | `clangd`                      | Handles both C and C++       |

**LSP Language enum** (18 total): Python, Rust, TypeScript, JavaScript, JSON, Go, Java, C, C++, Ruby, PHP, Shell, YAML, TOML, XML, HTML, CSS, SQL, Other(custom).

Languages without a configured default server can still be used if the user manually installs and configures the LSP server via `wqm lsp install`.

### LSP Server Lifecycle

The daemon manages LSP server instances through a state machine:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LSP Server Lifecycle                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   RegisterProject          DeprioritizeProject                       │
│        │                          │                                  │
│        ▼                          ▼                                  │
│  ┌───────────┐            ┌───────────────┐                         │
│  │  Stopped  │──spawn──→  │  Initializing │                         │
│  └───────────┘            └───────┬───────┘                         │
│        ▲                          │                                  │
│        │                          ▼ initialized                      │
│        │                   ┌───────────┐                            │
│        │◄──stop────────────│  Running  │◄──────┐                    │
│        │                   └───────┬───┘       │                    │
│        │                          │            │                    │
│        │                          ▼ unhealthy  │ healthy            │
│        │                   ┌───────────┐       │                    │
│        │◄──max retries─────│  Failed   │───────┘                    │
│        │                   └───────────┘  restart                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Lifecycle states:**
- **Stopped**: No server process running
- **Initializing**: Server spawned, waiting for LSP initialize handshake
- **Running**: Server healthy, accepting queries
- **Failed**: Server crashed or unhealthy

**Deferred shutdown:**
When a project is deprioritized, the daemon checks the queue before stopping LSP servers:
1. Check `unified_queue` for pending items with `tenant_id = project_id`
2. If queue has items, defer shutdown (configurable delay, default 60s)
3. Re-check queue after delay
4. Only stop server when queue is empty

This prevents stopping LSP servers while enrichment queries are still pending.

**State persistence (Task 1.18):**
Server states are persisted to SQLite for recovery after daemon restart:
- Stored: `project_id`, `language`, `project_root`, `restart_count`, `last_started_at`
- Cleaned up: States older than 24 hours
- On initialization: Restore states and re-spawn servers for active projects

**Language server management:**

```bash
wqm lsp install python    # Installs pyright or pylsp
wqm lsp install rust      # Installs rust-analyzer
wqm lsp list              # Shows available/installed servers
wqm lsp remove <lang>     # Removes language server
wqm lsp status            # Show running LSP servers and metrics
```

**PATH configuration:** CLI manages `environment.user_path` in the configuration file.

**Update triggers:**

- CLI installation
- Every CLI invocation
- MCP server startup (which invokes CLI)

**Processing steps:**

1. **Expansion:** Retrieve `$PATH` and expand all environment variables recursively (e.g., `~` → `/Users/chris`, `$XDG_CONFIG_HOME` → `$HOME/.config` → `/Users/chris/.config`)

2. **Merge:** Append the existing `user_path` from config to the expanded `$PATH`, split by OS path separator (`:` on Unix, `;` on Windows), preserving order

3. **Deduplicate:** Remove duplicate path segments, keeping the **first occurrence** only (earlier entries take precedence)

4. **Save:** Recombine segments into a string and write to config **only if different** from the current value (avoids unnecessary disk writes)

**Note:** Only the CLI writes to the configuration file. Daemon reads `user_path` on startup and uses it to locate language server binaries.

### Semantic Code Chunking

Instead of arbitrary text chunks, code files are split into semantic units:

#### Chunk Types

| Chunk Type         | Contains                                       | Example                   |
| ------------------ | ---------------------------------------------- | ------------------------- |
| `preamble`         | Imports, module docstring, constants           | File header               |
| `function`         | Complete function with docstring               | `def validate_token(...)` |
| `class`            | Class signature, docstrings, class-level attrs | `class AuthService:`      |
| `method`           | Method body (linked to parent class)           | `def login(self, ...)`    |
| `struct`           | Struct/dataclass definition                    | `struct Config { ... }`   |
| `trait`/`protocol` | Interface definition                           | `trait Validator { ... }` |

#### Chunking Algorithm

```
For each code file:
1. Parse with Tree-sitter → AST
2. Extract preamble (imports, module-level items)
3. Walk AST, create chunk for each:
   - Function definition
   - Class/struct definition (signature only)
   - Method definition (separate chunk, linked to class)
   - Trait/protocol/interface
4. For large units (>200 lines):
   - Fall back to overlap chunking
   - Mark as is_fragment=true
```

#### Chunk Payload Schema

```json
{
  "project_id": "abc123",
  "file_path": "src/auth.rs",
  "chunk_type": "function",
  "symbol_name": "validate_token",
  "symbol_kind": "function",
  "parent_symbol": null,
  "language": "rust",
  "start_line": 42,
  "end_line": 67,
  "docstring": "Validates JWT token and returns claims.",
  "signature": "fn validate_token(token: &str) -> Result<bool>",
  "calls": ["decode_jwt", "check_expiry"],
  "is_fragment": false
}
```

**LSP enrichment adds:**

```json
{
  "references": [
    { "file": "src/api.rs", "line": 23 },
    { "file": "src/middleware.rs", "line": 56 }
  ],
  "type_info": "fn(&str) -> Result<bool>"
}
```

#### Benefits

| Benefit               | Explanation                              |
| --------------------- | ---------------------------------------- |
| Complete context      | LLM gets whole function, not fragments   |
| Better search         | Query returns complete, meaningful units |
| Symbol association    | Function name tied to its implementation |
| Relationship tracking | Method → Class, Function → Module        |

### CLI Commands

```bash
# Library Management
wqm library add <tag> <path> --mode sync           # Register library (metadata only)
wqm library add <tag> <path> --mode incremental    # Register library (append-only)
wqm library watch <tag> <path>                     # Start watching library folder
wqm library unwatch <tag>                          # Stop watching (keeps content)
wqm library remove <tag>                           # Delete library + all vectors
wqm library list                                   # List all libraries
wqm library info [tag]                             # Show library details
wqm library status                                 # Show watch status for all libraries
wqm library rescan <tag> [--force]                 # Re-ingest library content
wqm library config <tag> --mode <mode>             # Update library configuration

# Library Document Ingestion (Single File)
wqm library ingest <file> --library <tag>          # Ingest single document
wqm library ingest <file> --library <tag> \
    --chunk-tokens 105 --overlap-tokens 12         # With custom chunking

# Code Graph
wqm graph query --node-id <id> --tenant <t> --hops 2   # Traverse related nodes
wqm graph impact --symbol <name> --tenant <t>           # Impact analysis
wqm graph stats --tenant <t>                            # Node/edge counts by type
wqm graph pagerank --tenant <t> --top-k 20              # PageRank centrality
wqm graph communities --tenant <t> --min-size 3         # Community detection
wqm graph betweenness --tenant <t> --top-k 20           # Betweenness centrality
wqm graph migrate --from sqlite --to ladybug            # Backend migration

# Search Instrumentation
wqm stats overview [--period day|week|month|all]   # View search analytics
wqm stats log-search --tool=rg --query="pattern"   # Log search event

# Watch management (admin)
wqm watch list                                     # List all watches
wqm watch disable <watch_id>                       # Temporarily disable
wqm watch enable <watch_id>                        # Re-enable
```

---

