# LLM Integration Guide: workspace-qdrant-mcp Best Practices

This guide covers how to configure, instruct, and get the most from the workspace-qdrant-mcp MCP server when integrating with Claude Desktop, Claude Code, or any MCP-compatible client.

## Table of Contents

- [MCP Server Setup](#mcp-server-setup)
- [Instructing LLMs to Use Tools Effectively](#instructing-llms-to-use-tools-effectively)
- [Search Strategies](#search-strategies)
- [Rules as Behavioral Memory](#rules-as-behavioral-memory)
- [Workflow Patterns](#workflow-patterns)
- [Performance Tips](#performance-tips)

---

## MCP Server Setup

### Prerequisites

Before connecting Claude to the MCP server, ensure the daemon and Qdrant are running:

```bash
# Start Qdrant (if not already running)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Start the daemon
wqm service install   # One-time setup
wqm service start

# Verify health
wqm admin health      # Should print: healthy
```

### Claude Desktop

Add to your `claude_desktop_config.json` (located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

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

For Qdrant Cloud, add the API key:

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "node",
      "args": ["/path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js"],
      "env": {
        "QDRANT_URL": "https://your-cluster.qdrant.io",
        "QDRANT_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Claude Code

Use the `claude mcp add` command to register the server:

```bash
claude mcp add workspace-qdrant -- node /path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js
```

To include environment variables:

```bash
claude mcp add workspace-qdrant \
  -e QDRANT_URL=http://localhost:6333 \
  -- node /path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js
```

Verify registration:

```bash
claude mcp list
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | No | (none) | API key for Qdrant Cloud |
| `FASTEMBED_MODEL` | No | `all-MiniLM-L6-v2` | Embedding model (384-dim) |
| `WQM_DATABASE_PATH` | No | `~/.workspace-qdrant/state.db` | Override SQLite state path |
| `WQM_LOG_LEVEL` | No | `INFO` | Log level: DEBUG, INFO, WARN, ERROR |

---

## Instructing LLMs to Use Tools Effectively

### Recommended CLAUDE.md Snippet

Place this in your project's `CLAUDE.md` or in your global `~/.claude/CLAUDE.md` to instruct Claude to use workspace-qdrant-mcp proactively:

```markdown
## workspace-qdrant-mcp (Codebase Intelligence)

First tool when executing searches in the codebase or knowledge-base
(files, symbols, literal strings, regex, relationships, similarities).
You can fine tune the search scope, extend or narrow it.

### Session start protocol

At the start of every session:
1. Call `rules(action="list")` to load all behavioral rules for this project
2. Verify project registration — if search returns empty or status "degraded",
   run `store(type="project", path="<cwd>")` to register the project
3. Begin work using search, grep, and list tools for codebase exploration

### Tool selection

- `search` — semantic and keyword search across indexed code, docs, and notes.
  Use for concept-level queries: "how does authentication work", "error handling pattern"
- `grep` — exact substring or regex match. Use when you know the literal string:
  function names, import paths, specific constants, error messages
- `list` — browse project structure. Use to orient yourself before diving in,
  or to find files by type/language
- `retrieve` — direct access to a known document ID or metadata filter.
  Use after `search` to paginate through large documents chunk-by-chunk
- `store` — persist reference documentation, notes, or web pages to the
  libraries collection. Also used to register new projects
- `rules` — read and write behavioral rules that persist across sessions

### Search scope guidance

- Prefer `scope="project"` (default) for current-project work
- Use `scope="all"` only when looking for patterns across multiple projects
- Use `collection="libraries"` when looking for reference documentation
- Use `include_libraries=true` to search project code and reference docs together

### Rules as memory

When the user states a preference, convention, or instruction that should
persist ("always use X", "never do Y", "for this project, prefer Z"):
- Call `rules(action="add", ...)` to save it immediately
- Choose `scope="project"` for project-specific preferences
- Choose `scope="global"` for universal preferences
```

### Explanation of Each Section

**Session start protocol**: The MCP server does not automatically inject rules into context. The LLM must call `rules(action="list")` explicitly at session start to load behavioral preferences. Checking project registration ensures file watching is active and search results are current.

**Tool selection**: Each tool has a distinct purpose. Using the wrong tool (semantic search for exact strings, or grep for concept queries) produces poor results. The section establishes clear decision criteria.

**Search scope guidance**: By default, search is scoped to the current project and current branch. Explicit scope guidance prevents accidental cross-project leakage and keeps queries fast.

**Rules as memory**: Rules stored via the `rules` tool persist in the vector database and survive session restarts. This section trains the LLM to treat conversational instructions as persistent state, not ephemeral context.

---

## Search Strategies

### Choosing the Right Tool

| Situation | Tool | Reason |
|-----------|------|--------|
| "How does X work?" | `search` | Concept-level, semantic match |
| "Where is function `parseToken` defined?" | `grep` | Exact function name |
| "Find all files importing `auth`" | `grep` with regex | Exact pattern match |
| "What error handling patterns exist?" | `search` | Semantic similarity |
| "What files are in `src/daemon/`?" | `list` | Browse structure |
| "Show me the interface for type X" | `grep` or `search` | Depends on specificity |
| "Get chunks 2-5 of document abc123" | `retrieve` | Known ID, pagination |

#### When to use `search`

Use `search` for questions and concepts where you do not know the exact text. The hybrid search engine combines dense vector similarity (semantic) with BM25 keyword scoring for the best of both approaches.

```
# Good search queries
"JWT token validation and expiry checking"
"database connection pooling and retry logic"
"error types returned by the file watcher"
```

#### When to use `grep`

Use `grep` when you know the exact string or pattern. It queries the FTS5 trigram index — much faster than semantic search for known strings, and it returns line numbers with context.

```
# Good grep patterns
pattern="validate_token"           # exact function name
pattern="impl.*FileWatcher"        # regex for trait implementations
pattern="QDRANT_URL"               # exact environment variable
pattern="tokio::spawn"             # exact usage pattern
```

#### When to use `list`

Use `list` to understand project structure before reading code. Start with `format="summary"` to get a high-level overview, then drill into specific paths with `format="tree"`.

```
# Good list usage
list(format="summary")                           # Project overview first
list(path="src/rust/daemon/core", format="tree") # Then drill in
list(language="rust", fileType="code")           # Find all Rust source files
```

### Query Formulation Tips

**Be specific with technical terms.** The embedding model was trained on code and technical text. Using the correct technical vocabulary improves result relevance significantly.

- Weak: "find the function that checks things"
- Strong: "validate JWT token signature and expiry"

**Include context in queries.** The search engine uses surrounding context, not just the query terms.

- Weak: "error handling"
- Strong: "error handling in the file watching pipeline when inotify fails"

**Use `component` to narrow scope.** For monorepo-style projects with multiple sub-systems, the `component` filter restricts search to a specific component tree.

```typescript
search({
  query: "queue processing and retry logic",
  component: "daemon.core"
})
```

**Use `pathGlob` for file-type targeting.** When you know the answer is in a specific file type, glob filtering is faster than post-filtering results.

```typescript
search({
  query: "configuration schema and defaults",
  pathGlob: "**/*.toml"
})
```

### Filter Reference

#### `scope` parameter

| Value | Behavior |
|-------|----------|
| `"project"` (default) | Current project, current branch |
| `"all"` | All projects in the index |
| `"global"` | For `rules` collection: global rules only |

#### `collection` parameter

| Value | Contains |
|-------|----------|
| `"projects"` (default) | All indexed project files (code, docs, tests, configs) |
| `"libraries"` | Reference documentation stored via `store` |
| `"rules"` | Behavioral rules stored via `rules` tool |
| `"scratchpad"` | Temporary notes stored via `store(type="scratchpad")` |

#### `mode` parameter (search only)

| Value | Behavior |
|-------|----------|
| `"hybrid"` (default) | Semantic + keyword with Reciprocal Rank Fusion |
| `"semantic"` | Pure vector similarity (dense embeddings only) |
| `"keyword"` | Keyword/exact matching (sparse BM25 only) |

#### `file_type` parameter

| Value | Files included |
|-------|----------------|
| `"code"` | Source code (.rs, .py, .ts, .go, etc.) |
| `"doc"` | Documentation (.md, .txt, .rst) |
| `"test"` | Test files (test_*, *_test.*, spec.*) |
| `"config"` | Configuration (.yaml, .json, .toml, .env) |
| `"note"` | User notes and scratch content |
| `"artifact"` | Build outputs, generated files |

### Result Limit Tuning

The default limit is 10 results. Adjust based on task:

- **Quick lookup** (finding one specific thing): `limit=3` to `limit=5`
- **Exploration** (understanding a subsystem): `limit=10` to `limit=20`
- **Exhaustive review** (audit, refactoring): `limit=20` to `limit=50`

Higher limits increase latency. Prefer filtering (scope, file_type, component) over high limits to keep results focused.

---

## Rules as Behavioral Memory

The `rules` tool stores LLM behavioral instructions that persist in the vector database across sessions. Unlike context window instructions, rules survive session restarts and are available on every future session when loaded via `rules(action="list")`.

### Label Format

Labels are the unique identifier for a rule. They follow strict formatting constraints:

- Maximum 15 characters
- Format: `word-word-word` (lowercase, hyphen-separated)
- Must be unique within a scope (global or project)

Good labels:
```
prefer-uv           # Python package manager preference
no-mock-fs          # Testing constraint
strict-types        # TypeScript strict mode
use-pytest          # Test framework preference
pre-commit-tests    # Git workflow rule
deploy-after-build  # Post-change workflow
cli-first-wqm       # Tool usage preference
```

Avoid:
```
my_preference       # Underscores not allowed
UseUVForPython      # Not lowercase hyphenated
this-label-is-way-too-long  # Over 15 chars
```

### Scope Selection

| Scope | When to use | Example |
|-------|-------------|---------|
| `"global"` | Convention applies to all projects | "always use 4-space indentation" |
| `"project"` | Convention is project-specific | "use the wqm CLI before raw SQLite" |

When a user gives an instruction without specifying scope, default to `"project"` for technical/tooling preferences and `"global"` for universal style preferences.

### Rule Content Best Practices

Rules should be written in imperative voice, be specific, and be actionable. Vague rules are ignored or misapplied.

**Weak rule content:**
```
Be careful with the database. Also use the right tools.
```

**Strong rule content:**
```
Always use wqm CLI commands in priority over raw sqlite3 or direct DB access.
This serves dual purpose: accomplishing the task AND validating that the CLI
works as intended. If the CLI succeeds: task done, CLI validated. If the CLI
fails or lacks the needed subcommand/flag: (1) work around the issue to unblock
the immediate task, (2) check specs/PRD to determine if the failed/missing
feature is specified, (3) if specified but broken: create a bug task.
```

### Example Rules

Project-scoped rules for a Rust project:

```typescript
// Deployment workflow
rules({
  action: "add",
  label: "deploy-after-build",
  title: "Deploy binaries and restart daemon after changes",
  content: "After making changes to the Rust codebase, once tests and compilation " +
    "are successful, always build release binaries and deploy them: " +
    "cargo build --release, then cp target/release/memexd ~/.local/bin/memexd, " +
    "then restart the daemon via launchctl.",
  scope: "project",
  tags: ["workflow", "deployment", "rust"]
})

// Shared type ownership
rules({
  action: "add",
  label: "use-common-crate",
  title: "Shared types must live in wqm-common crate",
  content: "When introducing or modifying shared data structures used by more than " +
    "one component, always define the canonical implementation in the wqm-common crate. " +
    "Other components must import from wqm-common rather than defining their own copies.",
  scope: "project",
  tags: ["architecture", "shared-types"]
})
```

Global rules for a Python developer:

```typescript
// Package manager preference
rules({
  action: "add",
  label: "prefer-uv",
  title: "Use uv instead of pip for Python packages",
  content: "Always use uv instead of pip for Python package management. " +
    "Use 'uv add' instead of 'pip install', 'uv run' instead of 'python -m'.",
  scope: "global",
  tags: ["python", "tooling"]
})
```

### Label Generation Strategy

When creating rules from natural language user instructions:

1. Extract the key subject and action from the instruction
2. Form a 2-3 word hyphenated identifier
3. Use action verbs as prefix when describing a tool or method preference: `use-`, `prefer-`, `avoid-`, `no-`
4. Use nouns for constraints or architectural rules: `strict-types`, `daemon-owns-db`

Examples:
```
"Always run tests before committing"      → pre-commit-tests
"Use rust-analyzer, not rls"              → prefer-rust-analyzer
"Don't modify global config files"        → no-global-config
"Store shared types in common crate"      → use-common-crate
```

---

## Workflow Patterns

### Session Start

At the start of each session, follow this sequence to ensure rules are loaded and the project is registered:

```typescript
// 1. Load behavioral rules for the current project
const rules = await rules({ action: "list" })
// Apply any rules found to your behavior

// 2. Verify project is indexed (search will return status: "degraded" if not)
const check = await search({ query: "project structure", limit: 1 })
if (check.status === "degraded") {
  // Register the project — daemon will begin watching and indexing
  await store({ type: "project", path: process.cwd() })
}

// 3. Begin work — codebase is ready to query
```

### Code Understanding: Progressive Refinement

Use a three-step pattern when exploring an unfamiliar subsystem:

**Step 1 — Orient with `list`**

```typescript
// Get high-level project structure
list({ format: "summary" })
// Then drill into the relevant subsystem
list({ path: "src/rust/daemon/core", format: "tree", depth: 2 })
```

**Step 2 — Discover with `search`**

```typescript
// Understand the concept
search({ query: "queue processing and retry logic", component: "daemon" })
search({ query: "file watcher event debouncing" })
```

**Step 3 — Pinpoint with `grep`**

```typescript
// Find the exact implementation
grep({ pattern: "UnifiedQueueProcessor", contextLines: 3 })
grep({ pattern: "impl.*Debounce", regex: true, pathGlob: "**/*.rs" })
```

This sequence moves from broad structure to semantic concepts to exact code locations, progressively narrowing scope.

### Library Documentation: Store Then Search

When bringing in reference documentation for a library or framework:

```typescript
// Store a documentation page
await store({
  type: "url",
  url: "https://docs.rs/tokio/latest/tokio/",
  libraryName: "tokio"
})

// Or store a local file
await store({
  type: "library",
  content: documentationText,
  title: "Tokio Runtime Guide",
  libraryName: "tokio"
})

// Later — search the documentation
search({
  query: "spawning tasks and cancellation",
  collection: "libraries",
  scope: "tokio"
})

// Or search project code and library docs together
search({
  query: "async task spawning pattern",
  include_libraries: true
})
```

### Refactoring Workflow

When refactoring code, use the search tools to understand impact before changing anything:

```typescript
// 1. Find all usages of the symbol being changed
grep({ pattern: "validate_token", contextLines: 2 })

// 2. Understand callers and dependents semantically
search({ query: "validate_token authentication flow", includeGraphContext: true })

// 3. Check for related tests
search({ query: "token validation test cases", file_type: "test" })

// 4. Find configuration that may reference the symbol
grep({ pattern: "validate_token", pathGlob: "**/*.toml" })
```

### Storing Session Notes

Use the scratchpad for analysis or intermediate notes during a session:

```typescript
// Save analysis for later reference
await store({
  type: "scratchpad",
  content: "Analysis of queue processor bottleneck: The fairness scheduler " +
    "alternates between high-priority batches of 10 and low-priority batches of 3. " +
    "The bottleneck is at embedding generation, not queue dequeue.",
  tags: ["performance", "queue", "analysis"]
})

// Retrieve later
search({ query: "queue processor bottleneck analysis", collection: "scratchpad" })
```

---

## Performance Tips

### Use `scope="project"` by Default

The default scope is the current project, which is also the fastest. Cross-project search (`scope="all"`) scans the entire `projects` collection with only tenant filtering. Narrow scope means fewer Qdrant points to score.

```typescript
// Faster — searches only current project
search({ query: "error handling", scope: "project" })

// Slower — scans all indexed projects
search({ query: "error handling", scope: "all" })
```

### Control Result Size with `limit`

Each additional result adds scoring and serialization cost. Use the smallest limit that satisfies the task.

```typescript
// Finding one specific thing — limit 3 is enough
grep({ pattern: "TokenValidator", maxResults: 3 })

// Exploring a concept — 10 is the right default
search({ query: "authentication middleware" })

// Only increase limit for exhaustive tasks
search({ query: "all error types", limit: 30 })
```

### Use `grep` for Known Strings

`grep` queries the FTS5 trigram index built on raw file content. It is deterministic and significantly faster than semantic search for known strings. When you know the exact text you are looking for, `grep` is always the right choice.

```typescript
// These are faster as grep, not search:
grep({ pattern: "QDRANT_URL" })            // env var
grep({ pattern: "fn process_file" })       // function name
grep({ pattern: "use crate::storage" })    // import path
grep({ pattern: "error: queue full" })     // error message
```

### Use `list(format="summary")` Before `format="tree"`

The `summary` format returns aggregated statistics (file counts by language, component breakdown, directory structure) without listing individual files. Use it first to understand what you are dealing with before requesting the full tree.

```typescript
// First call — cheap, gives overview
list({ format: "summary" })

// Second call — drill into the relevant area only
list({ path: "src/rust/daemon/core/src/keyword_extraction", format: "tree" })
```

### Use `pathGlob` and `component` Filters

Both filters happen at query time inside Qdrant, before scoring. They reduce the candidate set and improve both speed and relevance.

```typescript
// Without filter — scores all project files
search({ query: "configuration defaults" })

// With filter — only config files, much smaller candidate set
search({ query: "configuration defaults", pathGlob: "**/*.toml" })

// With component — only the daemon subsystem
search({ query: "queue retry logic", component: "daemon.core" })
```

### Avoid Redundant Cross-Collection Searches

The `include_libraries=true` parameter executes a second search against the `libraries` collection and fuses the results with RRF. Only use it when you genuinely need both project code and reference documentation in the same result set. Otherwise, make two separate targeted queries.

```typescript
// Use when you need both together
search({ query: "gRPC client setup", include_libraries: true })

// Prefer separate targeted queries when you know which you need
search({ query: "daemon gRPC client implementation" })
search({ query: "gRPC documentation examples", collection: "libraries" })
```
