# MCP Tools Reference

The workspace-qdrant MCP server exposes six tools to AI assistants. All tools communicate with the `memexd` daemon over gRPC.

## Tool Index

| Tool | Purpose |
|------|---------|
| [`search`](#search) | Hybrid semantic + keyword search across indexed content |
| [`retrieve`](#retrieve) | Direct document lookup by ID or metadata filter |
| [`rules`](#rules) | Manage persistent behavioral rules |
| [`store`](#store) | Store content, register projects, save notes |
| [`grep`](#grep) | Exact substring or regex search using FTS5 |
| [`list`](#list) | List project files and folder structure |

---

## search

Search for documents using hybrid semantic and keyword search. Use this tool first when answering questions about the user's codebase, project architecture, or stored knowledge. Results come from the user's actual indexed content, which is more accurate than training data.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | The search query text |
| `collection` | string | No | — | Specific collection to search: `projects`, `libraries`, `rules`, `scratchpad` |
| `mode` | string | No | `hybrid` | Search mode: `hybrid`, `semantic`, or `keyword` |
| `scope` | string | No | `project` | Search scope: `project` (current project only), `global` (all projects), or `all` |
| `limit` | number | No | `10` | Maximum number of results to return |
| `projectId` | string | No | — | Specific project ID to search |
| `libraryName` | string | No | — | Library name when searching the `libraries` collection |
| `branch` | string | No | — | Filter results by branch name |
| `fileType` | string | No | — | Filter by file type |
| `includeLibraries` | boolean | No | `false` | Include library content in project search results |
| `tag` | string | No | — | Filter results by concept tag (exact match) |
| `tags` | string[] | No | — | Filter results by multiple concept tags (OR logic) |
| `pathGlob` | string | No | — | File path glob filter, e.g. `"**/*.rs"` or `"src/**/*.ts"` |
| `component` | string | No | — | Filter by project component, e.g. `"daemon"` or `"daemon.core"`. Supports prefix matching. |
| `exact` | boolean | No | `false` | Use exact substring search instead of semantic search |
| `contextLines` | number | No | `0` | Lines of context to include before/after matches when `exact` is `true` |
| `includeGraphContext` | boolean | No | `false` | Include code relationship graph context (callers/callees) for matched symbols |

### Collections

| Value | Contents |
|-------|----------|
| `projects` | Indexed source files from all registered projects |
| `libraries` | Reference documentation, PDFs, and ingested library content |
| `rules` | Behavioral rules |
| `scratchpad` | Temporary notes and scratch content |

### Examples

Search the current project for authentication-related code:

```json
{
  "query": "JWT token validation",
  "scope": "project",
  "limit": 10
}
```

Search only Rust files using a path glob:

```json
{
  "query": "error handling retry",
  "scope": "project",
  "pathGlob": "**/*.rs",
  "limit": 15
}
```

Search library documentation for a specific concept:

```json
{
  "query": "connection pooling configuration",
  "collection": "libraries",
  "libraryName": "tokio-docs",
  "limit": 5
}
```

Search across all projects in semantic mode:

```json
{
  "query": "database migration strategy",
  "scope": "global",
  "mode": "semantic",
  "limit": 20
}
```

Search with graph context to understand callers:

```json
{
  "query": "process_queue_item",
  "scope": "project",
  "exact": true,
  "includeGraphContext": true
}
```

### Response Format

Returns an array of result objects. Each object includes:

- `id` — document or chunk identifier
- `score` — relevance score (0.0–1.0, higher is better)
- `content` — matched text content
- `metadata` — document metadata including file path, language, branch, component, and concept tags
- `graphContext` — (when `includeGraphContext: true`) callers and callees of matched symbols

---

## retrieve

Retrieve documents by ID or metadata filter. Use `retrieve` when you already know the document ID. Use `search` for discovery.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `documentId` | string | No | — | Document ID to retrieve |
| `collection` | string | No | `projects` | Collection to retrieve from: `projects`, `libraries`, `rules`, `scratchpad` |
| `filter` | object | No | — | Metadata filter as key-value pairs. Values must be strings. |
| `limit` | number | No | `10` | Maximum number of results |
| `offset` | number | No | `0` | Pagination offset |
| `projectId` | string | No | — | Project ID for the `projects` collection |
| `libraryName` | string | No | — | Library name for the `libraries` collection |

At least one of `documentId` or `filter` should be provided.

### Examples

Retrieve a document by its known ID:

```json
{
  "documentId": "abc123def456",
  "collection": "projects"
}
```

Retrieve all documents from a specific file:

```json
{
  "collection": "projects",
  "filter": {
    "file_path": "src/auth/validator.rs"
  }
}
```

Paginate through library content:

```json
{
  "collection": "libraries",
  "libraryName": "rust-book",
  "limit": 20,
  "offset": 40
}
```

### Response Format

Returns an array of document objects. Each object includes:

- `id` — document identifier
- `content` — document text content
- `metadata` — associated metadata fields

---

## rules

Manage persistent behavioral rules. Rules guide how the AI assistant behaves across sessions. They are loaded at the start of each session and persist in the `rules` collection.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | — | Action to perform: `add`, `update`, `remove`, or `list` |
| `label` | string | Conditional | — | Rule label (max 15 chars, format: `word-word-word`, e.g. `prefer-uv`). Required for `add`, `update`, `remove`. |
| `content` | string | Conditional | — | Rule content. Required for `add` and `update`. |
| `scope` | string | No | `global` | Rule scope: `global` or `project` |
| `projectId` | string | No | — | Project ID for project-scoped rules |
| `title` | string | No | — | Rule title (max 50 chars) |
| `tags` | string[] | No | — | Categorization tags (max 5 tags, max 20 chars each) |
| `priority` | number | No | — | Rule priority (higher number = more important) |
| `limit` | number | No | `50` | Maximum rules to return for `list` action |

### Actions

| Action | Required Parameters | Description |
|--------|--------------------|-|
| `add` | `label`, `content` | Create a new rule |
| `update` | `label`, `content` | Update an existing rule |
| `remove` | `label` | Delete a rule |
| `list` | — | List rules, optionally filtered by scope |

### Label Format

Labels must be lowercase, hyphen-separated words, maximum 15 characters total. Examples: `prefer-uv`, `use-pytest`, `no-commits`.

### Examples

List all global rules at session start:

```json
{
  "action": "list",
  "scope": "global"
}
```

Add a global preference rule:

```json
{
  "action": "add",
  "label": "prefer-types",
  "content": "Always use explicit type annotations in TypeScript. Avoid `any`.",
  "scope": "global",
  "priority": 8
}
```

Add a project-scoped rule:

```json
{
  "action": "add",
  "label": "no-direct-db",
  "content": "Never write directly to the database. All writes must go through the queue.",
  "scope": "project",
  "projectId": "abc123def456"
}
```

Remove a rule:

```json
{
  "action": "remove",
  "label": "prefer-types",
  "scope": "global"
}
```

### Response Format

- `add` / `update` / `remove`: returns a confirmation message and the affected rule label.
- `list`: returns an array of rule objects, each with `label`, `content`, `scope`, `priority`, `title`, and `tags`.

---

## store

Store content or register a project. Use `type` to select the storage destination.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `type` | string | No | `library` | Storage type: `library`, `url`, `scratchpad`, or `project` |
| `content` | string | Conditional | — | Content to store. Required when `type` is `library`. |
| `libraryName` | string | Conditional | — | Library name. Required when `type` is `library` unless `forProject` is `true`. |
| `forProject` | boolean | No | `false` | When `true`, stores to the libraries collection scoped to the current project. `libraryName` defaults to `"project-refs"`. |
| `path` | string | Conditional | — | Project directory path. Required when `type` is `project`. |
| `name` | string | No | — | Project display name when `type` is `project`. Defaults to the directory name. |
| `title` | string | No | — | Content title for `library` type |
| `url` | string | No | — | Source URL for web content |
| `filePath` | string | No | — | Source file path |
| `tags` | string[] | No | — | Tags for `scratchpad` entries |
| `sourceType` | string | No | `user_input` | Source type: `user_input`, `web`, `file`, `scratchbook`, or `note` |
| `metadata` | object | No | — | Additional metadata as string key-value pairs |

### Storage Types

| Type | Destination | Use Case |
|------|-------------|----------|
| `library` | `libraries` collection | Store reference documentation, notes, code snippets |
| `url` | `libraries` or `projects` | Fetch and ingest a web page |
| `scratchpad` | `scratchpad` collection | Save temporary working notes |
| `project` | Daemon registration | Register a project directory for file watching and indexing |

### Examples

Store reference documentation in a library:

```json
{
  "type": "library",
  "libraryName": "project-notes",
  "title": "Architecture Decision: Queue Design",
  "content": "The unified queue uses SQLite with WAL mode for crash resistance...",
  "sourceType": "note"
}
```

Store a note scoped to the current project:

```json
{
  "type": "library",
  "forProject": true,
  "title": "API contract notes",
  "content": "The gRPC service exposes RegisterProject which enqueues to the unified queue..."
}
```

Fetch and ingest a web page:

```json
{
  "type": "url",
  "url": "https://docs.rs/tokio/latest/tokio/",
  "libraryName": "tokio-docs",
  "title": "Tokio API Reference"
}
```

Save a scratchpad note:

```json
{
  "type": "scratchpad",
  "content": "Investigating slow queue processing: suspect IDF penalty too aggressive",
  "tags": ["investigation", "queue", "performance"]
}
```

Register a project directory:

```json
{
  "type": "project",
  "path": "/Users/chris/dev/projects/my-service",
  "name": "My Service"
}
```

### Response Format

Returns a confirmation message with the stored document ID or registration status.

---

## grep

Search code with exact substring or regex pattern matching. Uses an FTS5 trigram index for fast line-level search across all indexed files. Unlike `search`, `grep` does not use embeddings and always returns exact matches.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pattern` | string | Yes | — | Search pattern (exact substring or regex) |
| `regex` | boolean | No | `false` | Treat `pattern` as a regular expression |
| `caseSensitive` | boolean | No | `true` | Case-sensitive matching |
| `pathGlob` | string | No | — | File path glob filter, e.g. `"**/*.rs"` or `"src/**/*.ts"` |
| `scope` | string | No | `project` | Search scope: `project` (current project) or `all` (all projects) |
| `contextLines` | number | No | `0` | Lines of context to include before and after each match |
| `maxResults` | number | No | `1000` | Maximum number of results to return |
| `branch` | string | No | — | Filter by branch name |
| `projectId` | string | No | — | Specific project ID to search |

### When to Use grep vs search

| Situation | Tool |
|-----------|------|
| Looking for an exact function name, string literal, or identifier | `grep` |
| Looking for code that does a particular thing, conceptually | `search` |
| Verifying a specific string exists in the codebase | `grep` |
| Finding related code by meaning or similarity | `search` |
| Tracking all uses of an API call | `grep` |

### Examples

Find all occurrences of a function call:

```json
{
  "pattern": "process_queue_item",
  "pathGlob": "**/*.rs"
}
```

Case-insensitive regex search in TypeScript files:

```json
{
  "pattern": "use(Effect|Callback|Memo)",
  "regex": true,
  "caseSensitive": false,
  "pathGlob": "**/*.tsx",
  "contextLines": 2
}
```

Find all `TODO` comments across the entire project:

```json
{
  "pattern": "TODO",
  "scope": "project",
  "maxResults": 200
}
```

### Response Format

Returns an array of match objects. Each object includes:

- `filePath` — relative path to the matched file
- `lineNumber` — line number of the match (1-indexed)
- `lineContent` — the matched line text
- `contextBefore` — lines before the match (when `contextLines` > 0)
- `contextAfter` — lines after the match (when `contextLines` > 0)

---

## list

List project files and folder structure. Only shows indexed files; gitignored paths, `node_modules`, and build artifacts are excluded automatically.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | No | project root | Subfolder relative to the project root |
| `depth` | number | No | `3` | Maximum directory depth (maximum accepted value: `10`) |
| `format` | string | No | `tree` | Output format: `tree`, `summary`, or `flat` |
| `fileType` | string | No | — | Filter by file category: `code`, `text`, `data`, `config`, `build`, or `web` |
| `language` | string | No | — | Filter by programming language, e.g. `"rust"` or `"typescript"` |
| `extension` | string | No | — | Filter by file extension, e.g. `"rs"` or `"ts"` |
| `pattern` | string | No | — | Glob pattern applied to relative paths, e.g. `"**/*.test.ts"` |
| `includeTests` | boolean | No | `true` | Include test files in results |
| `limit` | number | No | `200` | Maximum number of entries returned (maximum: `500`) |
| `projectId` | string | No | current project | Specific project ID |
| `component` | string | No | — | Filter by component using dot-separated ID prefix, e.g. `"daemon"` or `"daemon.core"`. Auto-detected from `Cargo.toml`/`package.json` workspaces. |

### Output Formats

| Format | Description |
|--------|-------------|
| `tree` | Hierarchical directory tree with file names |
| `summary` | High-level overview showing directory counts and top-level structure |
| `flat` | Flat list of relative file paths |

### Examples

Get a high-level project overview:

```json
{
  "format": "summary"
}
```

Browse a specific subdirectory:

```json
{
  "path": "src/rust/daemon/core/src",
  "depth": 2,
  "format": "tree"
}
```

List all Rust source files in a component:

```json
{
  "component": "daemon.core",
  "language": "rust",
  "format": "flat",
  "includeTests": false
}
```

Find all test files:

```json
{
  "pattern": "**/*.test.ts",
  "format": "flat",
  "limit": 500
}
```

List only configuration files:

```json
{
  "fileType": "config",
  "format": "flat"
}
```

### Response Format

- `tree` format: returns a formatted directory tree string.
- `summary` format: returns counts of files per directory and language breakdown.
- `flat` format: returns an array of relative file path strings.

---

## Common Patterns

### Session initialization

At the start of each session, load behavioral rules before doing any work:

```json
{
  "tool": "rules",
  "action": "list",
  "scope": "global"
}
```

Then check for project-specific rules:

```json
{
  "tool": "rules",
  "action": "list",
  "scope": "project"
}
```

### Codebase exploration

Start with a summary to understand the project layout:

```json
{
  "tool": "list",
  "format": "summary"
}
```

Then drill into a specific area with semantic search:

```json
{
  "tool": "search",
  "query": "queue processing pipeline",
  "scope": "project",
  "limit": 10
}
```

Confirm a specific implementation detail with grep:

```json
{
  "tool": "grep",
  "pattern": "UnifiedQueueClient::connect",
  "pathGlob": "**/*.rs"
}
```

### Library lookup

Search reference documentation:

```json
{
  "tool": "search",
  "query": "async trait object safety",
  "collection": "libraries",
  "libraryName": "rust-reference",
  "limit": 5
}
```
