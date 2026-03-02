## Search Instrumentation

The system tracks search behavior to measure the effectiveness of the workspace-qdrant MCP search against traditional tools (rg, grep) and identify opportunities for improvement.

### Architecture

**Data Collection:**

Three SQLite tables track search events and resolutions:

| Table | Purpose | Writer |
|-------|---------|--------|
| `search_events` | Every search operation (tool, query, results) | MCP server, CLI wrappers |
| `resolution_events` | Document open/expand after search | External (future: IDE plugins) |
| `search_behavior` | SQL view classifying bypass/success/fallback patterns | Auto-computed |

**Instrumented Tools:**

1. **MCP Search Tool** - Logs all searches via `mcp_qdrant`
2. **CLI Wrappers** - `rg-instrumented` and `grep-instrumented` shell scripts
3. **Future: IDE Plugins** - Log resolution events when users open/expand results

### search_events Table

**Schema:**

```sql
CREATE TABLE search_events (
    id TEXT PRIMARY KEY,                    -- UUID
    session_id TEXT,                        -- MCP session or shell session
    project_id TEXT,                        -- Current project (if available)
    actor TEXT NOT NULL,                    -- 'claude' or 'user'
    tool TEXT NOT NULL,                     -- 'mcp_qdrant', 'rg', 'grep'
    op TEXT NOT NULL,                       -- 'search', 'open', 'expand'
    query_text TEXT,                        -- Search query
    filters TEXT,                           -- JSON filter spec (MCP only)
    top_k INTEGER,                          -- Result limit
    result_count INTEGER,                   -- Actual results returned
    latency_ms INTEGER,                     -- Query latency (MCP only)
    top_result_refs TEXT,                   -- JSON array of top 5 results
    ts TEXT NOT NULL,                       -- ISO 8601 timestamp (Z suffix)
    created_at TEXT NOT NULL
);
```

**Indexes:**

```sql
CREATE INDEX idx_search_events_ts ON search_events(ts);
CREATE INDEX idx_search_events_tool ON search_events(tool);
CREATE INDEX idx_search_events_session ON search_events(session_id);
```

**Example Records:**

```json
// MCP search
{
  "id": "a1b2c3d4-...",
  "session_id": "sess-xyz",
  "project_id": "abc123",
  "actor": "claude",
  "tool": "mcp_qdrant",
  "op": "search",
  "query_text": "authentication middleware",
  "filters": "{\"branch\":\"main\"}",
  "top_k": 10,
  "result_count": 7,
  "latency_ms": 45,
  "top_result_refs": "[{\"id\":\"point-1\",\"score\":0.87,...}]",
  "ts": "2026-02-15T14:23:45.123Z",
  "created_at": "2026-02-15T14:23:45.123Z"
}

// CLI wrapper (rg)
{
  "id": "e5f6g7h8-...",
  "session_id": null,
  "project_id": null,
  "actor": "claude",
  "tool": "rg",
  "op": "search",
  "query_text": "fn validate_token",
  "ts": "2026-02-15T14:24:10.456Z",
  "created_at": "2026-02-15T14:24:10.456Z"
}
```

### resolution_events Table

Tracks when users open or expand search results (indicates search was useful).

**Schema:**

```sql
CREATE TABLE resolution_events (
    id TEXT PRIMARY KEY,                    -- UUID
    search_event_id TEXT,                   -- Links to search_events.id
    session_id TEXT,
    project_id TEXT,
    actor TEXT NOT NULL,
    tool TEXT NOT NULL,                     -- Tool that produced the result
    op TEXT NOT NULL,                       -- 'open' or 'expand'
    file_path TEXT,                         -- File that was opened
    point_id TEXT,                          -- Qdrant point ID (if MCP)
    ts TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (search_event_id) REFERENCES search_events(id)
);
```

**Indexes:**

```sql
CREATE INDEX idx_resolution_events_search ON resolution_events(search_event_id);
CREATE INDEX idx_resolution_events_ts ON resolution_events(ts);
```

### search_behavior View

SQL view that classifies search patterns into behavioral categories.

**Classification Logic:**

| Pattern | Condition | Interpretation |
|---------|-----------|----------------|
| `bypass` | First event is `rg` or `grep` with no prior event | User bypassed MCP search, went straight to CLI |
| `success` | MCP search followed by `open` or `expand` within reasonable time | MCP search succeeded, user opened a result |
| `fallback` | MCP search followed by `rg`/`grep` within 2 minutes | MCP failed, user fell back to CLI |
| `unknown` | Other patterns | Uncategorized behavior |

**View Definition:**

```sql
CREATE VIEW search_behavior AS
WITH windowed_events AS (
    SELECT
        session_id,
        tool,
        op,
        ts,
        LAG(tool) OVER (PARTITION BY session_id ORDER BY ts) AS prev_tool,
        LAG(ts) OVER (PARTITION BY session_id ORDER BY ts) AS prev_ts,
        LEAD(op) OVER (PARTITION BY session_id ORDER BY ts) AS next_op,
        (julianday(ts) - julianday(LAG(ts) OVER (PARTITION BY session_id ORDER BY ts))) AS time_since_prev
    FROM search_events
    WHERE session_id IS NOT NULL
)
SELECT
    session_id,
    tool,
    op,
    ts,
    prev_tool,
    next_op,
    CASE
        WHEN tool IN ('rg', 'grep') AND prev_tool IS NULL THEN 'bypass'
        WHEN tool = 'mcp_qdrant' AND (next_op = 'open' OR next_op = 'expand') THEN 'success'
        WHEN tool = 'mcp_qdrant' AND time_since_prev < 0.00139
             AND prev_tool IN ('rg', 'grep', 'mcp_qdrant') THEN 'fallback'
        ELSE 'unknown'
    END AS behavior
FROM windowed_events;
```

**Time Window:** `0.00139` Julian days ≈ 2 minutes (reasonable time for fallback)

### CLI Commands

**View Search Stats:**

```bash
# Overview of search activity (default: last 7 days)
wqm stats overview

# Filter by time period
wqm stats overview --period day
wqm stats overview --period week
wqm stats overview --period month
wqm stats overview --period all
```

**Output Example:**

```
Search Instrumentation Stats (Last 7 days)
───────────────────────────────────────────
Total Events: 1,247

Tool Distribution:
  mcp_qdrant     856 (69%)
  rg             312 (25%)
  grep            79 (6%)

Behavior Classification:
  success        512 (60%)
  bypass         203 (24%)
  fallback       134 (16%)

Performance (mcp_qdrant):
  Searches with latency   856
  Average latency         42 ms
  P50                     38 ms
  P95                     87 ms
  P99                     124 ms

Top Queries:
  23 x  authentication middleware
  18 x  database migration
  15 x  error handling
  ...

Resolution Rate:
  Searches with resolution  634 (74%)
```

**Log Search Events (Wrapper Usage):**

```bash
# Log a search from wrapper script (fire-and-forget)
wqm stats log-search --tool=rg --query="pattern" --actor=claude
```

### Instrumented Wrappers

Lightweight shell scripts that wrap `rg` and `grep` to log search events.

**Location:** `assets/wrappers/`

**Installation:**

```bash
# Copy wrappers to PATH
cp assets/wrappers/rg-instrumented ~/.local/bin/
cp assets/wrappers/grep-instrumented ~/.local/bin/
chmod +x ~/.local/bin/rg-instrumented
chmod +x ~/.local/bin/grep-instrumented

# Configure Claude Code to use instrumented versions
# Add to ~/.claude/settings.json or project .claude/settings.json:
{
  "allowedTools": [
    "Bash(rg-instrumented *)",
    "Bash(grep-instrumented *)"
  ]
}
```

**Wrapper Implementation (rg-instrumented):**

```bash
#!/bin/bash
# Fire-and-forget event logging (background, no wait)
wqm stats log-search --tool=rg --query="$*" --actor=claude &>/dev/null &

# Pass through to real rg
exec rg "$@"
```

**Design Principles:**

1. **Fire-and-forget**: Event logging happens in background, never blocks the search
2. **Zero latency impact**: User never waits for logging to complete
3. **Graceful degradation**: If `wqm` is unavailable, search still works
4. **Drop-in replacement**: Can replace `rg` in tool configuration without code changes

### Analytics Queries

**Bypass Rate:**

```sql
SELECT
    COUNT(*) FILTER (WHERE behavior = 'bypass') * 100.0 / COUNT(*) AS bypass_rate
FROM search_behavior
WHERE ts >= date('now', '-7 days');
```

**Success Rate:**

```sql
SELECT
    COUNT(*) FILTER (WHERE behavior = 'success') * 100.0 / COUNT(*) AS success_rate
FROM search_behavior
WHERE ts >= date('now', '-7 days');
```

**Fallback Rate:**

```sql
SELECT
    COUNT(*) FILTER (WHERE behavior = 'fallback') * 100.0 / COUNT(*) AS fallback_rate
FROM search_behavior
WHERE ts >= date('now', '-7 days');
```

**Top Queries by Tool:**

```sql
SELECT tool, query_text, COUNT(*) as count
FROM search_events
WHERE query_text IS NOT NULL
  AND ts >= date('now', '-7 days')
GROUP BY tool, query_text
ORDER BY count DESC
LIMIT 10;
```

---

## Text Search (grep Tool)

The `workspace-qdrant/grep` MCP tool provides exact substring and regex pattern matching across all indexed code. It is powered by a dedicated FTS5 search database separate from the main state database.

### Architecture

**Search Database:**

- **Path:** `~/.workspace-qdrant/search.db`
- **Owner:** Rust daemon (memexd) — same ownership model as `state.db`
- Separate from `state.db` to avoid FTS5 index overhead on the state database
- Two tables: `file_metadata` (per-file-version metadata) and `code_lines` (FTS5 virtual table)

**Two Search Modes:**

| Mode | Input | Engine | Use Case |
|------|-------|--------|----------|
| Exact | Literal substring | FTS5 trigram index | `fn validate_token`, `TODO:` |
| Regex | Regular expression | Hybrid FTS5 + grep-searcher | `fn\s+\w+_test`, `\.(await\|unwrap)` |

### Exact Substring Search

Uses FTS5 trigram tokenization for fast substring matching. The query is broken into trigrams (3-character sequences) that FTS5 intersects against the index.

**Flow:**

1. Pattern → FTS5 trigram query
2. FTS5 returns candidate line rowids
3. Rust verifies exact match (FTS5 trigram matching can have false positives)
4. Apply scope filters (tenant_id, branch, path_glob)
5. Return matching lines with file metadata

### Regex Search: Hybrid FTS5 + grep-searcher Dispatch

Regex search uses a hybrid dispatch strategy. For most queries, FTS5 trigram pre-filtering narrows candidates efficiently. But for high-frequency patterns (e.g., `\.(await|unwrap|expect)\b` producing 10K+ candidates), SQLite row-fetch overhead (~3μs/row) dominates. In those cases, the engine delegates to ripgrep's `grep-searcher` crate for SIMD-accelerated file scanning.

**Dispatch flow:**

1. Extract literal substrings from the regex for FTS5 pre-filtering
2. Run a lightweight FTS5-only probe: `SELECT rowid FROM code_lines_fts WHERE content MATCH ?1 LIMIT 1 OFFSET ?2` (no JOINs, sub-millisecond)
3. If candidates exceed threshold (5,000) → delegate to `grep-searcher` module which scans source files directly via `file_metadata` paths
4. Otherwise → stream FTS5 candidates with Rust regex verification

**Dependencies:** `grep-searcher`, `grep-regex`, `grep-matcher` (ripgrep's library crates)

**Module:** `src/rust/daemon/core/src/grep_search.rs`

The grep path uses `tokio::task::spawn_blocking` for synchronous file I/O, supports context lines via grep-searcher's built-in `before_context`/`after_context`, and applies the same glob/scope filters as the FTS5 path. The `SearchResults.search_engine` field indicates which path was used (`"fts5"` or `"grep"`).

### FTS5 Query Optimization: Redundant AND Elimination

When affix merging prepends a prefix to all alternation branches (e.g., `pub (fn|struct|enum|trait|type) \w+` → branches `"pub fn "`, `"pub struct "`, etc.), the standalone mandatory term `"pub "` is redundant since it's a prefix of every branch. The query builder detects this and omits the redundant AND clause, reducing FTS5 intersection work.

### Scope and Filtering

All text searches support the following scope parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tenant_id` | Scope to a specific project | All projects |
| `branch` | Scope to a specific branch | All branches |
| `path_prefix` | Filter by file path prefix (e.g., `src/`) | None |
| `path_glob` | Filter by glob pattern (e.g., `**/*.rs`) | None |
| `case_sensitive` | Case-sensitive matching | `true` |
| `max_results` | Maximum results to return | 1000 |
| `context_lines` | Lines of context before/after each match | 0 |

When `path_glob` is set, it takes precedence over `path_prefix`. A SQL prefix is extracted from the glob for pre-filtering, then `glob::Pattern` verifies in Rust.

### Result Caching

The TextSearchService gRPC layer maintains a short-lived result cache (5-second TTL) to avoid redundant FTS5 queries when `CountMatches` and `Search` are called with the same parameters in quick succession.

**Cache key:** `(pattern, regex, case_sensitive, tenant_id, branch, path_prefix, path_glob)`

**Behavior:**
- On cache miss: runs the full query (no max_results cap, no context lines) and caches the result
- `CountMatches` returns the count from cached results
- `Search` applies `max_results` truncation from cached results; if `context_lines > 0`, re-runs with context (context is not cached to save memory)
- Cache evicts expired entries when capacity (32 entries) is reached

### Consistency with Qdrant

The search DB follows the same reference-counting deletion logic as Qdrant — the decision (keep/delete old base_point) is made **once** in the queue processor's decision phase and applied to **both** destinations:

- **Qdrant**: Delete/create chunk points
- **Search DB**: Delete/create file_metadata + code_lines entries

Within search.db, the delete-old + insert-new is **atomic** (SQLite transaction). This ensures no window where a file version is partially present.

The per-destination state machine in the unified queue (`qdrant_status`, `search_status`) tracks completion independently. Both destinations execute in **parallel** with no ordering dependency. See [Per-destination processing flow](04-write-path.md#queue-schema) for details.

### Relation to Search Instrumentation

Text search events from the `grep` MCP tool are logged as `tool = 'mcp_qdrant'` in `search_events` (since they go through the MCP server). The `op` field distinguishes semantic search (`search`) from text search (`grep`). Traditional `rg` and `grep` CLI usage is tracked separately via the instrumented wrappers described above.

---

