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

