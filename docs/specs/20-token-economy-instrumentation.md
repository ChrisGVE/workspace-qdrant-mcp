# Spec 20: Token Economy Instrumentation

## Problem

The MCP server exists, in large part, to **shrink the context an agent has to
load** to answer a question. Where a `Read` would slurp a whole file, a
`search`/`grep`/`retrieve` returns a focused slice. That savings is the central
value proposition of the server — but today **we cannot measure it**.

We need to answer three questions with hard numbers:

1. **How many bytes is the server saving the agent per call, per session, per
   project?** — the "economy" itself.
2. **Is the agent actually getting what it wanted?** — a small payload that the
   agent rejects and re-queries is not a win.
3. **Are we trending in the right direction?** — does a server change (new
   chunker, ranking tweak, payload cap) move the needle, or regress it?

Bytes are a proxy for tokens. We cannot measure tokens directly without
binding to a specific tokenizer (and tokenizers differ across Claude
versions, OpenAI models, etc.), but `bytes ≈ tokens × 3.5–4` for English/code
is stable enough for trend analysis. The spec uses **bytes** as the unit of
record and treats tokens as a derived metric.

This spec extends [09-search-instrumentation.md](09-search-instrumentation.md);
it does not replace it. The existing `search_events` / `resolution_events`
tables, the `search_behavior` view, and the `wqm stats` command are the
substrate. Token economy adds new columns, a new view, and a new CLI subtree —
all additive.

## Design

### 1. Metrics

All metrics are computed per `search_events` row and aggregated by tool,
session, project, or time window.

#### 1.1 Per-call metrics (raw, persisted per row)

| Metric | Definition | Unit |
|---|---|---|
| `bytes_in` | Sum of bytes the agent **would have had to load** to access the same information without the tool. For `search`/`grep`: sum of file sizes referenced by the hits, capped at `top_k × FILE_PROBE_CAP` (default `FILE_PROBE_CAP = 64 KiB`) to avoid pathological inflation. For `retrieve`: full document size. For `list`: estimated bytes of an equivalent `ls -R` output on the same paths. | bytes |
| `bytes_out` | Bytes of the JSON payload the server actually ships to the MCP client, measured **after** `search-shaping.ts` runs (i.e. after truncate/summary). | bytes |
| `hits_truncated` | Number of hits whose `content`/`parent_context.unit_text` was truncated by the shaping pass (0 in summary mode). | int |
| `mode` | `truncate` (default), `summary`, or `none` (cap disabled). | enum |

Derived per-row:

```
savings_bytes  = bytes_in - bytes_out
savings_ratio  = savings_bytes / bytes_in   -- 0.0–1.0
```

`bytes_in` is intentionally conservative. It is **not** a claim about what the
agent would have done absent the server — it is an upper bound: "if the agent
had to read every file referenced in the result set, this is what it would
cost." That keeps the metric inflate-proof (we don't credit ourselves with
documents the agent never would have opened).

#### 1.2 Effectiveness signals (already partly modeled in spec 09)

| Signal | Definition | Source |
|---|---|---|
| `followup_rate` | Share of `search` events followed by another `search` from the same `session_id` within `FOLLOWUP_WINDOW = 60s` with overlapping query terms. High = first call didn't suffice. | New view; uses existing `op = 'followup'` column. |
| `escalation_rate` | Share of `search` events followed by a `retrieve` of one of the hit `documentId`s within `ESCALATION_WINDOW = 120s`. Means the trimmed body wasn't enough; agent paid the cost of the full document. | New view; joins `search_events` ↔ `resolution_events` on `search_event_id`. |
| `success_rate` | Inherited from spec 09 `search_behavior`. MCP search followed by `open`/`expand`. | Existing view. |
| `bypass_rate` | Inherited from spec 09. Agent went straight to `rg`/`grep` CLI. | Existing view. |

A search is "effective" if it is `success` and **not** `followup`/`escalation`.
A high `savings_ratio` paired with a high `escalation_rate` is a red flag: we
are shrinking payloads at the cost of forcing follow-up calls.

### 2. Schema additions

Extend `search_events` (defined in
[search_events_schema.rs](../../src/rust/daemon/core/src/unified_queue_schema/search_events_schema.rs)).
All new columns are nullable to preserve compatibility with rows written
before the migration.

```sql
ALTER TABLE search_events ADD COLUMN bytes_in       INTEGER;
ALTER TABLE search_events ADD COLUMN bytes_out      INTEGER;
ALTER TABLE search_events ADD COLUMN hits_truncated INTEGER;
ALTER TABLE search_events ADD COLUMN shape_mode     TEXT;     -- 'truncate' | 'summary' | 'none'
ALTER TABLE search_events ADD COLUMN tool_version   TEXT;     -- MCP server version, for trend attribution
```

Migration lives next to existing migrations in `unified_queue_schema/`.
Existing rows have these columns as `NULL` and are excluded from token-economy
aggregates (counted as `unknown`).

#### 2.1 New view: `token_savings`

```sql
CREATE VIEW token_savings AS
SELECT
    se.id,
    se.session_id,
    se.project_id,
    se.tool,
    se.op,
    se.shape_mode,
    se.ts,
    se.bytes_in,
    se.bytes_out,
    se.bytes_in - se.bytes_out                                      AS savings_bytes,
    CASE WHEN se.bytes_in > 0
         THEN 1.0 * (se.bytes_in - se.bytes_out) / se.bytes_in
         ELSE NULL END                                              AS savings_ratio,
    se.hits_truncated,
    EXISTS (
        SELECT 1 FROM search_events nxt
        WHERE nxt.session_id = se.session_id
          AND nxt.tool       = se.tool
          AND nxt.op         = 'followup'
          AND julianday(nxt.ts) - julianday(se.ts) BETWEEN 0 AND 0.000694  -- 60s
    ) AS had_followup,
    EXISTS (
        SELECT 1 FROM resolution_events re
        WHERE re.search_event_id = se.id
          AND re.op = 'open'
          AND julianday(re.ts) - julianday(se.ts) BETWEEN 0 AND 0.001389   -- 120s
    ) AS had_escalation
FROM search_events se
WHERE se.bytes_in IS NOT NULL;
```

Index to support the followup probe:

```sql
CREATE INDEX IF NOT EXISTS idx_search_events_session_tool_ts
    ON search_events(session_id, tool, ts);
```

### 3. Instrumentation points

#### 3.1 `search` tool

The shaping pass already runs at the single outer boundary in
[src/typescript/mcp-server/src/tools/search.ts](../../src/typescript/mcp-server/src/tools/search.ts).
Modify [search-shaping.ts](../../src/typescript/mcp-server/src/tools/search-shaping.ts)
to return a `ShapingMetrics` sidecar alongside the shaped response:

```ts
export interface ShapingMetrics {
  bytes_in_shaped: number;   // sum of pre-shape result.content + parent_context.unit_text
  bytes_out_shaped: number;  // sum of post-shape, same fields
  hits_truncated: number;
  mode: 'truncate' | 'summary' | 'none';
}

export function shapeHitPayloads(
  response: SearchResponse,
  options: SearchOptions,
): { response: SearchResponse; metrics: ShapingMetrics };
```

`bytes_in_shaped` is **not** the same as the spec's `bytes_in` — it covers
only the shaped fields. The final `bytes_in` recorded in `search_events`
also includes the file-size probe described in §1.1, computed by `SearchTool`
from the hit `documentId → file_path → file_size` mapping (already available
via the daemon's `tracked_files` table).

The call site in `search.ts` writes the event using the existing pipeline
that already populates `latency_ms` etc.

#### 3.2 `grep` tool

[src/typescript/mcp-server/src/tools/grep.ts](../../src/typescript/mcp-server/src/tools/grep.ts)
already emits a search event with `op = 'grep'`. Add:

- `bytes_out` = JSON-stringified payload length.
- `bytes_in` = sum of file sizes for each unique `file_path` in the match set,
  capped at `FILE_PROBE_CAP` per file.
- `shape_mode = 'none'` (grep does not shape today; if shaping is added later,
  populate accordingly).

#### 3.3 `retrieve` tool

Emit a `search_events` row with `op = 'retrieve'`. `bytes_in = bytes_out =
document_size` for a full retrieve (savings = 0, but the row matters for
escalation rate). For a ranged retrieve, `bytes_in = full document size`,
`bytes_out = range size`.

#### 3.4 `list` tool

`bytes_in` is approximated as `total_files × AVG_FILE_PATH_BYTES`
(default 96) for the equivalent `ls -R` output; `bytes_out` is the JSON length.
This is the loosest baseline — flag it as such in the CLI output.

#### 3.5 `store` and `rules`

Out of scope. These are write paths; no token economy to measure.

### 4. CLI surface

A new subtree under `wqm admin` (not `wqm stats`, to keep the existing CLI
grouping clean):

```
wqm admin token-savings              # 7d summary
wqm admin token-savings --window 1d
wqm admin token-savings --window 30d
wqm admin token-savings --project <id>
wqm admin token-savings --tool search
wqm admin token-savings --json
wqm admin token-savings --by-session     # session-level breakdown
wqm admin token-savings --by-day         # daily time-series
```

Default output (table form):

```text
Token Savings (last 7 days)
────────────────────────────────────────────────────────────────────
Tool      Calls   Bytes in     Bytes out    Saved %   Followup%   Escalation%
search    1,243   18.4 MiB     2.1 MiB      88.6%     12.3%       4.7%
grep        891    6.8 MiB     0.9 MiB      86.7%      4.1%       n/a
retrieve    214    4.2 MiB     3.9 MiB       7.1%      n/a        ─
list         52    1.1 MiB     0.3 MiB      72.7%      n/a        ─
────────────────────────────────────────────────────────────────────
TOTAL     2,400   30.5 MiB     7.2 MiB      76.4%
                  ↓ approx tokens (÷4): 7.6M → 1.8M = ~5.8M tokens saved
```

`--json` returns machine-readable form for piping into dashboards.

### 5. Anomaly hooks

This spec does **not** define the alerting system itself (see open question
in §8), but it intentionally provides the columns an external watcher
would need:

- A baseline of `savings_ratio` per tool per project.
- A baseline of `escalation_rate` per tool per project.
- A `tool_version` column to attribute regressions to a deploy.

A simple cron (e.g. `wqm admin token-savings --window 1d --json`) compared
against a `--window 30d` baseline is sufficient for an initial alarm: page
when `today.savings_ratio < baseline.savings_ratio - 10pp` **or** when
`today.escalation_rate > baseline.escalation_rate + 10pp`.

### 6. Privacy and storage

- **No query text is added.** `search_events.query_text` already exists from
  spec 09; this spec adds only sizes and counters. Sizes do not leak content.
- **Local-only.** All data stays in the daemon's SQLite. No remote sink.
- **Retention.** Reuse the rolling-window retention policy already in place
  for `search_events`. If retention is not yet bounded, default to **90 days**
  for `search_events` rows where `bytes_in IS NOT NULL`.
- **Per-row cost.** Five integer/text columns × ~30 bytes amortized ≈ 150 B
  per event. At 10k events/day that is ~50 MiB/year — negligible.

### 7. Implementation sequence

1. Add migration: 5 new columns on `search_events`, new index, new
   `token_savings` view.
2. Change `shapeHitPayloads` signature to return `ShapingMetrics`. Update its
   single caller in `search.ts` to capture metrics and the per-hit
   `documentId → file_size` lookup. Add `bytes_in`/`bytes_out`/`hits_truncated`/
   `shape_mode`/`tool_version` to the `SearchEventInput` written to the daemon.
3. Instrument `grep` (size sum from match `file_path`s).
4. Instrument `retrieve` (full-doc vs range).
5. Instrument `list` (loosest baseline; flagged as such in the CLI).
6. Add `wqm admin token-savings` subcommand in
   [src/rust/cli/src/commands/admin/](../../src/rust/cli/src/commands/admin/),
   following the structure of `perf.rs` + `perf_queries.rs`.
7. Add unit tests for `ShapingMetrics` math (parallel to the existing
   `search-shaping.test.ts`).
8. Update [09-search-instrumentation.md](09-search-instrumentation.md) with a
   pointer to this spec and an updated `search_events` schema table.
9. Update [08-api-reference.md](08-api-reference.md) — note that `search`,
   `grep`, `retrieve`, `list` now contribute size metrics (no behavioral
   change for callers).

Step 1+2 alone produce a working metric for the most-used tool (`search`).
Steps 3–5 broaden coverage but can ship independently.

### 8. Open questions

- **Tokenizer mapping.** Should the CLI display an estimated token count for a
  specific tokenizer (e.g. Claude's `cl100k`-equivalent), or stay strictly in
  bytes? Argument for tokens: more legible to humans reasoning about model
  budgets. Argument against: locks the metric to a tokenizer choice. Proposal:
  display both; record only bytes.
- **Alert delivery.** The §5 hooks tell us *when* something is wrong but say
  nothing about *how* the operator finds out. A follow-up spec should pick a
  sink (structured log file the user already tails, `wqm admin watch`, etc.).
- **`bytes_in` for `list`.** The `ls -R` baseline is the weakest of the four.
  Open to dropping `list` from the metric if reviewers find it noisy.

## Non-Goals

- **Real tokenizer counting.** Bytes only; tokens are derived for display.
- **Cross-tool reasoning quality.** Whether the agent's *final answer* was
  better with the MCP server in the loop is out of scope. This spec measures
  context efficiency, not answer quality.
- **Sampling.** All events are recorded. Sampling can be added later if the
  storage cost becomes meaningful (it will not at the scale projected in §6).
- **Realtime dashboards.** SQLite + CLI is the surface. A web dashboard, if
  ever wanted, can read from the same view.
