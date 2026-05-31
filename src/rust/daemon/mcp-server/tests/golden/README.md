# Golden Fixture Corpus

Committed JSON fixtures capturing TypeScript server responses for known inputs.
The conformance suite (`tests/conformance.rs`) drives the Rust handlers with
the same inputs and compares the outputs after applying the S10.4 normalizer.

## Fixture format

Each `.json` file holds the **inner result** — the value BEFORE the MCP
content-block wrapper:

- **Success path**: the object passed to `JSON.stringify(result, null, 2)`.
- **Error envelope** (validation errors, unknown tool): `{ "is_error": true, "text": "Error: <msg>" }` or `{ "is_error": true, "text": "Unknown tool: <name>" }`.
- **In-band errors** (e.g. embedding daemon-down): the error object itself,
  with no `is_error` flag (the TS handler returns it inline, not thrown).

## S10.4 Normalizer rules

Applied to both Rust output and golden before comparison:

| Rule | Field(s) | Treatment |
|------|----------|-----------|
| 1. Volatile masking | `*_ms`, `latency_ms`, `queue_id`, `session_id`, `event_id`, `createdAt`, `updatedAt` | Replaced with `"MASKED"` sentinel |
| 2. Float precision (OQ-8) | `score`, `similarity`, `diversity_score` | Rounded to **6 decimal places** |
| 3. Equal-score order | `results[]` with tied scores | Sorted by `id` ascending |
| 4. Health presence | `health` key | Value replaced with `true` sentinel |
| 5. Parsed equality | All fields | Compare `serde_json::Value`, not raw bytes |
| 6. Canonical format | Full output | One byte-exact test per applicable golden via `serde_json::to_string_pretty` (2-space) |

## Corpus inventory (20 infra-less fixtures)

| Dir | File | Tool | Case |
|-----|------|------|------|
| `dispatch/` | `unknown_tool` | dispatch | unknown tool name |
| `dispatch/` | `unknown_tool_empty` | dispatch | empty tool name |
| `rules/` | `err_missing_action` | rules | missing `action` key |
| `rules/` | `err_invalid_action` | rules | `action: "foobar"` |
| `grep/` | `err_missing_pattern` | grep | missing `pattern` |
| `grep/` | `err_missing_pattern_null` | grep | `pattern: null` |
| `store/` | `err_missing_content` | store (library) | no `content` |
| `store/` | `err_missing_library_name` | store (library) | no `libraryName`, no `forProject` |
| `store/` | `err_project_missing_path` | store (project) | no `path` |
| `store/` | `err_project_daemon_not_connected` | store (project) | daemon offline |
| `embedding/` | `daemon_down` | embedding | daemon gRPC failure (in-band) |
| `embedding/` | `success` | embedding | fastembed healthy response |
| `search/` | `f001_refusal` | search | F-001: scope=project, no project_id |
| `search/` | `f001_refusal_multi_collection` | search | F-001: two refused collections |
| `search/` | `degraded_daemon_down` | search | daemon down, scope=all (fallback) |
| `retrieve/` | `unresolved_scope_projects` | retrieve | collection=projects, no project_id |
| `retrieve/` | `unresolved_scope_scratchpad` | retrieve | collection=scratchpad, no project_id |
| `list/` | `no_project` | list | no project_id in session or args |
| `list/` | `project_not_in_db` | list | project_id set but not in SQLite |

## Deferred to task-34 (require live daemon / Qdrant)

These cases are omitted from the default corpus. They will be added as
integration fixtures gated via `#[cfg(feature="integration-tests")]`:

- **search**: real semantic/hybrid/keyword results (score values, ranked hits)
- **retrieve**: point lookup by document_id (requires indexed Qdrant points)
- **rules**: add / update / remove / list (require live gRPC and Qdrant collection)
- **grep**: live FTS5 results from daemon TextSearchService
- **store library/url/scratchpad**: require daemon enqueue → queue_id
- **store project**: requires daemon RegisterProject gRPC call
- **list**: real tracked-files enumeration (live SQLite + daemon)

## Parity divergences

| Tool | Case | TS output | Rust output | Reason |
|------|------|-----------|-------------|--------|
| rules | `err_missing_action` | `"Invalid rules action: undefined"` | `"Invalid rules action: "` | JS coerces absent key to `undefined`; Rust uses `unwrap_or("")` |

## Updating fixtures

When the TS server changes its output for a case covered here:

1. Re-run `tmp/20260530-1000_capture_goldens.mjs` (or write a new throwaway
   capture script for the affected case).
2. Overwrite the affected `.json` file with the new captured output.
3. Run the conformance suite: `cargo test -p mcp-server conformance`.
4. If the Rust output diverges intentionally, document it in the table above.
5. Delete the capture script before committing.

## Running the conformance suite

```bash
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib \
  cargo test --manifest-path src/rust/Cargo.toml -p mcp-server conformance
```
