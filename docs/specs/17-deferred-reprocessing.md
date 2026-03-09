# Spec 17: Deferred Re-processing for Grammar/LSP Availability Changes

## Problem

Files ingested before a grammar is downloaded or an LSP server is running receive
degraded processing:

- **Missing grammar**: File is text-chunked instead of semantically chunked
  (`treesitter_status = 'skipped'` or `'failed'`).
- **Missing LSP**: File lacks code intelligence enrichment
  (`lsp_status = 'skipped'` or `'failed'`).

When the grammar becomes available or the LSP server starts, these files should
be re-processed to upgrade their quality — without requiring the user to
manually touch or re-save them.

## Design

### Tracking: Why a File Was Degraded

The `tracked_files` table already has `treesitter_status` and `lsp_status`
columns with values `none | done | failed | skipped`. A file is a candidate for
deferred re-processing when:

| Column              | Value       | Meaning                                |
|---------------------|-------------|----------------------------------------|
| `treesitter_status` | `skipped`   | Grammar not available at ingest time   |
| `treesitter_status` | `failed`    | Grammar load/parse error (may be transient) |
| `lsp_status`        | `skipped`   | LSP server not running at ingest time  |
| `lsp_status`        | `failed`    | LSP request failed (timeout, crash)    |

No new columns are needed. The existing status values already encode the reason.

### Trigger Events

#### Grammar Download Completion

When `GrammarManager::download_and_load()` succeeds for language `L`:

1. Query `tracked_files` for files with `language = L` AND
   `treesitter_status IN ('skipped', 'failed')`.
2. Enqueue each as `(item_type=file, op=uplift, ...)` into the unified queue.
3. The `uplift` operation signals the processor to bypass the hash
   short-circuit and re-process the file.

#### LSP Server Becomes Healthy

When an LSP server transitions to `ServerStatus::Running` for language `L`:

1. Query `tracked_files` for files with `language = L` AND
   `lsp_status IN ('skipped', 'failed')`.
2. Enqueue each as `(item_type=file, op=uplift, ...)`.

#### Manual Trigger

A CLI command `wqm admin uplift [--language L] [--tenant T]` allows the user to
manually trigger re-processing for degraded files.

### The `uplift` Queue Operation

Add a new `QueueOperation::Uplift` variant. This operation:

1. **Bypasses the hash short-circuit**: The file content hasn't changed, so the
   `file_hash` will match. The `uplift` operation skips the hash comparison and
   proceeds directly to `ingest_file_content`.
2. **Does NOT delete existing Qdrant points first**: The `base_point` is
   content-addressed (`SHA256(tenant_id|branch|relative_path|file_hash)[:32]`).
   Since the file hasn't changed, the `base_point` is the same. The processor
   deletes old points and re-upserts — same as a normal update where the hash
   changes, but triggered by capability upgrade rather than content change.
3. **Updates `treesitter_status` / `lsp_status`** to `done` on success, or
   `failed` with a new `last_error` on failure.

### Query for Affected Files

```sql
SELECT file_id, file_path, watch_folder_id, branch, language
FROM tracked_files
WHERE language = ?1
  AND (treesitter_status IN ('skipped', 'failed')
       OR lsp_status IN ('skipped', 'failed'))
```

Index: The existing `idx_tracked_files_language` covers this query efficiently.

### Batching and Priority

- **Batch size**: Enqueue at most 500 files per trigger event. If more files are
  affected, enqueue in batches with a 1-second delay between batches to avoid
  flooding the queue.
- **Priority**: `uplift` items should be processed at lower priority than normal
  `add`/`update` operations. The dequeue logic should prefer non-uplift items
  when both are pending. This can be achieved by sorting by `op` in the dequeue
  query (`add` < `update` < `uplift`).

### Idempotency

- The `idempotency_key` for an uplift is
  `SHA256(file|uplift|tenant_id|collection|payload_json)[:32]`.
- If the same file is enqueued for uplift multiple times (e.g., grammar download
  completes while a previous uplift is still pending), the `INSERT OR IGNORE`
  deduplication prevents double-processing.
- If the uplift fails and is retried, the `decision_json` mechanism ensures
  retry-safe execution.

### Re-processing Safety

- **Unchanged content**: The file content hasn't changed, so the new
  `base_point` equals the old one. Delete old Qdrant points by `base_point`,
  then upsert new points with the same `base_point`. This is atomic from
  Qdrant's perspective.
- **Concurrent modifications**: If the file is modified while an uplift is
  pending, the normal `update` operation will supersede the uplift (the
  `file_path` UNIQUE constraint in the queue ensures only one pending item per
  file). The update re-processes with the new content and new hash, rendering
  the uplift unnecessary.
- **Grammar still unavailable**: If the uplift runs but the grammar is still not
  available (e.g., download failed), the status remains `skipped`/`failed`. No
  infinite retry loop — the next grammar availability event will re-trigger.

## Implementation Sequence

1. Add `Uplift` variant to `QueueOperation` enum (`wqm-common`).
2. Update `enqueue_unified` to accept `uplift` as a valid operation.
3. Update the file processing strategy to handle `uplift`:
   - Skip hash comparison.
   - Delete old points by `base_point`.
   - Re-run `ingest_file_content` with current grammar/LSP state.
   - Update `treesitter_status` / `lsp_status` in `tracked_files`.
4. Add trigger in `GrammarManager::download_and_load()` to enqueue uplifts.
5. Add trigger in LSP lifecycle when server reaches `Running` status.
6. Add `wqm admin uplift` CLI command.
7. Update dequeue priority to prefer non-uplift items.

## Non-Goals

- **Automatic grammar discovery**: This spec does not change when grammars are
  downloaded. It only defines what happens after a grammar becomes available.
- **Partial re-processing**: The entire file is re-processed. Surgical
  chunk-level updates are a future optimization.
- **Cross-project coordination**: Each project's files are uplifted
  independently based on their own `tracked_files` records.
