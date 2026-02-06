# Handover: Exclusion Cleanup for Active Projects

**Date:** 2026-02-06
**Status:** Ready for implementation

## Context

Bug fixes completed in previous session:
1. Hidden file/directory exclusion at all depths (commit `9fa47c4b`)
2. LSP enrichment no longer gated by project activity (commit `9fa47c4b`)

All 627 daemon tests pass. Fixes are committed but daemon needs rebuild.

## Next Task: Auto-Removal of Excluded Files from Qdrant

**Goal:** When a project is active, automatically remove any already-indexed files that now match exclusion rules.

### Design

**Trigger points:**
1. On project activation (MCP server sends `RegisterProject` with existing project)
2. After scan completes for an active project
3. Optionally: on daemon startup for all active projects

**Implementation approach:**

1. After `scan_project_directory` completes, add a cleanup phase:
   ```rust
   async fn cleanup_excluded_files(
       tenant_id: &str,
       collection: &str,
       project_root: &Path,
       storage_client: &Arc<StorageClient>,
       queue_manager: &QueueManager,
   ) -> Result<u64> {
       // 1. Query Qdrant for all file_path values in this tenant
       // 2. For each file_path, check should_exclude_file(relative_path)
       // 3. If excluded, queue (File, Delete) item
       // 4. Return count of files queued for deletion
   }
   ```

2. Call this after scan in `process_project_item` for `QueueOperation::Scan`

3. Only run for active projects to avoid unnecessary work on dormant projects

**Key files to modify:**
- `src/rust/daemon/core/src/unified_queue_processor.rs` - Add cleanup after scan
- `src/rust/daemon/core/src/patterns/exclusion.rs` - Ensure `should_exclude_file` is accessible

**Qdrant query pattern:**
```rust
// Scroll through all points for this tenant
let filter = Filter::must([
    Condition::matches("tenant_id", tenant_id),
    Condition::matches("item_type", "file"),
]);
// Extract file_path from each point's payload
// Check exclusion rules
// Queue deletions
```

### Testing approach

1. Index a project with hidden files (before fix was applied)
2. Run cleanup on that project
3. Verify hidden files are queued for deletion
4. Verify non-hidden files are untouched

### Scope limitation

- Only clean up active projects (not all projects)
- Only run after scan completes (not continuously)
- Queue deletions rather than direct delete (respect queue processing)

## Rebuild Commands (Before Starting)

```bash
# Build with ONNX Runtime (Intel Mac)
cd /Users/chris/dev/projects/mcp/workspace-qdrant-mcp
ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo build --release \
  --manifest-path src/rust/Cargo.toml --package memexd

# Restart daemon with new binary
launchctl stop com.workspace-qdrant.memexd
cp src/rust/target/release/memexd ~/.local/bin/
launchctl start com.workspace-qdrant.memexd
```

## Files Reference

| File | Purpose |
|------|---------|
| `unified_queue_processor.rs:920` | `scan_project_directory` - add cleanup call here |
| `unified_queue_processor.rs:882` | `QueueOperation::Scan` case - alternative location |
| `patterns/exclusion.rs:409` | `should_exclude_file` function |
| `storage_client.rs` | Qdrant scroll/query operations |

## Task Master

This work should be tracked. Consider adding via:
```bash
task-master add-task --prompt="Implement auto-removal of excluded files from Qdrant for active projects after scan completes"
```
