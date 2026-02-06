# Handover: Post-Scan Exclusion Cleanup

**Date:** 2026-02-06
**Status:** Feature implemented, needs rebuild and deployment

## Completed Work

### Previous Session
1. Hidden file/directory exclusion at all depths (commit `9fa47c4b`)
2. LSP enrichment no longer gated by project activity (commit `9fa47c4b`)

### This Session
3. **Auto-removal of excluded files from Qdrant after scan** (commit `33eb8101`)
   - Task Master: task 503 (development tag) - marked done
   - Added `scroll_file_paths_by_tenant()` to StorageClient (`storage.rs`)
   - Added `cleanup_excluded_files()` to UnifiedQueueProcessor (`unified_queue_processor.rs`)
   - Called automatically after `scan_project_directory()` completes
   - Queues (File, Delete) items at priority 7 for any indexed file matching exclusion rules
   - 3 unit tests added and passing (6 total in module)
   - All 626 daemon lib tests pass (1 pre-existing flaky file watcher test)

## What Was Built

**StorageClient.scroll_file_paths_by_tenant()** (`storage.rs:692-778`):
- Paginates through Qdrant using scroll API with tenant_id filter
- Extracts `file_path` from each point's payload
- Handles offset-based pagination in batches of 100
- Uses existing retry_operation for resilience

**UnifiedQueueProcessor.cleanup_excluded_files()** (`unified_queue_processor.rs:1068-1172`):
- Checks if collection exists before scrolling
- Scrolls all file paths for the tenant from Qdrant
- Strips project root to get relative path, checks `should_exclude_file()`
- Queues File/Delete items at priority 7 (higher than scan's priority 5)
- Graceful degradation: errors logged but don't fail the scan

## Rebuild & Deploy

The daemon binary needs to be rebuilt with these changes:

```bash
# Build with ONNX Runtime (Intel Mac)
ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo build --release \
  --manifest-path /Users/chris/dev/projects/mcp/workspace-qdrant-mcp/src/rust/Cargo.toml \
  --package memexd

# Restart daemon with new binary
launchctl stop com.workspace-qdrant.memexd
cp /Users/chris/dev/projects/mcp/workspace-qdrant-mcp/src/rust/target/release/memexd ~/.local/bin/
launchctl start com.workspace-qdrant.memexd
```

## Pre-existing Issues

- 1 flaky test: `watching::tests::single_folder_watch_tests::test_detect_file_modification` (timing-sensitive, intermittent)
- 2 missing test modules: `lsp_daemon_integration_tests`, `daemon_state_persistence_tests` (test "mod" compilation fails)
- 64+ pre-existing deprecation warnings in `queue_operations.rs` (legacy API)

## Next Steps

No further tasks in the development tag are pending. Potential follow-up work:
1. Rebuild and deploy the daemon binary
2. Push commits to remote (currently 11 ahead of origin/main)
3. Verify cleanup works with a real Qdrant instance by triggering a project scan
