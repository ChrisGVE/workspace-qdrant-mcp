# Handover: Bug Fixes Complete - Ready for Testing

**Date:** 2026-02-05
**Status:** Ready for verification testing

## Completed Fixes

### 1. Hidden File/Directory Exclusion at All Depths

**Problem:** Hidden files and directories (starting with `.`) were being indexed when they should be excluded by default. This included tool caches, IDE configs, etc.

**Solution:** Added `check_hidden_components()` method in `src/rust/daemon/core/src/patterns/exclusion.rs`:
- Checks ALL path components for hidden names (starting with `.`)
- Excludes any file/directory at any depth: `/project/.hidden/`, `/project/src/.cache/`, etc.
- Whitelist exception: `.github/` is allowed (useful for CI/CD understanding)

**Code locations:**
- `exclusion.rs:is_github_path()` - Line ~293-306: Whitelist check for .github
- `exclusion.rs:check_hidden_components()` - Line ~310-343: Hidden component detection
- `exclusion.rs:should_exclude()` - Line ~178-190: Integration of both checks

**Test coverage:** 4 new tests added:
- `test_hidden_files_excluded_at_all_depths`
- `test_github_directory_not_excluded`
- `test_non_hidden_paths_not_excluded_by_hidden_rule`
- `test_hidden_component_with_various_formats`

### 2. LSP Enrichment Bug - Activity Check Removed

**Problem:** LSP enrichment was skipped entirely when `is_project_active = false`, with error message "Project not active". This was incorrect - project activity should ONLY affect queue priority, not feature availability.

**Solution:** Removed the activity check gate in `enrich_chunk()` method in `src/rust/daemon/core/src/lsp/project_manager.rs`:
- The `is_project_active` parameter is now ignored (kept for API stability)
- All projects receive full LSP enrichment regardless of activity state
- Activity state only affects queue priority (handled elsewhere)

**Code locations:**
- `project_manager.rs:enrich_chunk()` - Line ~1617-1633: Activity check removed
- Tests updated to reflect new behavior

**Test changes:**
- `test_enrich_chunk_skipped_inactive_project` → renamed to `test_enrich_chunk_runs_regardless_of_activity_state`
- `test_enrich_chunk_increments_metrics` - Updated expectations
- `test_enrichment_continues_after_query_error` - Updated expectations
- `tests.rs:test_enrichment_for_inactive_project` → renamed to `test_enrichment_runs_regardless_of_activity_state`
- `tests.rs:test_metrics_tracking` - Updated expectations

## Verification Commands

### Test the exclusion fix:
```bash
# Check that hidden files are now excluded
sqlite3 ~/.workspace-qdrant/state.db "SELECT COUNT(*) FROM unified_queue WHERE item_type='file' AND payload_json LIKE '%/.%'"
# Should return 0 or very few (only .github files)

# Check .github files are still indexed
sqlite3 ~/.workspace-qdrant/state.db "SELECT COUNT(*) FROM unified_queue WHERE item_type='file' AND payload_json LIKE '%.github%'"
# Should return > 0 (GitHub Actions workflows)
```

### Test the LSP enrichment fix:
```bash
# Query Qdrant for LSP enrichment status
curl -s -X POST "http://localhost:6333/collections/projects/points/scroll" \
  -H "Content-Type: application/json" \
  --data '{"limit": 5, "with_payload": true}' | \
  jq '.result.points[].payload | {file_path, lsp_enrichment_status}'
# Should no longer show "Skipped" with "Project not active" error
```

### Run all daemon tests:
```bash
cd src/rust/daemon
ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo test --package workspace-qdrant-core --lib
# Should show: test result: ok. 627 passed
```

## Not Addressed (Parking Lot)

### Exclusion Cleanup Mechanism
The scan only queues new files; it doesn't retroactively remove already-ingested files that now match new exclusion rules. A cleanup mechanism would need to:
1. Scan Qdrant for points matching new exclusions
2. Queue `(File, Delete)` items for those files
3. OR provide CLI command: `wqm admin cleanup-excluded`

This is deferred as it's not critical for the current work.

## Next Steps

1. Wait for current ingestion to complete
2. Rebuild and restart daemon to apply fixes
3. Run verification commands above
4. Consider re-scanning projects to apply new exclusion rules to fresh data

## Rebuild Commands

```bash
# Force rebuild
touch src/rust/daemon/core/src/patterns/exclusion.rs
touch src/rust/daemon/core/src/lsp/project_manager.rs

# Build with ONNX Runtime (Intel Mac)
ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo build --release \
  --manifest-path src/rust/Cargo.toml --package memexd

# Reinstall and restart
launchctl stop com.workspace-qdrant.memexd
cp src/rust/target/release/memexd ~/.local/bin/
launchctl start com.workspace-qdrant.memexd
```
