# Handover: Project Scan Handler - TESTED AND WORKING

**Date:** 2026-02-05
**Status:** ✅ End-to-End Testing Complete

## Summary

The project scan handler has been fully tested and is working correctly. Files are being scanned, queued, ingested, and are searchable via MCP tools.

## Test Results

### Scan Performance
- **Files queued:** 9,509
- **Files excluded:** 305,123 (respecting .gitignore, build artifacts, etc.)
- **Errors:** 0
- **Scan time:** 16.3 seconds

### Ingestion Status
- **Points in Qdrant:** 1,734+ (and growing)
- **Processing rate:** ~10-20 files per 10 seconds
- **Queue depth:** ~9,000 files remaining

### Search Verification
```bash
# This query returns results:
mcp-cli call workspace-qdrant/search '{"query": "LSP", "collection": "projects", "limit": 3}'

# Results include content from indexed files with correct tenant isolation
```

## Issue Fixed During Testing

**Problem:** The scan handler wasn't being called - logs showed "Unsupported operation Scan"

**Root Cause:** The daemon binary wasn't rebuilt after the code changes. The clean/build process didn't fully recompile the changes.

**Fix:**
```bash
# Force recompilation by touching the source file
touch src/rust/daemon/core/src/unified_queue_processor.rs

# Rebuild with ONNX Runtime location (Intel Mac)
ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo build --release \
  --manifest-path src/rust/Cargo.toml --package memexd

# Reinstall and restart
launchctl stop com.workspace-qdrant.memexd
cp src/rust/target/release/memexd ~/.local/bin/
launchctl start com.workspace-qdrant.memexd
```

## Architecture Verified

```
MCP Server                    Daemon
    │                           │
    │── RegisterProject ───────>│
    │                           ├── Create watch_folders (if new)
    │                           ├── Queue (Project, Scan)    ✅
    │<─── project_id ───────────│
    │                           │
    │                    [Queue Processor]
    │                           ├── Process (Project, Scan)  ✅
    │                           │   ├── Walk directory       ✅
    │                           │   ├── Filter files         ✅
    │                           │   └── Queue File items     ✅
    │                           ├── Process (File, Ingest)   ✅
    │                           │   ├── Read file content    ✅
    │                           │   ├── Chunk content        ✅
    │                           │   ├── Generate embeddings  ✅
    │                           │   └── Store in Qdrant      ✅
    │                           │
    │── search ────────────────>│
    │<─── results (working) ────│  ✅
```

## Code References

| File | Line | Description |
|------|------|-------------|
| `unified_queue_processor.rs` | 920 | `scan_project_directory` function |
| `unified_queue_processor.rs` | 882 | `QueueOperation::Scan` case in `process_project_item` |
| `project_service.rs` | ~60 | Scan queuing on new project registration |
| `patterns/exclusion.rs` | 409 | `should_exclude_file` function |

## Known Issues

### Exclusion Patterns Too Permissive
The current exclusion filters are not strict enough. Hidden files and directories (starting with `.`) are being queued for ingestion when they should generally be excluded.

**Principle:** Hidden files/directories in projects should NOT be indexed by default. They typically contain:
- Tool caches (`.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`)
- IDE/editor configs (`.vscode/`, `.idea/`)
- Version control (`.git/` - already excluded)
- Environment configs (`.env` files)
- Build artifacts and caches

**Exceptions** (hidden files that MAY be useful to index):
- `.github/` - GitHub Actions workflows (useful for understanding CI/CD)
- `.gitignore` - Project structure info
- Configuration files at root level (`.eslintrc`, `.prettierrc`, etc.) - debatable

**File to modify:** `src/rust/daemon/core/src/patterns/exclusion.rs`

**Action required:** Add a rule to exclude all hidden files/directories at ANY depth in the project tree:
- `/project/.hidden/` - excluded
- `/project/src/.cache/` - excluded
- `/project/deep/path/.mypy_cache/` - excluded

The rule should match any path component starting with `.` (e.g., regex: `/\.[^/]+` in path), with explicit exceptions for useful dirs like `.github/` that apply at all levels.

### Exclusion Cleanup Not Implemented
**Question:** If we add new exclusions, will the daemon remove already-ingested files that now match?

**Answer:** NO - this doesn't exist. The scan only queues new files; it doesn't retroactively clean up.

**Action required:** Implement cleanup mechanism that:
1. On startup or exclusion pattern change, scan Qdrant for points matching new exclusions
2. Queue `(File, Delete)` items for files that should no longer be indexed
3. Alternatively, provide a CLI command: `wqm admin cleanup-excluded`

### Tree-sitter/LSP Enrichment Status
**Question:** How do we know if tree-sitter/LSP info is being ingested?

**Answer:** Check the payload fields in Qdrant:
- `document_type` - Shows file classification (e.g., `Code("shell")`, `Code("rust")`)
- `lsp_enrichment_status` - Shows LSP status (`Skipped`, `Enriched`, `Failed`)
- `lsp_enrichment_error` - Error reason if skipped/failed

**Current state:** LSP enrichment shows `Skipped` with error `Project not active`.

**BUG:** This is incorrect behavior. Project activity should ONLY affect queue priority, NOT feature availability. LSP enrichment should run regardless of whether the project is active or not.

**Action required:** Fix the LSP enrichment logic to not check project activity state. The activity flag is for prioritization only:
- Active project = higher queue priority
- Inactive project = lower queue priority
- Both should get full LSP/tree-sitter enrichment

**Check via:**
```bash
curl -s -X POST "http://localhost:6333/collections/projects/points/scroll" \
  -H "Content-Type: application/json" \
  --data '{"limit": 5, "with_payload": true}' | \
  jq '.result.points[].payload | {file_path, document_type, lsp_enrichment_status}'
```

## Next Steps

1. **Fix exclusion patterns:** Add hidden file/folder exclusion at all depths
2. **Monitor queue drain:** ~9,000 files remaining - will complete over time
3. **File watcher integration:** Connect notify file watcher for real-time updates
4. **Performance optimization:** Consider batch processing, parallel embedding
5. **Memory rules:** Integrate memory collection for behavioral rules

## Commands for Monitoring

```bash
# Check queue progress
sqlite3 ~/.workspace-qdrant/state.db "SELECT status, COUNT(*) FROM unified_queue WHERE item_type='file' GROUP BY status"

# Check Qdrant point count
curl -s http://localhost:6333/collections/projects | jq '.result.points_count'

# Check daemon logs
tail -f ~/Library/Logs/workspace-qdrant/daemon.jsonl | jq -r '.fields.message'

# Test search
mcp-cli call workspace-qdrant/search '{"query": "your query here", "collection": "projects", "limit": 5}'
```
