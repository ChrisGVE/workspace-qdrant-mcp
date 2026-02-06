# Handover: Daemon Crash Fix and Queue Processing

**Date:** 2026-02-06
**Status:** All issues resolved, daemon running, queue processed

## Completed Work

### Previous Sessions
1. Hidden file/directory exclusion at all depths (commit `9fa47c4b`)
2. LSP enrichment no longer gated by project activity (commit `9fa47c4b`)
3. Auto-removal of excluded files from Qdrant after scan (commit `33eb8101`)

### This Session
4. **Fixed daemon SIGABRT crash** (commit `38e068f5`)
   - Root cause: `document_processor.rs` line 631 — string slicing at byte offset that fell inside multi-byte UTF-8 character (`─`, 3-byte box-drawing char)
   - The `panic = "abort"` release profile converted the panic to SIGABRT (exit code 134)
   - Both `chunk_by_paragraphs` and `chunk_by_characters` had the same bug
   - Added `floor_char_boundary()` helper to find nearest valid UTF-8 boundary
   - Fixed overlap slicing in `chunk_by_paragraphs`
   - Fixed chunk boundary calculation in `chunk_by_characters` + infinite loop guard
   - 3 new unit tests: `test_floor_char_boundary`, `test_chunk_by_paragraphs_with_multibyte_overlap`, `test_chunk_by_characters_with_multibyte`
   - All 13 document_processor tests pass

5. **Resolved 167 stuck queue items**
   - 167 items stuck as `in_progress` from 17 crash-loop worker instances
   - Reset to `pending`, rebuilt binary with static ONNX Runtime, redeployed
   - Removed `ORT_DYLIB_PATH` from launchd plist (was pointing to non-existent file)
   - Manually cleaned 19 `.fastembed_cache` and `.mypy_cache` files from queue (shouldn't have been queued)
   - All 167 items now processed, queue is empty

6. **Pushed 13 commits to origin/main**

## Known Issues / Follow-up Tasks

### Priority 5 in queue
- User noted queue items should have priorities 0 and 1 only, but `calculate_priority()` in `watching_queue.rs` uses 3, 5, and 8
- Needs clarification on intended priority scheme

### File watcher missing exclusion check
- `watching_queue.rs` uses only basic glob patterns from watch_folders config (`.git/*`, `__pycache__/*`)
- Does NOT use the comprehensive `should_exclude_file()` exclusion engine
- This is how `.fastembed_cache` files got into the queue
- **Recommended fix**: Add `should_exclude_file()` check in `enqueue_file_operation()` before queueing

### Pre-existing Issues
- 1 flaky test: `watching::tests::single_folder_watch_tests::test_detect_file_modification`
- 2 missing test modules: `lsp_daemon_integration_tests`, `daemon_state_persistence_tests`
- 64+ deprecation warnings in `queue_operations.rs` (legacy API)

## Daemon Status
- Running via launchd: `com.workspace-qdrant.memexd`
- Binary: `/Users/chris/.local/bin/memexd` (statically linked ONNX Runtime)
- Queue: empty (all items processed)
- Build command: `ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo build --release --manifest-path src/rust/Cargo.toml --package memexd`
