# Handover

## Current State

All prior tasks (574) complete. All commits pushed to `origin/main`. File watcher wired into daemon and verified working. 9 improvement points discussed across two sessions — all decisions finalized. Spec updated with automated affinity grouping section. Ready for PRD creation and task generation.

## Recently Completed Work (2026-02-11)

### Wire File Watcher into Daemon (Plan: recursive-seeking-moler.md)

4 commits implementing real-time file watching:
1. `fix(watcher)`: Align WatchManager SQL with actual `watch_folders` schema
2. `feat(grpc)`: Wire watch refresh signal through gRPC services
3. `feat(daemon)`: Wire WatchManager into daemon lifecycle
4. `build(core)`: Add wqm-common dependency to core crate

Daemon logs confirm: "File watchers started: 1 active watches" and "signal-driven: true".

### Spec Update: Automated Affinity Grouping

Added new subsection under "Automated Tagging and Grouping" in WORKSPACE_QDRANT_MCP.md covering:
- Embedding-based affinity pipeline (LLM-free)
- Taxonomy sourced from package registry categories (crates.io, npm, PyPI)
- Zero-shot taxonomy matching using existing FastEmbed model
- Open questions on embedding dimensionality, TF-IDF limitations, concept labeling gap
- Phased implementation plan

## Known Issues

- 1 pre-existing test failure: `test_alias_canonical_name_rejection` in `collection_service_tests.rs`

---

## Discussion: 9 Improvement Points

### User Request (verbatim summary)

1. **Daemon CPU spike at startup** — 100% multi-core for 10-15s after machine restart. Needs configurable self-throttling.
2. **CLI global hooks** — CLI should set up Claude Code hooks.
3. **Project disambiguation edge cases** — What happens with same-name local projects? What about remote rename?
4. **Branch creation chunk duplication** — When creating a new branch, can we duplicate existing Qdrant chunks (using stored file hashes) instead of re-embedding?
5. **Keep deleted projects in Qdrant** — Qdrant as backup. Submodules shared across projects shouldn't be deleted.
6. **MCP Server project activation/deactivation** — Is there an MCP protocol terminate signal we could use instead of hooks?
7. **Memory default scope** — Should default to project-local, not global.
8. **Cascading project changes to memory/library** — When project renames/moves, do memory and library objects update? Are libraries branch-less?
9. **Systematic functional testing** — Full test plan covering every feature of server, daemon, CLI. Use a "ladybug daemon" dummy project with full lifecycle. Reproducible. Collect all errors before fixing.

### Research Findings & Analysis

#### Point 1: Daemon Startup CPU Spike

**Root cause:** Startup recovery (`startup_recovery.rs`) enqueues ALL filesystem changes at once. Queue processor then processes with:
- 50ms inter-item delay, max 2 concurrent embeddings, batch size 10, 500ms poll interval
- Embedding (FastEmbed ONNX) is the CPU-intensive part
- Only yields every 500 files during enqueue — insufficient throttling

**Current configurable thresholds** (in `config.rs` ResourceLimitsConfig):
- `nice_level`: 10 (env: `WQM_RESOURCE_NICE_LEVEL`)
- `inter_item_delay_ms`: 50 (env: `WQM_RESOURCE_INTER_ITEM_DELAY_MS`)
- `max_concurrent_embeddings`: 2 (env: `WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS`)
- `max_memory_percent`: 70% (env: `WQM_RESOURCE_MAX_MEMORY_PERCENT`)

**Other mechanisms:** Memory pressure pause (5s at >70%), embedding semaphore, fairness scheduler (10:3 high/low ratio), exponential retry backoff.

#### Point 3: Project Disambiguation

**Same-name local projects:** WORKS. Local projects use `local_` + SHA256(canonical_path)[0:12] — different paths always produce different IDs.

**Remote rename:** GAP. Daemon stores `git_remote_url` at registration, never checks for changes. MCP server doesn't even pass `git_remote` during session activation. No periodic drift detection.

**Disambiguation code:** `DisambiguationPathComputer` in `project_disambiguation.rs` computes minimal relative paths across clones. BUT `calculate_tenant_id()` in `watching_queue.rs` does NOT apply disambiguation paths — only the full registration path does.

#### Point 4: Branch Chunk Duplication

**Current architecture:** Branch already tracked per-file via `tracked_files` UNIQUE(watch_folder_id, file_path, branch). Branch stored in Qdrant point payloads. `qdrant_chunks` tracks per-file chunks via FK.

**Point ID problem:** Currently `SHA256(document_id, chunk_index)` — NOT branch-scoped. Same file on two branches would collide.

**Comparative analysis:**

| | Branch-qualified | Content-addressed | Diff-based | Hybrid (branch ID + vector copy) |
|---|---|---|---|---|
| Mental model | Simple — each branch independent | Complex — many-to-many | Complex — chains | Simple — each branch independent |
| Deletion | Trivial (delete by branch filter) | Reference counting required | Chain invalidation risk | Trivial (delete by branch filter) |
| Search scoping | Easy (filter by branch payload) | Requires join/lookup | Requires reconstruction | Easy (filter by branch payload) |
| Storage cost | High (N copies for N branches) | Minimal (deduplicated) | Minimal (diffs only) | High (N copies of vectors) |
| Embedding CPU | High (re-embed per branch) | Zero for duplicates | Zero for duplicates | Zero for duplicates (copy vector) |
| Write complexity | Low | Medium (manage refs) | High (compute diffs) | Medium (content hash lookup) |
| Branch lifecycle | Clean (create/delete = add/remove points) | Complex (ref counting on delete) | Fragile (cascade on base delete) | Clean (create/delete = add/remove points) |

#### Point 5: Keeping Deleted Projects

Submodule consideration: A submodule may exist in multiple project groups. Deleting one project shouldn't archive the submodule if it's used elsewhere.

**Submodule schema:** Each submodule gets its own `watch_folders` entry with its own `tenant_id`, linked to parent via `parent_watch_id`. The `path` column is UNIQUE, so the same filesystem path = one row. But the same remote can appear at different paths under different parents. Archive safety check: "are there other `watch_folders` entries with the same `remote_hash`/`git_remote_url` that are still active?"

#### Point 6: MCP Protocol Terminate Signal

For STDIO transport: **NO explicit terminate signal.** SDK fires `server.onclose` when transport closes. HTTP transports have `onsessioninitialized`/`onsessionclosed` but don't apply to STDIO.

Current implementation already uses `server.onclose` → `cleanup()` → `deprioritizeProject()`.

#### Point 7: Memory Default Scope

Currently `scope = 'global'` in `memory.ts` line 127. One-line change to default to `'project'`.

#### Point 8: Cascading Changes

**NOT IMPLEMENTED.** Spec mentions it (line 2491-2493) but no code exists. When `tenant_id` changes, Qdrant memory/library records are NOT updated.

**Libraries:** Already branch-agnostic (correct). Stored under `libraries` collection with `library_name` as tenant, no branch metadata.

### Decisions Made (All Finalized)

#### Point 1: IMPLEMENT — Combined Startup Throttling

Implement ALL THREE throttling mechanisms during a configurable startup period:
- **Warmup delay** before queue processor starts consuming
- **Startup throttle mode** with reduced concurrency/increased delays
- **Batch enqueue** in startup recovery (not all at once)
- All during a configurable warmup window (default: 30 seconds)
- After warmup, resume normal operations with existing (also configurable) resource limits

#### Point 2: IMPLEMENT — CLI Global Hooks

`wqm hooks install` to set up Claude Code hooks. Non-controversial.

#### Point 3: IMPLEMENT — Remote Rename Detection with Queue-Mediated Cascade

**Detection:** Periodic check (during daemon polling or on file events) — run `git remote get-url origin` and compare with stored `git_remote_url` in `watch_folders`. The stored value IS the previous remote.

**On detection of change:**
1. SQLite transaction: update `watch_folders` (`tenant_id`, `git_remote_url`, `remote_hash`), `tracked_files`, `qdrant_chunks`
2. In same transaction: enqueue a single `cascade_rename` queue item with `{old_tenant_id, new_tenant_id, collection}`
3. Commit SQLite transaction (SQLite is always consistent)
4. Queue processor picks up cascade item, calls Qdrant `set_payload(collection, filter={tenant_id: old}, payload={tenant_id: new})`
5. If Qdrant call fails, queue retries with existing backoff/retry logic
6. Qdrant is eventually consistent

**Design rationale:** Recompute tenant_id (keeps derivation formula authoritative). Use queue for Qdrant updates (maintains integrity via retry). SQLite updated first (source of truth). This approach collapses Points 3 and 8 into a shared mechanism.

#### Point 4: IMPLEMENT — Hybrid Point ID Structure

**Decision:** Branch-qualified IDs with vector copy on content hash match.

**New point ID formula:** `SHA256(tenant_id | branch | file_path | chunk_index)`

**Branch creation workflow:**
- For each file, check if identical content hash exists on source branch
- If yes: create new point with new branch-qualified ID, copy vector from existing point (no re-embedding)
- If no: embed and create new point normally

**Rationale:** Clean mental model (each branch independent), trivial deletion/search scoping, eliminates embedding CPU cost for identical content, bounded storage (3-5 active branches typical), content hash lookup is cheap SQLite query.

#### Point 5: IMPLEMENT — Archive State with Submodule Safety

**Archive semantics:**
- Add `is_archived` flag to `watch_folders`
- Archived projects remain **fully searchable** (NO search exclusion — widened searches are for code exploration, archived projects are fair game)
- Archive only stops daemon from watching/ingesting
- User can un-archive

**Submodule safety on archive:**
- Set `is_archived = 1` on project AND its submodule entries
- **Preserve `parent_watch_id` as-is** — it's historical fact, no detaching
- Before archiving submodule data: check if same `remote_hash`/`git_remote_url` exists in any other active watch_folder entry
- If yes: submodule data stays fully active (other project still uses it)
- If no: archive alongside parent (but data remains searchable)

#### Point 6: DROPPED — No SessionEnd Hook

Existing `server.onclose` + daemon inactivity timeout is sufficient.

#### Point 7: IMPLEMENT — Memory Default to Project-Local

Change `scope = 'global'` to `scope = 'project'` in `memory.ts`. Non-controversial.

#### Point 8: IMPLEMENT — Cascading Changes (merged with Point 3)

Queue-mediated cascade mechanism (same as Point 3). When `tenant_id` changes for any reason (remote rename, project move), cascade to all Qdrant collections via queue.

#### Point 9: IMPLEMENT — Systematic Testing

Each test phase must include file changes (create, delete, change, move) with AND without content being involved.

**Test plan structure:**
- Phase A: Local project lifecycle (create, files, modifications, deletions, moves) — with/without content
- Phase B: Git initialization, branching, branch operations — with/without content changes
- Phase C: Remote setup, rename, submodule addition — with/without content changes
- Phase D: MCP server interaction (search, memory, retrieve)
- Phase E: Edge cases (concurrent changes, large files, special characters, symlinks, permissions)
- Phase F: Cleanup/archive scenarios

Use a "ladybug daemon" dummy project. Reproducible via script + git commits. Collect all errors before fixing, then batch-fix and retest.

### Additional Spec Changes Needed

1. **Python snippets** (~15 occurrences in spec): Update to Rust or TypeScript to match actual codebase
2. **Point ID formula**: Update spec to reflect new `SHA256(tenant_id | branch | file_path | chunk_index)`
3. **Archive semantics**: Add `is_archived` to watch_folders schema documentation
4. **Cascade rename mechanism**: Document the queue-mediated cascade approach
5. **Submodule archive safety**: Document the cross-reference check before archiving
6. **Automated affinity grouping**: DONE — added to spec under "Automated Tagging and Grouping"

## Next Steps — Action Items

### Quick Wins (implement first)
1. **Point 7**: Change memory default scope to project-local (one-line change)
2. **Point 1**: Implement startup warmup + throttle + batch enqueue

### Medium Effort
3. **Point 2**: CLI global hooks (`wqm hooks install`)
4. **Point 5**: Archive state for deleted projects (no Qdrant cleanup on path removal)
5. **Point 8/3**: Cascading tenant_id updates via queue-mediated cascade

### Requires Implementation
6. **Point 3**: Remote rename detection (periodic git remote check + cascade)
7. **Point 4**: Point ID redesign for branch scoping + vector copy on branch creation

### Major Project
8. **Point 9**: Create PRD for systematic functional testing, generate tasks, execute with ladybug project

### Spec Updates (do alongside implementation)
9. Update Python snippets to Rust/TS
10. Update point ID formula, archive semantics, cascade mechanism in spec

### Dropped
11. **Point 6**: SessionEnd hook — not needed, existing mechanisms sufficient

### Tagging/Affinity (future work, spec updated)
12. Implement zero-shot taxonomy matching as Phase 1 (spec section already written)
