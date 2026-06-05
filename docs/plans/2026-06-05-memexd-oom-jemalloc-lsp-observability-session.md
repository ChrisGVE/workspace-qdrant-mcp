# Session outcomes — 2026-06-05: memexd OOM, jemalloc, LSP scalable startup, observability, lease reaper

Branch: `claude/fix-memexd-chunk-oom` (all commits pushed to origin).

## 1. memexd OOM — runaway chunk splitting (the catastrophic leak)

`split_chunk_with_overlap` (`tree_sitter/chunker/splitting.rs`) had a **non-terminating loop**.
`find_line_boundary` (`rfind('\n')`) snapped `actual_end` back to an early newline; when a long run
**without newlines** followed it (> `target_size`), the next `start = actual_end - FRAGMENT_OVERLAP`
never advanced, allocating fragments without bound (jeprof: ~85% of an 11 GB live heap in
`SemanticChunk::new` + `RawVec::finish_grow`). Trigger: `tests/language-support/pascal/bookshelf` — a
**committed 1.1 MB Mach-O binary** with a 228 KB no-newline line.

**Fix (defense in depth):**
- splitting: stride forward by a guaranteed `step_size` from the window start (not from `actual_end`);
  accept a line-boundary cut only if it still covers a full stride. Terminates in ~`total_fragments`
  iterations, keeps the overlap, no coverage gaps.
- extraction: binary gate in `document_processor/extraction/text.rs` — reject content with a NUL byte in the
  first 8 KiB (UTF BOM excepted) before the lossy/legacy decode that turned executables into garbage "text".
- hygiene: fixed `tests/language-support/.gitignore` (build.sh moves the fpc output to `pascal/bookshelf`,
  the old ignore pointed at `pascal/src/bookshelf`) and untracked the committed binary.

## 2. RSS bloat is glibc allocator RETENTION, not a leak → jemalloc

After the loop fix, sustained embedding still grew RSS to ~11 GB while the **live heap stayed ~3 GB**
(`jeprof --inuse_space`: ~71 % `onnxruntime::BFCArena`, retained by design during concurrent embedding).
glibc malloc does not return freed arenas to the OS.

**Fix:** bake jemalloc into `docker/Dockerfile.memexd` — `libjemalloc2` + `ENV LD_PRELOAD=libjemalloc.so.2`
+ `MALLOC_CONF=background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:10000`. Validated: draining 2480
queued items kept RSS flat ~2.4 GB vs 11 GB under glibc.

**Diagnostic rule:** to distinguish a code leak from allocator retention, run the same load under jemalloc and
check whether RSS recedes (it did: peak ~4.5 GB → ~2.9 GB). See `docs/runbooks/memexd-heap-profiling.md`.

## 3. Observability — daemon self-exports CPU/RSS (cAdvisor fails on WSL2)

cAdvisor on Docker-Desktop/WSL2 only sees host systemd slices, not the `wqm-*` container cgroups (they live
in the separate `docker-desktop` WSL distro), so per-container metrics never populate. Instead the daemon
self-exports its own process metrics (each container reads its own `/proc`):

- `memexd_process_resident_memory_bytes` — RSS from `/proc/self/statm`
- `memexd_process_cpu_percent` — delta of `/proc/self/stat` utime+stime over the 1 s sampling interval

Sampled in `start_uptime_tracker` (`memexd/src/background.rs`), registered on the daemon registry, plotted on
the Grafana **system-overview** ("memexd CPU (%)", "memexd Memory (RSS)") plus a **Recent Error Traces** panel
(Tempo TraceQL `{ status = error }`).

## 4. LSP scalable startup (3 phases)

Fixes the first-enrichment-query timeout (server still indexing) and the activation thundering-herd.

- **Phase 1 — warm-up grace.** `is_server_ready_for_file` defers enrichment until `start + grace`, where
  grace = `lsp.warmup_grace_secs` floor raised by a per-language minimum (java 120 s, rust 60 s,
  go/c/cpp 30 s, others 5 s). Deferred chunks stay `pending` and are backfilled by the metadata-uplift pass.
- **Phase 2 — real readiness.** `ServerInstance.ready_signal` (AtomicBool) is flipped by a notification
  handler that correlates `$/progress` begin/end by indexing-titled token (avoids false-early on unrelated
  progress) and honors jdtls `language/status` Started/ServiceReady. Readiness = `signalled OR grace-elapsed`
  via a lock-free atomic that never blocks behind an in-flight LSP request.
- **Phase 3 — staggered starts.** `start_semaphore` (default 2, env `WQM_LSP_MAX_CONCURRENT_STARTS`) holds a
  permit from spawn until the server signals ready (or the grace elapses), so activating many projects does
  not run N heavy indexers at once. (LSP server memory is **outside** the daemon's jemalloc arena.)

## 5. Stale-lease reaper (queue resilience)

`unified_queue` items can stick at `in_progress` when a worker dies mid-process (panic/OOM/SIGKILL). The
reclaim query (`recover_stale_unified_leases`: reset `in_progress` rows with an expired `lease_until` back to
`pending`) existed but ran **only at startup** (`queue_init.rs`). Added `spawn_stale_lease_reaper` — a 60 s
periodic task in `spawn_recovery_tasks` — so orphaned leases recover at runtime, not just on the next restart.

## Config knobs added

| Knob | Where | Default | Purpose |
|---|---|---|---|
| `lsp.warmup_grace_secs` | YAML | 15 | Global warm-up grace floor (per-language minimums apply on top). |
| `WQM_LSP_MAX_CONCURRENT_STARTS` | env | 2 | Phase-3 stagger: max servers warming/indexing at once. |
| `MALLOC_CONF` / `LD_PRELOAD` | Dockerfile | (jemalloc) | Bound RSS to the live heap. |
