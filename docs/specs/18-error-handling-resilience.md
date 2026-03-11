# Spec 18: Error Handling and Resilience

## Overview

This spec documents the canonical error handling model for the unified queue
processor, covering how each failure mode is classified, what action is taken,
and how the daemon recovers autonomously from systemic failures.

The guiding principle is **self-healing over operator intervention**: the daemon
should resolve transient failures on its own, expose diagnostics for persistent
failures, and never silently discard items that could be re-processed later.

---

## 1. Error Taxonomy

Every processing failure is classified into one of five categories. The category
determines the fate of the queue item and the daemon's response.

### 1.1 `permanent_gone`

**Definition:** The target resource no longer exists and never will again.

**Examples:**
- File deleted or moved before processing (`FileNotFound`)
- Project deregistered between enqueue and processing (`no watch_folder found`)
- File access permanently blocked by OS-level ACL (`permission denied`)

**Action:** Silently dequeue — `DELETE FROM unified_queue WHERE queue_id = ?`.
No error logged (or `debug!` at most). The resource is gone; retrying is
meaningless and would flood logs.

**Note:** `FileNotFound` is not an error condition — it is a normal event in a
filesystem-watching system. The item is quietly removed without incrementing any
error counter.

**Invariant — both destinations must be marked Done before returning Ok:**
The `cleanup_missing_file` path must explicitly set `qdrant_status = Done` and
`search_status = Done` before returning `Ok(())`. `check_and_finalize()` only
deletes an item when both destinations are `Done`; if either remains
`in_progress` (a stale value from a prior failed attempt), the item survives
finalization and re-enters the queue on the next stale-lease recovery cycle.

### 1.2 `permanent_data`

**Definition:** The payload or data is structurally invalid and retrying will
never succeed without a code fix or manual correction.

**Examples:**
- `payload_json` fails deserialization (`InvalidPayload`)
- Queue operation validation failure (`validation failed`)
- Unsupported item type or operation combination (`UnsupportedOperation`)

**Causes of `InvalidPayload` in practice:**
- Daemon version mismatch: an item enqueued by an older daemon version uses a
  field name or structure that the current version no longer recognises. This is
  the dominant cause during active development.
- Enqueue-side bug: serialising the wrong struct, missing required fields.
- Manual SQLite intervention producing malformed JSON.
- Truncated write (disk full at write time, though SQLite WAL makes this rare).

**Action:** Log at `error!` level (this signals a code bug or schema mismatch),
then permanently fail the item — set `status = 'failed'`, no retry. The
`error_message` is prefixed `[permanent_data]` for observability. The item is
**not** resurrected by the periodic retry task (see §4).

### 1.3 `transient_resource`

**Definition:** A local subsystem that the daemon owns failed transiently. The
failure is expected to self-resolve once the subsystem recovers.

**Examples:**
- Embedding model or ONNX runtime fails to initialise at startup
- Inference OOM during embedding generation
- Embedding semaphore closed during shutdown

**Action:** Retry up to `max_retries` (default: 3) with exponential backoff
(`60s × 2^retry_count`, capped at 1 hour). If all retries are exhausted, set
`status = 'failed'`. The item is eligible for periodic resurrection (see §4).

**Daemon response:** The embedding subsystem uses lazy re-initialisation (see
§3). The daemon does not crash; it continues processing all non-embedding work
while attempting to recover the subsystem in the background.

### 1.4 `transient_infrastructure`

**Definition:** An external service the daemon depends on is temporarily
unavailable. The failure is expected to self-resolve once the service recovers.

**Examples:**
- Qdrant unreachable (service stopped, network partition)
- Qdrant returning 5xx errors
- SQLite `SQLITE_BUSY` / database locked

**Action:** Unlike `transient_resource`, an infrastructure outage affects
**every** item in the queue simultaneously. Processing individual items and
retrying each one independently is wasteful and produces misleading error counts.

The correct response is a **queue-wide pause**:
1. After a configurable number of consecutive `Storage` errors
   (`circuit_breaker_threshold`, default: 5), the processor enters
   **infrastructure pause mode**.
2. All dequeuing stops.
3. A lightweight health probe (e.g., `GET /healthz` on Qdrant) runs every
   `infra_probe_interval_secs` (default: 30).
4. When the probe succeeds, the processor resumes normal dequeuing.
5. During the pause, SQLite-only operations (lease recovery, status updates)
   continue unaffected — the local database is always available.

Items that already failed with `transient_infrastructure` before the circuit
breaker opened are eligible for periodic resurrection (see §4).

### 1.5 `partial`

**Definition:** The item was partially processed — some destinations succeeded,
others failed. Only the failed destinations need retrying.

**Examples:**
- Qdrant upsert succeeded but SQLite search index update failed
- Enrichment phase failed after base embedding succeeded

**Action:** Retry only the failed destinations. The item remains in the queue
with per-destination status tracking (`qdrant_status`, `search_status`). Not
currently resurrected by the periodic task — the per-destination retry handles
recovery.

---

## 2. Classification Logic

Classification is performed in `UnifiedQueueProcessor::classify_error()` in
`unified_queue_processor/metrics.rs`. The mapping is:

| `UnifiedProcessorError` variant | Default category | Exceptions (message-sniffed) |
|---|---|---|
| `FileNotFound` | `permanent_gone` | — |
| `InvalidPayload` | `permanent_data` | — |
| `UnsupportedOperation` | `permanent_data` | — |
| `Embedding` | `transient_resource` | — |
| `Storage` | `transient_infrastructure` | — |
| `QueueOperation` | `transient_infrastructure` | `"no watch_folder found"` → `permanent_gone`; `"validation failed"` → `permanent_data` |
| `ProcessingFailed` | `transient_infrastructure` | `"permission denied"` / `"access denied"` → `permanent_gone`; `"unsupported"` → `permanent_data` |
| `ShutdownRequested` | *(silently skip)* | Item is left `in_progress`; stale lease recovery picks it up at next startup |

**Known classification gaps** (tracked for future improvement):
- HTTP 404 on URL items is classified as `transient_infrastructure` but should
  be `permanent_gone` (the resource is gone).
- `ShutdownRequested` falls through to the `_` wildcard catch-all. It currently
  works by accident because stale lease recovery resets `in_progress` items to
  `pending`. It should be handled explicitly.

---

## 3. Daemon Self-Recovery: Lazy Subsystem Re-initialisation

### 3.1 Problem

When a critical subsystem (currently: the embedding generator) fails to
initialise at startup, the daemon continues running but all embedding work fails
immediately. Items accumulate `transient_resource` errors and exhaust their
retries, landing in `status = 'failed'`. There is no automatic recovery path.

### 3.2 Design: Degraded Mode + Background Re-init

The daemon operates in **degraded mode** when the embedding subsystem is
unavailable:

- File watching, git events, folder scans, tenant management, and all
  non-embedding work continue normally.
- Embedding items dequeued during degraded mode are re-queued immediately with a
  backoff delay rather than being processed (they are not counted against their
  `max_retries` budget in this path — see §3.3).
- A background task (`EmbeddingWatchdog`) periodically attempts to
  re-initialise the embedding generator.

### 3.3 EmbeddingWatchdog

**Location:** `src/rust/daemon/core/src/embedding/watchdog.rs`

**Behaviour:**
1. On startup, if the embedding generator fails to initialise, the daemon marks
   it as `EmbeddingState::Unavailable` and starts the watchdog.
2. The watchdog attempts re-initialisation at increasing intervals:
   `[30s, 60s, 120s, 300s, 600s]` (capped at 10 minutes).
3. On success: `EmbeddingState` transitions to `Available`. All suspended
   embedding items are eligible to be dequeued immediately (their `lease_until`
   is cleared).
4. On failure: interval advances. After `max_watchdog_attempts` (default: 10)
   consecutive failures, the watchdog writes a structured diagnostic to
   `~/.workspace-qdrant/embedding-failure.json` and triggers a controlled daemon
   shutdown so launchd can restart it.
5. On daemon restart: launchd's `ThrottleInterval` (configured in the plist)
   prevents infinite restart loops. If the restart also fails to initialise
   embeddings, the watchdog cycle repeats.

**Diagnostic file format (`embedding-failure.json`):**
```json
{
  "timestamp": "2026-03-10T14:23:00Z",
  "daemon_start": "2026-03-10T14:20:00Z",
  "total_attempts": 10,
  "last_error": "OrtError: failed to load model from ...",
  "ort_lib_location": "/Users/chris/.onnxruntime-static/lib",
  "model_path": "~/.cache/huggingface/...",
  "action": "controlled_shutdown"
}
```

### 3.4 Interaction with Queue Items

- While the embedding subsystem is `Unavailable`, `Embedding` errors during
  dequeue are **not** counted against `retry_count`. Instead, the item's
  `lease_until` is set to `NOW() + short_delay` (default: 60s) and the status
  reset to `pending`. This preserves the retry budget for genuine inference
  failures once the subsystem recovers.
- Once the subsystem becomes `Available`, items in `status = 'pending'` with
  `lease_until` in the future are eligible as soon as the lease expires.
  There is no special re-queue step needed.

---

## 4. Periodic Resurrection of Failed Items

### 4.1 Problem

Items that exhaust their `max_retries` budget land in `status = 'failed'` and
are permanently invisible to the dequeue query. If the underlying cause was
transient (e.g., FastEmbed was broken at startup), the items will never be
processed even after the daemon recovers.

### 4.2 Design

A periodic **resurrection task** runs in the processing loop's idle path (same
mechanism as metadata uplift and grammar idle checks). It:

1. Finds all items with `status = 'failed'` whose `error_message` starts with
   `[transient_` (i.e., `[transient_resource]` or `[transient_infrastructure]`).
2. Resets them to `status = 'pending'`, `retry_count = 0`, `lease_until = NULL`.
3. The items re-enter the normal dequeue cycle.

**Frequency:** Configurable via `failed_resurrection_interval_secs` (default:
3600 — once per hour).

**Scope:** Only `transient_*` categories. `permanent_data` and `permanent_gone`
items are never resurrected. Items without a `[transient_*]` prefix in
`error_message` are treated as permanent.

**Bounded behaviour:** The resurrection task itself has no hard cap on how many
times an item can be resurrected. The assumption is that `transient_*` failures
are always expected to resolve eventually (Qdrant comes back, model is fixed,
etc.). If an item never resolves, it will oscillate between `pending` and
`failed` at the hourly cadence, which is observable via `wqm queue list
--status=failed`. Operators can investigate using `wqm debug errors`.

**SQL:**
```sql
UPDATE unified_queue
SET status        = 'pending',
    retry_count   = 0,
    lease_until   = NULL,
    worker_id     = NULL,
    qdrant_status = NULL,
    search_status = NULL,
    updated_at    = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE status = 'failed'
  AND error_message LIKE '[transient_%'
```

> **Invariant:** Any operation that resets `status` back to `pending` (retry,
> resurrection, stale-lease recovery) **must** also reset `qdrant_status` and
> `search_status` to `NULL`. Leaving stale `in_progress` in a destination column
> while `status = 'pending'` causes `ensure_destinations_resolved()` to preserve
> the stale value, which then prevents `check_and_finalize()` from ever deleting
> the item — creating an infinite processing loop.

---

## 5. Future Work

The following improvements are out of scope for the current implementation but
inform the long-term direction:

### 5.1 Separate Watcher Daemon

Split the daemon into two processes:

- **`wqmd-watch`**: Extremely simple, robust filesystem watcher. Enqueues change
  events to SQLite. No ONNX dependency. Highly unlikely to crash.
- **`memexd`**: Does the heavy processing (embedding, Qdrant, LSP). Can crash
  and restart without losing filesystem events, which are buffered by
  `wqmd-watch` in the queue.

This eliminates the situation where an embedding failure causes the filesystem
watcher to stop. It also reduces database lock contention because the two
processes have well-separated write paths.

### 5.2 Re-queue on Error Instead of Retry-in-Place

Currently, retries increment `retry_count` in-place on the same queue row.
A cleaner model would:
- On first failure: mark the original item `failed`.
- Insert a new queue item with `op = retry`, preserving the original
  `queue_id` in `metadata` for traceability.
- The retry item has its own independent retry budget.
- This makes the failure history append-only and fully auditable.

### 5.3 HTTP Status Code Classification for URL Items

URL items should classify HTTP 4xx responses as `permanent_gone` (the resource
is gone) rather than `transient_infrastructure`. HTTP 429 (Too Many Requests)
should be `transient_infrastructure` with a longer backoff.

### 5.4 Explicit `ShutdownRequested` Handling

`ShutdownRequested` should be handled as a distinct path: silently leave the
item `in_progress` (stale lease recovery at next startup will reset it to
`pending`). It should not touch `retry_count` or `error_message`.
