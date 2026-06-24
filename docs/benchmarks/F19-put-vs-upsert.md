# F19 Benchmark Report: overwrite_payload (PUT) vs upsert_points (UPSERT)

## Purpose

AC-F19.1: measure `overwrite_payload` (PUT) latency vs a paired `upsert_points`
baseline on a realistic 100k-point corpus, and select the F9 deletion strategy.

## Environment (verbatim)

| Parameter | Value |
|-----------|-------|
| CPU | Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz |
| CPU cores | 16 |
| RAM | 64.0 GB |
| OS | macOS 26.5.1 (Build 25F80) |
| Qdrant version | 1.18.2 |
| Qdrant endpoint | localhost:6334 gRPC / localhost:6333 REST (loopback) |
| Collection name | _f19_bench (throwaway, deleted after bench) |
| Dense vector dim | 1536 (matches live `projects` collection) |
| Sparse entries/point | 16 non-zero (BM25-realistic) |
| Point count (seed) | 100000 |
| Batch size | 1000 (AC-F19.1) |
| Warmup iterations | 3 (AC-F19.1: >=3) |
| Measured iterations | 10 each (AC-F19.1: >=10) |
| Payload per point | tenant_id (string), branch_id (2-entry array), collection_id (string) |
| PUT payload | survivor payload (1 branch -- simulates F9 REMOVE) |

## Raw Iteration Data

### UPSERT iterations (1000 points per iteration, ms)

| Iter | Latency (ms) |
|------|-------------|
| 1 | 436.31 |
| 2 | 493.83 |
| 3 | 357.47 |
| 4 | 489.53 |
| 5 | 518.54 |
| 6 | 404.88 |
| 7 | 471.61 |
| 8 | 496.24 |
| 9 | 447.31 |
| 10 | 456.32 |

### PUT (overwrite_payload) iterations (1000 points per iteration, ms)

| Iter | Latency (ms) |
|------|-------------|
| 1 | 2772.75 |
| 2 | 2601.09 |
| 3 | 2487.38 |
| 4 | 2600.35 |
| 5 | 2574.84 |
| 6 | 2589.90 |
| 7 | 3536.18 |
| 8 | 3240.38 |
| 9 | 3689.88 |
| 10 | 2967.61 |

## Measured Latencies (per 1000-point batch)

| Metric | PUT (overwrite_payload) | UPSERT (upsert_points) |
|--------|------------------------|------------------------|
| p50 | 2686.92 ms | 463.97 ms |
| p95 | 3620.71 ms | 508.50 ms |

## Strategy Selection (AC-F19.1)

| Parameter | Value |
|-----------|-------|
| PUT p50 / UPSERT p50 ratio | 5.791x |
| Threshold N (PRD section 14-Q1) | 3.0x |
| Ratio <= N? (synchronous OK?) | NO |
| Selected strategy | batched-outside-lock fallback (PUT p50 > 3x UPSERT p50) |
| F9 deletion SLA (AC-F19.2) | 5.0 s |

**Interpretation:** PUT p50 (2686.92 ms) exceeds 3x UPSERT p50 (463.97 ms).
F9 Step-6 activates the batched-outside-lock fallback (AC-F19.3).

Note: The PUT measurement issues 1000 sequential `overwrite_payload` calls
(one per point). The 5.791x ratio reflects Qdrant's per-call overhead on
this per-point sequential path vs. the batch upsert. The batched-outside-lock
fallback in `MembershipPutBatch::flush` uses the same sequential per-point
path but defers it outside the lock, preserving F04 race-freedom.

## AC-F19.2: F9 Deletion SLA

F9 deletion SLA = 5.0 s (PRD section 14-Q1). With PUT p50 = 2686.92 ms per
1000-point batch, a 400-batch job (400k blobs, worst case) would complete in
~1075 s synchronously -- far beyond the 5 s SLA. This confirms the
batched-outside-lock fallback is required: the SLA applies to the per-file
deletion operation (lock held for the SQLite DELETE + compute_membership only);
the Qdrant PUT batch flushes asynchronously after all locks are released.

## AC-F19.3: Fallback Code

`MembershipPutBatch` accumulator lives in:
`src/rust/storage-write/src/qdrant/membership_batch.rs`

Idempotency test: `batch_built_twice_from_same_state_is_identical` (no live Qdrant needed).

## Remote Leg: DEFERRED

The remote Qdrant leg is deferred per Chris decision 2026-06-24. No remote Qdrant
endpoint is configured. The local loopback ratio (5.791x) is the gate value used
until the remote leg runs.

Follow-up: run the same bench against the remote endpoint and update this report.
The ratio may differ due to network RTT; if the remote ratio also exceeds N=3, the
batched-outside-lock fallback (AC-F19.3) remains selected. If the remote ratio is
below N=3 (unlikely given the local result), synchronous mode may be enabled for
that deployment.

## AC-F19.4: Strategy Documented

Chosen strategy: **batched-outside-lock fallback (PUT p50 > 3x UPSERT p50)**

See also: `docs/ARCHITECTURE.md` (Write Path Architecture section, F19 note) for
the pointer to this report.

## Teardown Verification

After the bench run:
- `_f19_bench` collection: DELETED (confirmed via `curl localhost:6333/collections`)
- Production collections intact: projects, libraries, rules, scratchpad, images
