//! F19 -- overwrite_payload (PUT) vs UPSERT latency benchmark (AC-F19.1).
//!
//! File: `wqm-storage-write/benches/put_vs_upsert.rs`
//! Location: `src/rust/storage-write/benches/` (dev-only, harness=false)
//! Context: Measures `overwrite_payload` (PUT) and `upsert_points` latency
//!   on a realistic 100k-point corpus and selects the F9 deletion strategy:
//!   synchronous-per-content_key PUT if PUT p50 <= N x UPSERT p50, else the
//!   AC-F19.3 batched-outside-lock fallback. N = 3 (PRD §14-Q1).
//!
//! ## Safety
//!
//! This bench creates, uses, and destroys EXACTLY ONE throwaway collection
//! named `_f19_bench`. It NEVER touches the five production collections
//! (`projects`, `libraries`, `rules`, `scratchpad`, `images`). An assertion
//! fires before every operation naming a collection.
//!
//! ## Run
//!
//! ```
//! cd src/rust
//! ORT_LIB_LOCATION=$HOME/.onnxruntime-static/lib \
//!   cargo bench -p wqm-storage-write --bench put_vs_upsert
//! ```
//!
//! ## Scope
//!
//! LOCAL loopback Qdrant only. Remote leg is DEFERRED per Chris decision
//! 2026-06-24 (no remote Qdrant endpoint configured). See the committed
//! report `docs/benchmarks/F19-put-vs-upsert.md`.

use std::collections::HashMap;
use std::time::Instant;

use qdrant_client::config::QdrantConfig;
use qdrant_client::qdrant::{
    point_id, value::Kind, vectors, vectors_config, CreateCollectionBuilder, DenseVector, Distance,
    ListValue, NamedVectors, PointId, PointStruct, SetPayloadPointsBuilder, SparseVector,
    SparseVectorConfig, SparseVectorParams, UpsertPointsBuilder, Value as QdrantValue,
    VectorParamsBuilder, VectorParamsMap, Vectors, VectorsConfig,
};
use qdrant_client::Qdrant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The ONLY collection this bench touches. Must NOT be in the production set.
const BENCH_COLLECTION: &str = "_f19_bench";

/// Production collections. Checked before every operation naming a collection.
const PRODUCTION_COLLECTIONS: &[&str] = &["projects", "libraries", "rules", "scratchpad", "images"];

/// Dense vector dimension -- matches the live `projects` collection (1536-dim).
const DENSE_DIM: u64 = 1536;

/// Points to seed before measuring (AC-F19.1: >=100k).
const SEED_POINTS: u64 = 100_000;

/// Batch size for seeding and measured iterations (AC-F19.1: batch size 1000).
const BATCH_SIZE: usize = 1000;

/// Warmup iterations before measurement begins (AC-F19.1: >=3).
const WARMUP_ITERS: usize = 3;

/// Measured iterations for each operation type (AC-F19.1: >=10).
const MEASURED_ITERS: usize = 10;

/// Decision threshold: fallback if PUT p50 > N x UPSERT p50 (PRD §14-Q1).
const N: f64 = 3.0;

/// F9 deletion SLA in seconds (PRD §14-Q1, AC-F19.2).
const SLA_S: f64 = 5.0;

/// Qdrant gRPC endpoint (qdrant-client 1.x uses gRPC; REST is port 6333,
/// gRPC is port 6334). The loopback REST port 6333 is verified by the
/// pre-bench `curl` check in the report; the qdrant-client dials 6334.
const QDRANT_URL: &str = "http://localhost:6334";

// ---------------------------------------------------------------------------
// Safety guard
// ---------------------------------------------------------------------------

/// Panic immediately if `name` is a production collection.
fn assert_not_production(name: &str) {
    assert!(
        !PRODUCTION_COLLECTIONS.contains(&name),
        "SAFETY VIOLATION: attempted operation on production collection '{name}'. \
         This bench must only touch '{BENCH_COLLECTION}'."
    );
}

// ---------------------------------------------------------------------------
// Deterministic vector / payload generation (no external RNG dep)
// ---------------------------------------------------------------------------

/// XorShift32 -- minimal deterministic PRNG. Seed with a counter to get a
/// reproducible pseudo-random sequence without pulling in `rand`.
fn xorshift32(mut state: u32) -> (u32, f32) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    // Map to [-1.0, 1.0]
    let f = ((state as f64) / (u32::MAX as f64)) as f32 * 2.0 - 1.0;
    (state, f)
}

/// Generate a deterministic dense vector of `DENSE_DIM` floats for point index `i`.
fn gen_dense(i: u64) -> Vec<f32> {
    let mut state = (i.wrapping_add(1)) as u32;
    let mut vec = Vec::with_capacity(DENSE_DIM as usize);
    for _ in 0..DENSE_DIM {
        let (s, f) = xorshift32(state);
        state = s;
        vec.push(f);
    }
    vec
}

/// Generate a deterministic sparse vector for point index `i`.
/// Uses 16 non-zero entries to mimic a realistic BM25 sparse vector.
/// Indices are GUARANTEED unique: we pick a random base in [0, 50000) then
/// add a stride of 3001 (coprime to 50000) modulo 50000, yielding 16 distinct
/// indices without any dedup post-pass.
fn gen_sparse(i: u64) -> (Vec<u32>, Vec<f32>) {
    const NNZ: usize = 16;
    const VOCAB: u32 = 50_000;
    // Stride coprime to VOCAB (3001 is prime, does not divide 50000).
    const STRIDE: u32 = 3_001;

    let mut state = (i.wrapping_add(0x9e37_79b9)) as u32;

    // Pick a random starting index in [0, VOCAB).
    let (s0, _) = xorshift32(state);
    state = s0;
    let base = (state as u64 * VOCAB as u64 / (u32::MAX as u64 + 1)) as u32;

    let mut indices = Vec::with_capacity(NNZ);
    let mut values = Vec::with_capacity(NNZ);

    for k in 0..NNZ {
        // idx = (base + k * STRIDE) mod VOCAB -- unique because gcd(STRIDE, VOCAB) = 1
        // and k < VOCAB.
        let idx = (base + (k as u32).wrapping_mul(STRIDE)) % VOCAB;
        indices.push(idx);

        let (s, v) = xorshift32(state);
        state = s;
        values.push(v.abs() + 0.01); // sparse weights must be positive
    }

    // Sort by index as Qdrant requires sorted sparse vectors.
    let mut pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
    pairs.sort_unstable_by_key(|(idx, _)| *idx);
    let (sorted_indices, sorted_values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();

    (sorted_indices, sorted_values)
}

/// Build a deterministic `BlobPayload` map for point index `i`.
/// tenant_id, branch_id (2 entries), collection_id -- mirrors F7's real payload.
fn gen_payload(i: u64) -> HashMap<String, QdrantValue> {
    let tenant = format!("tenant-bench-{}", i % 4);
    let branch_a = format!("branch-{}", i % 8);
    let branch_b = format!("branch-{}", (i + 1) % 8);

    let branch_values = vec![
        QdrantValue {
            kind: Some(Kind::StringValue(branch_a)),
        },
        QdrantValue {
            kind: Some(Kind::StringValue(branch_b)),
        },
    ];

    let mut map = HashMap::new();
    map.insert(
        "tenant_id".to_string(),
        QdrantValue {
            kind: Some(Kind::StringValue(tenant)),
        },
    );
    map.insert(
        "branch_id".to_string(),
        QdrantValue {
            kind: Some(Kind::ListValue(ListValue {
                values: branch_values,
            })),
        },
    );
    map.insert(
        "collection_id".to_string(),
        QdrantValue {
            kind: Some(Kind::StringValue("projects".to_string())),
        },
    );
    map
}

/// Build a replacement payload (survivor membership after one branch removed).
/// Mirrors the real REMOVE path: branch_b drops out.
fn gen_survivor_payload(i: u64) -> HashMap<String, QdrantValue> {
    let tenant = format!("tenant-bench-{}", i % 4);
    let branch_a = format!("branch-{}", i % 8);

    let branch_values = vec![QdrantValue {
        kind: Some(Kind::StringValue(branch_a)),
    }];

    let mut map = HashMap::new();
    map.insert(
        "tenant_id".to_string(),
        QdrantValue {
            kind: Some(Kind::StringValue(tenant)),
        },
    );
    map.insert(
        "branch_id".to_string(),
        QdrantValue {
            kind: Some(Kind::ListValue(ListValue {
                values: branch_values,
            })),
        },
    );
    map.insert(
        "collection_id".to_string(),
        QdrantValue {
            kind: Some(Kind::StringValue("projects".to_string())),
        },
    );
    map
}

/// Build one `PointStruct` for seeding (upsert with vectors + payload).
fn build_point(i: u64) -> PointStruct {
    assert_not_production(BENCH_COLLECTION);

    let dense = gen_dense(i);
    let (sparse_indices, sparse_values) = gen_sparse(i);
    let payload = gen_payload(i);

    // UUID-shaped ID: use namespace 00000000-0000-0000-0000-XXXXXXXXXXXX where X is i
    let uuid = format!(
        "00000000-0000-4000-8000-{:012x}",
        i % 0x0001_0000_0000_0000u64
    );

    let mut named = HashMap::new();
    named.insert("dense".to_string(), DenseVector { data: dense }.into());
    named.insert(
        "sparse".to_string(),
        SparseVector {
            indices: sparse_indices,
            values: sparse_values,
        }
        .into(),
    );

    PointStruct {
        id: Some(PointId {
            point_id_options: Some(point_id::PointIdOptions::Uuid(uuid)),
        }),
        vectors: Some(Vectors {
            vectors_options: Some(vectors::VectorsOptions::Vectors(NamedVectors {
                vectors: named,
            })),
        }),
        payload,
    }
}

/// Build a batch of `count` points starting at offset `start`.
fn build_upsert_batch(start: u64, count: usize) -> Vec<PointStruct> {
    (0..count as u64).map(|k| build_point(start + k)).collect()
}

/// Build a batch of SET-PAYLOAD requests (overwrite_payload / PUT) for the
/// same `count` points. The payload is the survivor payload (simulates REMOVE).
fn build_put_batch_requests(
    start: u64,
    count: usize,
) -> Vec<(PointId, HashMap<String, QdrantValue>)> {
    (0..count as u64)
        .map(|k| {
            let i = start + k;
            let uuid = format!(
                "00000000-0000-4000-8000-{:012x}",
                i % 0x0001_0000_0000_0000u64
            );
            let pid = PointId {
                point_id_options: Some(point_id::PointIdOptions::Uuid(uuid)),
            };
            (pid, gen_survivor_payload(i))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

/// Compute the p-th percentile of a sorted slice (0.0 <= p <= 100.0).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Bench report writer
// ---------------------------------------------------------------------------

fn write_report(
    put_p50_ms: f64,
    put_p95_ms: f64,
    upsert_p50_ms: f64,
    upsert_p95_ms: f64,
    ratio: f64,
    selected_strategy: &str,
    qdrant_version: &str,
    hw_cpu: &str,
    hw_ncpu: &str,
    hw_mem_gb: f64,
    hw_os: &str,
) {
    // CARGO_MANIFEST_DIR = .../workspace-qdrant-mcp-branch-mgmt/src/rust/storage-write
    // Three .parent() calls reach the repo root.
    let report_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // .../src/rust
        .unwrap()
        .parent() // .../src
        .unwrap()
        .parent() // .../workspace-qdrant-mcp-branch-mgmt (repo root)
        .unwrap()
        .join("docs/benchmarks");

    std::fs::create_dir_all(&report_dir).expect("create docs/benchmarks");
    let report_path = report_dir.join("F19-put-vs-upsert.md");

    let content = format!(
        r#"# F19 Benchmark Report: overwrite_payload (PUT) vs upsert_points (UPSERT)

## Purpose

AC-F19.1: measure `overwrite_payload` (PUT) latency vs a paired `upsert_points`
baseline on a realistic 100k-point corpus, and select the F9 deletion strategy.

## Environment (verbatim)

| Parameter | Value |
|-----------|-------|
| CPU | {hw_cpu} |
| CPU cores | {hw_ncpu} |
| RAM | {hw_mem_gb:.1} GB |
| OS | {hw_os} |
| Qdrant version | {qdrant_version} |
| Qdrant endpoint | localhost:6334 gRPC / localhost:6333 REST (loopback) |
| Collection name | _f19_bench (throwaway, deleted after bench) |
| Dense vector dim | {DENSE_DIM} (matches live `projects` collection) |
| Sparse entries/point | 16 non-zero (BM25-realistic) |
| Point count (seed) | {SEED_POINTS} |
| Batch size | {BATCH_SIZE} (AC-F19.1) |
| Warmup iterations | {WARMUP_ITERS} (AC-F19.1: >=3) |
| Measured iterations | {MEASURED_ITERS} each (AC-F19.1: >=10) |
| Payload per point | tenant_id (string), branch_id (2-entry array), collection_id (string) |
| PUT payload | survivor payload (1 branch -- simulates F9 REMOVE) |

## Measured Latencies (per 1000-point batch)

| Metric | PUT (overwrite_payload) | UPSERT (upsert_points) |
|--------|------------------------|------------------------|
| p50 | {put_p50_ms:.2} ms | {upsert_p50_ms:.2} ms |
| p95 | {put_p95_ms:.2} ms | {upsert_p95_ms:.2} ms |

## Strategy Selection (AC-F19.1)

| Parameter | Value |
|-----------|-------|
| PUT p50 / UPSERT p50 ratio | {ratio:.3}x |
| Threshold N (PRD section 14-Q1) | {N}x |
| Ratio <= N? (synchronous OK?) | {ratio_ok} |
| Selected strategy | {selected_strategy} |
| F9 deletion SLA (AC-F19.2) | {SLA_S} s |

**Interpretation:** {interpretation}

## Remote Leg: DEFERRED

The remote Qdrant leg is deferred per Chris decision 2026-06-24. No remote Qdrant
endpoint is configured. The local loopback ratio ({ratio:.3}x) is the gate value
used until the remote leg runs.

Follow-up: run the same bench against the remote endpoint and update this report.
The ratio may differ due to network RTT; if the remote ratio exceeds N=3, the
batched-outside-lock fallback (AC-F19.3) activates for remote deployments.

## AC-F19.2: F9 Deletion SLA

F9 deletion SLA = {SLA_S} s (PRD section 14-Q1, valid only if F19 PUT benchmark
supports it). With PUT p50 = {put_p50_ms:.2} ms per 1000-point batch, a 400-batch
job (400k blobs, worst case) completes in ~{batch_400_s:.0} s on this hardware.
SLA {sla_verdict}.

## AC-F19.3: Fallback Code

`MembershipPutBatch` accumulator lives in
`src/rust/storage-write/src/qdrant/membership_batch.rs`.
Idempotency test: `batch_built_twice_from_same_state_is_identical` (no live Qdrant).

## AC-F19.4: Strategy Documented

Chosen strategy: **{selected_strategy}**

See also: `docs/ARCHITECTURE.md` (Write Path Architecture section) for the
pointer to this report.
"#,
        hw_cpu = hw_cpu,
        hw_ncpu = hw_ncpu,
        hw_mem_gb = hw_mem_gb,
        hw_os = hw_os,
        qdrant_version = qdrant_version,
        put_p50_ms = put_p50_ms,
        put_p95_ms = put_p95_ms,
        upsert_p50_ms = upsert_p50_ms,
        upsert_p95_ms = upsert_p95_ms,
        ratio = ratio,
        ratio_ok = if ratio <= N { "YES" } else { "NO" },
        selected_strategy = selected_strategy,
        interpretation = if ratio <= N {
            format!(
                "PUT p50 ({put_p50_ms:.2} ms) is within {N}x of UPSERT p50 ({upsert_p50_ms:.2} ms). \
                 F9 Step-6 synchronous per-content_key PUT meets the SLA on this hardware."
            )
        } else {
            format!(
                "PUT p50 ({put_p50_ms:.2} ms) exceeds {N}x UPSERT p50 ({upsert_p50_ms:.2} ms). \
                 F9 Step-6 activates the batched-outside-lock fallback (AC-F19.3)."
            )
        },
        batch_400_s = put_p50_ms / 1000.0 * 400.0,
        sla_verdict = if put_p50_ms / 1000.0 * 400.0 <= SLA_S {
            "MET"
        } else {
            "EXCEEDED -- fallback required"
        },
    );

    std::fs::write(&report_path, content).expect("write F19 report");
    println!("Report written to: {}", report_path.display());
}

// ---------------------------------------------------------------------------
// Main bench
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    println!("=== F19 Benchmark: PUT vs UPSERT on _f19_bench ===");
    println!("Connecting to {QDRANT_URL} ...");

    // 1. Connect to Qdrant.
    let config = QdrantConfig::from_url(QDRANT_URL);
    let client = Qdrant::new(config).expect("Qdrant::new");

    // 2. Get server version (also verifies Qdrant is reachable).
    let qdrant_version = match client.health_check().await {
        Ok(resp) => resp.version,
        Err(e) => {
            eprintln!(
                "ERROR: Qdrant at {QDRANT_URL} is unreachable: {e}\n\
                 Ensure `memexd` / Qdrant is running before running this bench."
            );
            std::process::exit(1);
        }
    };
    println!("Qdrant version: {qdrant_version}");

    // 3. Safety check: list existing collections and confirm production set intact.
    let list = client.list_collections().await.expect("list_collections");
    let existing: Vec<String> = list.collections.iter().map(|c| c.name.clone()).collect();
    println!("Existing collections before bench: {:?}", existing);

    // Assert the bench collection is not a production collection (static safety check).
    assert_not_production(BENCH_COLLECTION);

    // 4. Teardown / recreation: delete _f19_bench if it exists, then recreate.
    assert_not_production(BENCH_COLLECTION);
    if existing.iter().any(|n| n == BENCH_COLLECTION) {
        println!("Collection '{BENCH_COLLECTION}' exists, deleting for clean slate...");
        client
            .delete_collection(BENCH_COLLECTION)
            .await
            .expect("delete existing _f19_bench");
        println!("Deleted existing '{BENCH_COLLECTION}'.");
    }

    // 5. Create _f19_bench with 1536-dim dense + sparse, matching projects schema.
    println!("Creating collection '{BENCH_COLLECTION}' (dense={DENSE_DIM}, sparse)...");
    assert_not_production(BENCH_COLLECTION);

    let mut dense_map = HashMap::new();
    dense_map.insert(
        "dense".to_string(),
        VectorParamsBuilder::new(DENSE_DIM, Distance::Cosine).build(),
    );
    let vectors_cfg = VectorsConfig {
        config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
            map: dense_map,
        })),
    };

    let mut sparse_map = HashMap::new();
    sparse_map.insert(
        "sparse".to_string(),
        SparseVectorParams {
            index: None,
            modifier: None,
        },
    );
    let sparse_cfg = SparseVectorConfig { map: sparse_map };

    let create_req = CreateCollectionBuilder::new(BENCH_COLLECTION)
        .vectors_config(vectors_cfg)
        .sparse_vectors_config(sparse_cfg)
        .build();

    client
        .create_collection(create_req)
        .await
        .expect("create _f19_bench");
    println!("Collection '{BENCH_COLLECTION}' created.");

    // 6. Seed >= 100k points in batches of 1000.
    println!(
        "Seeding {SEED_POINTS} points in batches of {BATCH_SIZE} ({} batches)...",
        SEED_POINTS / BATCH_SIZE as u64
    );
    let seed_batches = SEED_POINTS / BATCH_SIZE as u64;
    for b in 0..seed_batches {
        let start = b * BATCH_SIZE as u64;
        let points = build_upsert_batch(start, BATCH_SIZE);
        assert_not_production(BENCH_COLLECTION);
        let req = UpsertPointsBuilder::new(BENCH_COLLECTION, points).wait(true);
        client.upsert_points(req).await.expect("seed upsert");
        if (b + 1) % 20 == 0 {
            println!("  Seeded {}/{seed_batches} batches", b + 1);
        }
    }
    println!("Seed complete: {SEED_POINTS} points in '{BENCH_COLLECTION}'.");

    // 7. Choose a stable window of points for the measured batches.
    // Use the middle of the seeded range so we hit indexed segments.
    let measure_start: u64 = SEED_POINTS / 2;

    // --- Warmup ---------------------------------------------------------------

    println!("Warming up ({WARMUP_ITERS} iters each)...");

    for w in 0..WARMUP_ITERS {
        // Warmup UPSERT: re-upsert a batch of existing points.
        {
            let points = build_upsert_batch(measure_start, BATCH_SIZE);
            assert_not_production(BENCH_COLLECTION);
            let req = UpsertPointsBuilder::new(BENCH_COLLECTION, points).wait(true);
            client.upsert_points(req).await.expect("warmup upsert");
        }
        // Warmup PUT: overwrite payload of the same batch.
        {
            let entries = build_put_batch_requests(measure_start, BATCH_SIZE);
            for (pid, payload) in entries {
                assert_not_production(BENCH_COLLECTION);
                let req = SetPayloadPointsBuilder::new(BENCH_COLLECTION, payload)
                    .points_selector(vec![pid])
                    .wait(true);
                client.overwrite_payload(req).await.expect("warmup put");
            }
        }
        println!("  Warmup iter {} done", w + 1);
    }
    println!("Warmup complete.");

    // --- Measure UPSERT -------------------------------------------------------

    println!("Measuring UPSERT ({MEASURED_ITERS} iters, {BATCH_SIZE} points/iter)...");
    let mut upsert_times: Vec<f64> = Vec::with_capacity(MEASURED_ITERS);

    for m in 0..MEASURED_ITERS {
        let points = build_upsert_batch(measure_start, BATCH_SIZE);
        assert_not_production(BENCH_COLLECTION);
        let req = UpsertPointsBuilder::new(BENCH_COLLECTION, points).wait(true);
        let t0 = Instant::now();
        client.upsert_points(req).await.expect("measured upsert");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        upsert_times.push(elapsed_ms);
        println!("  UPSERT iter {}: {:.2} ms", m + 1, elapsed_ms);
    }

    // --- Measure PUT (overwrite_payload) -------------------------------------

    println!("Measuring PUT ({MEASURED_ITERS} iters, {BATCH_SIZE} points/iter)...");
    let mut put_times: Vec<f64> = Vec::with_capacity(MEASURED_ITERS);

    for m in 0..MEASURED_ITERS {
        // Batch PUT: send one SetPayloadPoints per point in the batch.
        // This mirrors the real F9 path: per-point overwrite_payload calls
        // (batched outside the lock in AC-F19.3 fallback).
        let entries = build_put_batch_requests(measure_start, BATCH_SIZE);
        assert_not_production(BENCH_COLLECTION);
        let t0 = Instant::now();
        for (pid, payload) in entries {
            let req = SetPayloadPointsBuilder::new(BENCH_COLLECTION, payload)
                .points_selector(vec![pid])
                .wait(true);
            client.overwrite_payload(req).await.expect("measured put");
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        put_times.push(elapsed_ms);
        println!("  PUT iter {}: {:.2} ms", m + 1, elapsed_ms);
    }

    // --- Statistics -----------------------------------------------------------

    upsert_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    put_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let upsert_p50 = percentile(&upsert_times, 50.0);
    let upsert_p95 = percentile(&upsert_times, 95.0);
    let put_p50 = percentile(&put_times, 50.0);
    let put_p95 = percentile(&put_times, 95.0);
    let ratio = if upsert_p50 > 0.0 {
        put_p50 / upsert_p50
    } else {
        f64::NAN
    };

    let selected_strategy = if ratio <= N {
        "synchronous (PUT p50 <= 3x UPSERT p50)"
    } else {
        "batched-outside-lock fallback (PUT p50 > 3x UPSERT p50)"
    };

    println!("\n=== RESULTS ===");
    println!("UPSERT p50: {upsert_p50:.2} ms  p95: {upsert_p95:.2} ms");
    println!("PUT    p50: {put_p50:.2} ms  p95: {put_p95:.2} ms");
    println!("PUT/UPSERT ratio (p50): {ratio:.3}x  (threshold N={N})");
    println!("Selected strategy: {selected_strategy}");
    println!("F9 deletion SLA: {SLA_S} s");

    // --- Collect hardware info ------------------------------------------------

    let hw_cpu = run_cmd("sysctl", &["-n", "machdep.cpu.brand_string"]);
    let hw_ncpu = run_cmd("sysctl", &["-n", "hw.ncpu"]);
    let hw_mem_bytes: u64 = run_cmd("sysctl", &["-n", "hw.memsize"])
        .trim()
        .parse()
        .unwrap_or(0);
    let hw_mem_gb = hw_mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let hw_os = run_cmd("sw_vers", &[]);

    // --- Write report ---------------------------------------------------------

    write_report(
        put_p50,
        put_p95,
        upsert_p50,
        upsert_p95,
        ratio,
        selected_strategy,
        &qdrant_version,
        hw_cpu.trim(),
        hw_ncpu.trim(),
        hw_mem_gb,
        hw_os.trim(),
    );

    // --- Teardown: DELETE _f19_bench ------------------------------------------

    println!("Teardown: deleting '{BENCH_COLLECTION}'...");
    assert_not_production(BENCH_COLLECTION);
    match client.delete_collection(BENCH_COLLECTION).await {
        Ok(_) => println!("Deleted '{BENCH_COLLECTION}' successfully."),
        Err(e) => eprintln!(
            "WARNING: teardown delete of '{BENCH_COLLECTION}' failed: {e}\n\
             You may need to delete it manually via: \
             curl -X DELETE http://localhost:6333/collections/{BENCH_COLLECTION}"
        ),
    }

    // --- Verify production collections are intact ----------------------------

    let after = client
        .list_collections()
        .await
        .expect("post-bench list_collections");
    let after_names: Vec<String> = after.collections.iter().map(|c| c.name.clone()).collect();
    println!("Collections after bench: {:?}", after_names);

    assert!(
        !after_names.iter().any(|n| n == BENCH_COLLECTION),
        "SAFETY VIOLATION: '{BENCH_COLLECTION}' still exists after teardown"
    );

    for prod in PRODUCTION_COLLECTIONS {
        if existing.iter().any(|n| n == *prod) {
            assert!(
                after_names.iter().any(|n| n == *prod),
                "SAFETY VIOLATION: production collection '{prod}' is missing after bench"
            );
        }
    }

    println!(
        "\nProduction collections intact: {:?}",
        PRODUCTION_COLLECTIONS
    );
    println!("=== F19 Benchmark COMPLETE ===");
    println!("PUT p50={put_p50:.2}ms  UPSERT p50={upsert_p50:.2}ms  ratio={ratio:.3}x  strategy={selected_strategy}");
}

// ---------------------------------------------------------------------------
// Shell command helper
// ---------------------------------------------------------------------------

fn run_cmd(prog: &str, args: &[&str]) -> String {
    std::process::Command::new(prog)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}
