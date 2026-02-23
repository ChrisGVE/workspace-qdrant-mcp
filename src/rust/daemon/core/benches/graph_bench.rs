//! Graph Performance Benchmarks (graph-rag Task 7)
//!
//! Measures:
//! 1. Edge insertion throughput (single and batch)
//! 2. Node upsert throughput (single and batch)
//! 3. 1-hop and 2-hop query latency at varying graph sizes
//! 4. Impact analysis latency
//! 5. Delete edges by file
//!
//! Targets (from PRD R10):
//! - Edge insertion: ≥10K edges/sec
//! - 1-hop query: <1ms
//! - 2-hop query: <10ms
//! - Per-file ingestion overhead: 5-15ms
//!
//! Run: cargo bench --manifest-path src/rust/Cargo.toml --package workspace-qdrant-core --bench graph_bench

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Duration;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use workspace_qdrant_core::graph::{
    EdgeType, GraphDbManager, GraphEdge, GraphNode, GraphStore, NodeType, SqliteGraphStore,
};

/// Build a graph store backed by a temp directory.
async fn setup_store(dir: &TempDir) -> SqliteGraphStore {
    let path = dir.path().join("graph.db");
    let manager = GraphDbManager::new(&path).await.unwrap();
    SqliteGraphStore::new(manager.pool().clone())
}

/// Generate N nodes for a single file in a tenant.
fn gen_nodes(tenant: &str, file_idx: usize, count: usize) -> Vec<GraphNode> {
    (0..count)
        .map(|i| {
            let mut n = GraphNode::new(
                tenant,
                format!("src/mod_{}.rs", file_idx),
                format!("func_{}_{}", file_idx, i),
                NodeType::Function,
            );
            n.start_line = Some((i * 10) as u32);
            n.end_line = Some((i * 10 + 8) as u32);
            n.language = Some("rust".into());
            n
        })
        .collect()
}

/// Generate edges forming a call chain within a file.
fn gen_edges(tenant: &str, nodes: &[GraphNode], file_idx: usize) -> Vec<GraphEdge> {
    let file = format!("src/mod_{}.rs", file_idx);
    nodes
        .windows(2)
        .map(|pair| {
            GraphEdge::new(
                tenant,
                &pair[0].node_id,
                &pair[1].node_id,
                EdgeType::Calls,
                &file,
            )
        })
        .collect()
}

/// Pre-populate a graph with `file_count` files, each with `funcs_per_file` functions.
async fn populate_graph(
    store: &SqliteGraphStore,
    tenant: &str,
    file_count: usize,
    funcs_per_file: usize,
) -> Vec<Vec<GraphNode>> {
    let mut all_nodes = Vec::with_capacity(file_count);
    for f in 0..file_count {
        let nodes = gen_nodes(tenant, f, funcs_per_file);
        let edges = gen_edges(tenant, &nodes, f);
        store.upsert_nodes(&nodes).await.unwrap();
        store.insert_edges(&edges).await.unwrap();
        all_nodes.push(nodes);
    }
    all_nodes
}

// ── Benchmarks ───────────────────────────────────────────────────────────

fn bench_edge_insertion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("graph_edge_insert");

    for batch_size in [10, 50, 100] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let dir = TempDir::new().unwrap();
                let store = rt.block_on(setup_store(&dir));
                let tenant = "bench_tenant";

                // Pre-create nodes so inserts don't fail on FK
                let nodes = gen_nodes(tenant, 0, size + 1);
                rt.block_on(store.upsert_nodes(&nodes)).unwrap();

                let edges: Vec<_> = nodes
                    .windows(2)
                    .map(|pair| {
                        GraphEdge::new(
                            tenant,
                            &pair[0].node_id,
                            &pair[1].node_id,
                            EdgeType::Calls,
                            "src/bench.rs",
                        )
                    })
                    .collect();

                b.iter(|| {
                    rt.block_on(store.insert_edges(black_box(&edges))).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_node_upsert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("graph_node_upsert");

    for batch_size in [10, 50, 100] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let dir = TempDir::new().unwrap();
                let store = rt.block_on(setup_store(&dir));
                let nodes = gen_nodes("bench_tenant", 0, size);

                b.iter(|| {
                    rt.block_on(store.upsert_nodes(black_box(&nodes))).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_query_1hop(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("graph_query_1hop");
    group.measurement_time(Duration::from_secs(5));

    for file_count in [10, 100, 500] {
        let funcs = 20;
        let label = format!("{}files_{}funcs", file_count, funcs);

        group.bench_function(BenchmarkId::new("query", &label), |b| {
            let dir = TempDir::new().unwrap();
            let store = rt.block_on(setup_store(&dir));
            let tenant = "bench_tenant";
            let all_nodes = rt.block_on(populate_graph(&store, tenant, file_count, funcs));

            // Query from the first node of the first file
            let start_id = &all_nodes[0][0].node_id;

            b.iter(|| {
                rt.block_on(store.query_related(
                    black_box(tenant),
                    black_box(start_id),
                    black_box(1),
                    None,
                ))
                .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_query_2hop(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("graph_query_2hop");
    group.measurement_time(Duration::from_secs(5));

    for file_count in [10, 100, 500] {
        let funcs = 20;
        let label = format!("{}files_{}funcs", file_count, funcs);

        group.bench_function(BenchmarkId::new("query", &label), |b| {
            let dir = TempDir::new().unwrap();
            let store = rt.block_on(setup_store(&dir));
            let tenant = "bench_tenant";
            let all_nodes = rt.block_on(populate_graph(&store, tenant, file_count, funcs));

            let start_id = &all_nodes[0][0].node_id;

            b.iter(|| {
                rt.block_on(store.query_related(
                    black_box(tenant),
                    black_box(start_id),
                    black_box(2),
                    None,
                ))
                .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_impact_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("graph_impact_analysis");
    group.measurement_time(Duration::from_secs(5));

    for file_count in [10, 100] {
        let funcs = 20;
        let label = format!("{}files_{}funcs", file_count, funcs);

        group.bench_function(BenchmarkId::new("impact", &label), |b| {
            let dir = TempDir::new().unwrap();
            let store = rt.block_on(setup_store(&dir));
            let tenant = "bench_tenant";
            let all_nodes = rt.block_on(populate_graph(&store, tenant, file_count, funcs));

            // Pick a middle function that has callers
            let target = &all_nodes[0][funcs / 2].symbol_name;

            b.iter(|| {
                rt.block_on(store.impact_analysis(
                    black_box(tenant),
                    black_box(target),
                    None,
                ))
                .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_delete_edges_by_file(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("graph_delete_by_file");

    for funcs in [10, 50, 100] {
        group.bench_function(BenchmarkId::from_parameter(funcs), |b| {
            let dir = TempDir::new().unwrap();
            let store = rt.block_on(setup_store(&dir));
            let tenant = "bench_tenant";

            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for i in 0..iters {
                    // Setup: insert nodes and edges for a fresh file
                    let nodes = gen_nodes(tenant, i as usize, funcs);
                    let edges = gen_edges(tenant, &nodes, i as usize);
                    rt.block_on(store.upsert_nodes(&nodes)).unwrap();
                    rt.block_on(store.insert_edges(&edges)).unwrap();

                    let file = format!("src/mod_{}.rs", i);
                    let start = std::time::Instant::now();
                    rt.block_on(store.delete_edges_by_file(tenant, &file))
                        .unwrap();
                    total += start.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

fn bench_reingest_file(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("graph_reingest_file");
    group.measurement_time(Duration::from_secs(10));

    for funcs in [10, 30, 50] {
        let label = format!("{}_funcs", funcs);
        group.bench_function(BenchmarkId::new("reingest", &label), |b| {
            let dir = TempDir::new().unwrap();
            let store = rt.block_on(setup_store(&dir));
            let tenant = "bench_tenant";
            let shared = workspace_qdrant_core::graph::SharedGraphStore::new(store.clone());

            // Initial population
            let nodes = gen_nodes(tenant, 0, funcs);
            let edges = gen_edges(tenant, &nodes, 0);
            rt.block_on(shared.reingest_file(tenant, "src/mod_0.rs", &nodes, &edges))
                .unwrap();

            b.iter(|| {
                rt.block_on(shared.reingest_file(
                    black_box(tenant),
                    black_box("src/mod_0.rs"),
                    black_box(&nodes),
                    black_box(&edges),
                ))
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_edge_insertion,
    bench_node_upsert,
    bench_query_1hop,
    bench_query_2hop,
    bench_impact_analysis,
    bench_delete_edges_by_file,
    bench_reingest_file,
);
criterion_main!(benches);
