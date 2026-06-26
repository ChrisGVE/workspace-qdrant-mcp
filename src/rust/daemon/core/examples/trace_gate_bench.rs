//! Microbenchmark for the B1 runtime trace cost-gate (PRD task 64, AC1).
//!
//! Compares a hot loop that constructs span attributes unconditionally against
//! the same loop gated by [`tracing_gate::if_tier`] with the tier set to `Off`.
//! With the gate off, the attribute closure never runs — the per-iteration cost
//! is a single relaxed atomic load — so the overhead vs. a no-tracing baseline
//! must sit within the §7 budget (≤2% CPU).
//!
//! Run with:
//! ```text
//! ORT_LIB_LOCATION=$HOME/.onnxruntime-static/lib \
//!   cargo run --release --example trace_gate_bench -p workspace-qdrant-core
//! ```
//!
//! This is a long-term-temporary benchmark artifact (not a CI test): the
//! ≤2% guarantee is structural (no closure evaluation + one atomic load), and
//! the closure-non-evaluation invariant is asserted deterministically in the
//! `tracing_gate` unit tests. This binary lets the timing be reproduced ad hoc.

use std::hint::black_box;
use std::time::Instant;

use workspace_qdrant_core::tracing_gate::{if_tier, set_trace_tier, TraceTier};

const ITERS: u64 = 50_000_000;

/// Stand-in for expensive span-attribute construction (allocates + formats).
fn build_attrs(i: u64) -> String {
    format!("file_{}.rs:{}:collection=projects", i % 997, i)
}

fn main() {
    // Baseline: hot loop doing only the trivial work, no tracing at all.
    let start = Instant::now();
    let mut acc = 0u64;
    for i in 0..ITERS {
        acc = acc.wrapping_add(black_box(i));
    }
    let baseline = start.elapsed();
    black_box(acc);

    // Gated-off: same loop, but each iteration also consults the gate. The
    // attribute closure must NOT run (tier is Off), so the added cost is the
    // atomic load only.
    set_trace_tier(TraceTier::Off);
    let start = Instant::now();
    let mut acc = 0u64;
    for i in 0..ITERS {
        acc = acc.wrapping_add(black_box(i));
        if let Some(s) = if_tier(TraceTier::Hot, || build_attrs(i)) {
            black_box(s);
        }
    }
    let gated_off = start.elapsed();
    black_box(acc);

    let base_ns = baseline.as_secs_f64() / ITERS as f64 * 1e9;
    let gate_ns = gated_off.as_secs_f64() / ITERS as f64 * 1e9;
    let overhead_pct = (gate_ns - base_ns) / base_ns * 100.0;

    println!("iterations:        {ITERS}");
    println!("baseline:          {baseline:?} ({base_ns:.3} ns/iter)");
    println!("gated-off:         {gated_off:?} ({gate_ns:.3} ns/iter)");
    println!("gate overhead:     {overhead_pct:.2}% (§7 budget: <= 2%)");

    if overhead_pct <= 2.0 {
        println!("RESULT: within budget");
    } else {
        println!("RESULT: OVER budget (note: micro-timing is noise-sensitive; rerun pinned)");
    }
}
