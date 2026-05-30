# Benchmark Harness — tool_overhead

Criterion micro-benchmarks for MCP tool-call overhead.  These benchmarks
measure **pure-CPU per-call cost** (envelope construction, input parsing,
payload canonicalisation) that does NOT hit the daemon, Qdrant, or SQLite.

## Latency Parity Targets (S7)

| Metric | Target |
|---|---|
| Tool-call overhead (any tool) | ≤ 5 ms p50 |
| Search e2e p50 | Within ±10 % of TS p50 |
| Search e2e p95 | Within ±15 % of TS p95 |
| Grep / List / Retrieve p50 | Within ±10 % of TS p50 |

These targets come from S7 in the MCP-Rust-rewrite PRD.

## Warmup Parameters (PERF-01)

| Parameter | Value | Rationale |
|---|---|---|
| Warmup calls N | ≥ 20 | PERF-01 default; avoids JIT / allocator cold-start skew |
| Criterion warm-up time | 3 s | At ≤ 5 ms/call this discards ≥ 600 calls >> 20 |
| Measurement time | 5 s | Statistically stable steady-state window |
| Sample size | 100 | Sufficient for p50 / p95 confidence |

Criterion automatically discards all measurements taken during the warm-up
phase before computing statistics.  The N = 20 requirement is therefore
satisfied conservatively (by a large margin) by the 3 s warm-up.

## Benchmark Groups

### `envelope`

| Benchmark | What it measures |
|---|---|
| `unknown_tool` | `CallToolResult` construction for unknown name |
| `error_text` | `CallToolResult` construction for error path |
| `ok_text_search_response` | Full success envelope including `serde_json::to_string_pretty` |
| `to_string_pretty_search_response` | JSON serialisation only (3-result `SearchResponse`) |

### `input_parsing`

| Benchmark | What it measures |
|---|---|
| `SearchInput/typical` | `serde_json::from_value` deserialisation of `SearchInput` |
| `ListInput/typical` | `ListInput::from_args` field extraction |
| `RetrieveInput/typical` | `RetrieveInput::from_args` field extraction |
| `GrepInput/typical` | `GrepInput::from_args` field extraction |

### `stable_stringify`

| Benchmark | What it measures |
|---|---|
| `rule_payload` | Canonical JSON of a 8-field rule payload |
| `nested_store_payload` | Canonical JSON of a store payload with nested object |

## Running the Benchmarks

### Quick compile check

```sh
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib \
  cargo build --benches \
  --manifest-path src/rust/Cargo.toml \
  -p mcp-server
```

### Short smoke run (fast, exercise one group)

Criterion 0.5 (with `harness = false`) does not accept `--warm-up-time` or
`--measurement-time` on the command line; those are configured in the Rust
source via `configured_criterion()`.  To exercise a subset of benchmarks
quickly, pass a filter regex after `--`:

```sh
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib \
  cargo bench \
  --manifest-path src/rust/Cargo.toml \
  -p mcp-server \
  -- stable_stringify
```

Or run all groups (will take ~24 s per group × 3 groups with current settings):

```sh
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib \
  cargo bench \
  --manifest-path src/rust/Cargo.toml \
  -p mcp-server
```

To change warm-up / measurement times, edit `configured_criterion()` in
`benches/tool_overhead.rs`.

### Full measurement run (recommended for parity comparison)

```sh
ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib \
  cargo bench \
  --manifest-path src/rust/Cargo.toml \
  -p mcp-server
```

HTML report written to:
`src/rust/target/criterion/report/index.html`

Criterion also prints summary statistics (mean ± std, p50 ≈ median, p95 in
the PDF plot) to stdout.

## Rust vs TypeScript Parity Comparison

### Rust side

Run the full benchmark as shown above and record the mean / p50 / p95 for
each group from the HTML report or stdout output.

### TypeScript side

The TS side must be measured separately because it is a different process.
There is no automated TS runner in this bench file.

**Procedure:**

1. Build the TypeScript server:
   ```sh
   cd src/typescript/mcp-server
   npm install && npm run build
   ```

2. Start the TS server in stdio mode and send representative MCP calls via
   a small harness script (e.g. Node.js or `npx ts-node`).  Record wall-clock
   times with `performance.now()` around each `tools/call` round-trip.

3. Apply the same warmup discipline: discard the first N = 20 call timings
   before collecting statistics.

4. Collect ≥ 100 post-warmup samples and compute p50 (median) and p95
   (95th percentile) for each tool.

5. Compare Rust p50 / p95 against TS p50 / p95 using the targets in the
   table above.

### What counts as passing

- Tool-call overhead benchmarks in `envelope/` and `input_parsing/` must
  individually read below 5 ms p50 (they currently run in the microsecond
  range, so this is a conservative bound).
- For e2e search / grep / list / retrieve the bounds are relative to TS
  baseline — a separate e2e benchmark (not in this file) running against a
  live daemon and Qdrant instance would measure those.  This file covers
  only the CPU-overhead slice.
