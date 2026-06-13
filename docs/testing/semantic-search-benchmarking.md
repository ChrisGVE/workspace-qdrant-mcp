# Semantic Search Benchmarking — Process, Re-test Guide & Lessons

How to measure semantic-search quality on this repo, reproduce a run, and what
we learned doing it. Companion design doc: the embedding/rerank evolution plan
in [docs/plans/2026-05-25-search-quality-next-steps.md](../plans/2026-05-25-search-quality-next-steps.md).

---

## What the benchmark is

- **Dataset:** `src/typescript/mcp-server/scripts/benchmark-data/semantic-search-quality.yaml`
  — 12 curated _known-item_ queries (natural-language questions about this
  codebase) each with 1–2 `expectedFiles`. It is a recall/ranking benchmark, not
  a relevance-grading benchmark.
- **Runner:** `src/typescript/mcp-server/scripts/benchmark-semantic-search.ts`
  (`npm run benchmark:semantic`). Runs each query in `semantic`, `hybrid`, and
  `exact` modes, computes per-query and aggregate metrics, prints a table, and
  optionally writes a JSON report (`--output`).
- **Metrics module + verdict:** `src/typescript/mcp-server/src/benchmarks/semantic-search.ts`
  (`evaluateSearchResults`, `summarizeModeRuns`, `classifySemanticSearchQuality`).

### Metrics

| Metric              | Meaning                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| top1 / top3 / top10 | fraction of queries with an expected file in the top-K **raw** results |
| recall@10           | fraction of expected files found in the top-10 (deduped)               |
| precision@10        | relevant / returned — **reported, not gated** (see lesson 6)           |
| MRR                 | mean reciprocal rank of the first relevant hit                         |
| duplicateRate       | `1 − uniquePaths/rawPaths` — same-file chunks wasting slots            |

Verdict gates (`DEFAULT_SEMANTIC_QUALITY_THRESHOLDS`): `top3UsefulRate ≥ 0.8`
and `recallAt10 ≥ 0.7`. 0 reasons → `good`, 1 → `mixed`, ≥2 → `poor`.

---

## Re-test in-process — the `search_eval` MCP tool (preferred)

For ad-hoc, agent-driven evaluation there is an MCP tool, `search_eval`, that
runs the same harness **inside the MCP server** (real index + Qdrant) — no DB
snapshot or host runner needed. Pass inline `cases` (or omit to use the bundled
dataset when reachable):

```jsonc
search_eval({
  projectId: "local_5288aa13ad6c",            // or rely on cwd auto-detect
  cases: [
    { query: "where is reciprocal rank fusion applied",
      expectedFiles: ["src/typescript/mcp-server/src/tools/search-qdrant.ts"] }
  ],
  includeTopPaths: true                          // optional, to debug misses
})
```

It returns per-mode `top1/top3/top10`, `recall@10`, `MRR`, `duplicateRate`,
plus the verdict and per-query hit flags. It runs all three modes
(semantic/hybrid/exact) per case. From the CLI, `wqm benchmark semantic-search`
is a thin wrapper around the same runner. Use this for the fast measure→edit→measure
loop on a small case set; use the host runner below for the full curated
dataset and to (re)generate the committed `reports/semantic-search.json`.

## Re-test guide (host runner)

The runner needs the daemon's real `memexd.db`, which lives in the
`workspace-qdrant-mcp_memexd_db` Docker **named volume** (not on the host) and is
opened `{ readonly, fileMustExist }`. The runtime MCP image has no `tsx`/sources,
so we run the **source** from the host via `tsx` against the live daemon+Qdrant.

**0. Measure on a settled index.** After any daemon restart / re-scan, wait for
the queue to drain (`workspace_index` `status_all` → `queue.pending_count ≈ 0`).
Measuring mid-reindex understates quality (target files transiently absent).

**1. Snapshot the daemon DB** to a host path via a sidecar (RW mount is safe
with the live daemon; `.backup` is WAL-consistent):

```powershell
New-Item -ItemType Directory -Force -Path .\tmp | Out-Null
docker run --rm `
  -v workspace-qdrant-mcp_memexd_db:/src `
  -v "${PWD}\tmp:/out" `
  alpine sh -c "apk add --no-cache sqlite && sqlite3 /src/memexd.db '.backup /out/bench-memexd.db'"
```

**2. Run the benchmark from `src/typescript/mcp-server`.** Use `=`-form flags —
`npm run benchmark:semantic -- --flag value` strips the flag _names_ (they arrive
as positionals and error), so call `tsx` directly:

```powershell
npx tsx scripts/benchmark-semantic-search.ts `
  --project-id=local_5288aa13ad6c `        # the tenant that actually holds the data
  --qdrant-url=http://localhost:6333 `
  --daemon-host=localhost --daemon-port=50051 `
  --database-path=<repo>\tmp\bench-memexd.db `
  --output=<repo>\tmp\bench-report.json
```

Because it runs the source, **edits to `src/tools/*` are picked up with no
rebuild** — a fast measure → edit → measure loop. Deploying a change to the
_live_ MCP still needs `docker compose build mcp && docker compose up -d mcp`.

**2b. Sweep runtime search parameters without redeploy.** For rerank A/B runs,
use the sweep wrapper. It runs the same dataset repeatedly while changing
per-call search options (`rerank`, `rerankWeight`) and prints one comparison
table:

```bash
npm run benchmark:semantic:sweep -- \
  --workspace-root=/home/alkmimm/respositorios/workspace-qdrant-mcp \
  --project-id=367157a01d98 \
  --qdrant-url=http://qdrant:6333 \
  --daemon-host=localhost --daemon-port=50051 \
  --database-path=/home/alkmimm/respositorios/workspace-qdrant-mcp/tmp/bench-memexd.db \
  --weights=0,0.05,0.10,0.15,0.25,0.5,1 \
  --output=/home/alkmimm/respositorios/workspace-qdrant-mcp/tmp/bench-sweep.json
```

After redeploying the CLI image, the same sweep is available from inside the
`wqm-memexd` container as a first-class `wqm` command:

```bash
wqm benchmark semantic-search-sweep \
  --workspace-root /home/alkmimm/respositorios/workspace-qdrant-mcp \
  --project-id 367157a01d98 \
  --qdrant-url http://qdrant:6333 \
  --daemon-host localhost \
  --daemon-port 50051 \
  --database-path /home/alkmimm/respositorios/workspace-qdrant-mcp/tmp/bench-memexd.db \
  --weights 0,0.05,0.10,0.15,0.25,0.5,1 \
  --output /home/alkmimm/respositorios/workspace-qdrant-mcp/tmp/bench-sweep.json
```

For a small exploratory run, add repeated `--query-id=<id>` flags or define
custom scenarios:

```bash
npm run benchmark:semantic:sweep -- \
  --project-id=367157a01d98 \
  --qdrant-url=http://qdrant:6333 \
  --query-id=embedding-provider \
  --query-id=impl-remote-embeddings \
  --scenario=current \
  --scenario=off:rerank=false \
  --scenario=weak:rerank=true,weight=0.05 \
  --scenario=pure:w=1
```

**3. Inspect a run** (per-query expected vs returned paths):

```powershell
$r = Get-Content tmp\bench-report.json -Raw | ConvertFrom-Json
foreach ($q in $r.queries) {
  $e = $q.modes.semantic.evaluation
  "$($q.id)  top3=$($e.top3Hit) firstRank=$($e.firstRelevantRank)"
  "  EXP: $($q.expectedFiles -join ' | ')"
  $i=0; foreach ($p in ($e.rawTopPaths | Select-Object -First 5)) { $i++; "   $i. $p" }
}
```

The `--project-id` is the tenant the data is indexed under, **not** the registry
projectId — see the multi-clone tenant note in the project memory if `search`
returns 0.

---

## Levers applied (and their effect)

Baseline → current, `semantic` mode, settled index, 12 queries:

| metric        | baseline | + dedup | + path-boost (α0.8) |
| ------------- | -------- | ------- | ------------------- |
| top-1         | 8.3%     | 8.3%    | 16.7%               |
| top-3         | 33.3%    | 33.3%   | 50.0%               |
| top-10        | 41.7%    | 75.0%   | 83.3%               |
| recall@10     | 20.8%    | 50.0%   | 70.8%               |
| MRR           | 0.22     | 0.25    | 0.39                |
| duplicateRate | 24.2%    | 0%      | 0%                  |

Verdict: `poor` → `mixed` (only `top3 < 80%` remains).

1. **Same-file dedup** (`search-helpers.ts` `dedupeByFile`) — collapse multiple
   chunks of one file to its best-scored chunk before slicing. Biggest recall
   win; a query had previously returned `watchdog.rs` four times in the top 6.
2. **Wider candidate pool** (`limit*2 → limit*5`) — so a precisely-named file
   buried by raw cosine is in scope for the re-rank. (This is Phase 2 "overfetch"
   of the rerank plan, made concrete.)
3. **Path-relevance re-rank** (`applyPathRelevanceBoost`, `PATH_BOOST_ALPHA=0.8`)
   — multiplicatively boost results whose file path/symbol contains the query's
   content words. Filename is a high-precision signal orthogonal to the
   dense/sparse legs. **`ALPHA` is tuned on 12 queries — treat as a starting
   point, not a validated optimum.**

---

## Lessons learned

1. **De-pollute the index before trusting results.** The repo was indexing its
   own eval artifacts (`reports/semantic-search.json`, the benchmark dataset),
   which contain the queries + expected paths verbatim and dominated results
   (data leakage). Now excluded via the root `.wqmignore`. Any generated
   report/fixture that echoes queries must stay out of the index.
2. **Measure only on a settled index.** A daemon restart re-enqueues a near-full
   re-scan; mid-reindex numbers are noisy and understate recall.
3. **Dedup by file is the cheapest large win.** Chunk-level results let one big
   or repetitive file eat several top-k slots. Collapsing to best-chunk-per-file
   roughly doubled top-10 and recall here.
4. **`hybrid` can underperform pure `semantic` on NL queries.** The sparse/BM25
   leg amplifies content-term "magnets" — a 2 357-line `default_configuration.yaml`
   and the `keyword_extraction/*` subsystem (semantically adjacent to
   "search/rank" but the wrong subsystem). Don't assume hybrid ≥ semantic.
5. **`exact` mode is 0% on these queries by design.** FTS5 substring can't match
   natural-language questions; use `exact=true`/`grep` only for known
   identifiers/strings, `semantic` for concepts.
6. **precision@10 is not a meaningful gate for known-item eval.** With 1–2
   relevant files per query it caps at ~`|expected|/10 ≈ 0.19` and is ≈
   `recall@10 × 0.19` — a rescaled copy of recall, no independent signal. The
   old `0.84` gate was unreachable. It is now reported but not gated.
7. **The filename is underused signal.** Several misses had the answer file's
   name literally containing the query terms (`search-score-threshold.test.ts`
   for a "score threshold" query) yet ranked ~9th by cosine. The path-boost
   exploits this; a future cross-encoder reranker would do it more generally.
8. **Beware overfitting to 12 queries.** Tune conservatively, prefer signals
   that are generally sound (filename relevance, dedup) over benchmark-specific
   hacks, and grow the dataset before trusting fine gains.

---

## Reaching the remaining gate (`top3 ≥ 80%`)

Cheap levers (dedup, overfetch, path-boost) took recall@10 past its gate but
top-3 ranking is still embedding-limited (all-MiniLM-L6-v2 conflates adjacent
subsystems). The designed path to higher top-3 is the **second-stage reranker**
already specced in [the search-quality plan](../plans/2026-05-25-search-quality-next-steps.md)
(Phase 3: pluggable `RerankProvider`, local `BAAI/bge-reranker-base` via
fastembed or remote `openai_compatible`, `fallback_to_rrf`). Grow the dataset
(>30 queries across subsystems) before tuning the reranker to avoid overfitting.
