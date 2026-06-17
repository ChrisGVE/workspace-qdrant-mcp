# Semantic search quality benchmarking

`wqm benchmark search-quality` measures whether the search pipeline returns the
*right* files for a set of known-item questions. It is distinct from
`wqm benchmark search`, which measures **latency** (FTS5 search-DB vs ripgrep).
This eval measures **quality**: for each query whose answer file is known ahead
of time, does that file appear among the top-ranked results?

It runs the same live pipeline a real `wqm search` runs — semantic, hybrid, and
exact modes through `wqm_client::search` — so the numbers reflect the actual
index, not a reimplemented searcher.

## Running it

Run from inside a registered project (the eval is project-scoped):

```bash
# Bundled gold set, human-readable report to stdout
wqm benchmark search-quality

# Custom gold set + JSON report file
wqm benchmark search-quality --dataset path/to/gold.yaml --output report.json

# Score only the top 5 ranked results per query (default: 10)
wqm benchmark search-quality --top-k 5
```

Requirements: the daemon (`memexd`) must be running and the current project must
be indexed. If the project is not registered, the eval prints how to register it
and exits without faking results.

## What it reports

Per ranked mode (`semantic`, `hybrid`, `exact`):

| Metric | Meaning |
|--------|---------|
| `top1` / `top3` / `top10` | Hit rate: an expected file appears in the first N **raw** ranked results |
| `recall@10` | Fraction of *distinct* expected files surfaced in the deduplicated top-k |
| `precision@10` | Fraction of the deduplicated top-k that are expected files |
| `mrr` | Mean reciprocal rank of the first relevant result |
| `duplicateRate` | Share of the ranked list that was repeat paths (`1 − deduped/raw`) |
| `avgLatencyMs` | Mean wall latency per query in that mode |

Hit rates are measured on the **raw** (duplicate-bearing) ranked list because a
reader scans results in returned order. Recall and precision use the
**deduplicated** list so a file repeated across chunks counts once.

The report also includes:

- **`byCategory`** — `top1`/`top3`/`top10` hit counts per category for the
  semantic and hybrid modes, so a weak category is visible instead of silently
  dragging the aggregate.
- **`perQuery`** — id, query, expected files, and per-mode `top1`/`top3`/`top10`
  plus `firstRelevantRank`.
- **`verdict`** — a grade for the semantic mode (see below).

## The verdict

The semantic mode is graded against two **independent** known-item signals:

| Gate | Default threshold | Question it answers |
|------|-------------------|---------------------|
| `top3UsefulRate` | ≥ 80% | Is a relevant file ranked high enough to be seen? |
| `recallAt10` | ≥ 70% | Did we surface the relevant file at all? |

Grade: zero failed gates → **good**, one → **mixed**, both → **poor**. Each
failed gate adds a reason naming the metric, its value, and the bar it missed.

`precision@10` is intentionally **not** a gate. With only 1–2 relevant files per
query it is approximately `recall@10 × (meanExpected/10)` — a rescaled copy of
recall — so gating on both would double-count one signal. It stays in the
reported metrics for visibility. Thresholds live in
`metrics.rs::QualityThresholds`.

## The gold dataset

The gold set is `src/rust/cli/src/commands/benchmark/quality/benchmark-data/semantic-search-quality.yaml`,
compiled into the binary so the eval runs without a bind-mounted repo. Each entry
is a query with the files a good search should surface:

```yaml
- id: impl-bm25-tokenizer
  query: Where is the BM25 tokenizer that splits and filters terms for sparse vectors?
  expectedFiles:
    - src/rust/daemon/core/src/embedding/bm25.rs
```

`expectedFiles` may be literal repo-relative paths or globs
(`**/proto/*.proto`, `.../exact_search/*.rs`). A query is a hit when an expected
file (or glob) matches a top-k result path.

**Zero-match glob contract.** A glob in `expectedFiles` that matches *no* result
path in the ranked list scores as a **MISS** — not a skip, not neutral. There is
no special handling for unmatched globs: `path_match.rs` evaluates each
`ExpectedMatcher` against the result paths and, if none return `true`, the
expected file simply does not appear in the hit set. This means a stale gold glob
that no longer matches any real path silently penalizes the score for that query.
Verified against `path_match.rs`: `ExpectedMatcher::matches` is a pure predicate;
the scoring layer in `metrics.rs` treats any expectation not satisfied by at least
one result as unfulfilled.

### Categories

Each query's category is its id prefix before the first `-`:

| Prefix | Category | Purpose |
|--------|----------|---------|
| (none / unknown) | `orig` | general known-item lookups |
| `sym-` | symbol | identifier lookups (exercise the BM25/sparse leg) |
| `impl-` | implementation | "where is X implemented" (code, not docs/tests) |
| `doc-` | documentation | ADR / spec lookups |
| `real-` | real-agent | terse, keyword-shaped phrasing agents actually use |
| `pt-` | Portuguese | **expected-weak** multilingual-gap probe |

The `pt-` set is included only as a visible probe of the multilingual gap: the
embedding model is English-only, so Portuguese queries mostly miss. Their low hit
rate is **expected** and is reported per-category so it never reads as a
regression. Grow or drop the set when a multilingual model lands.

Every `expectedFile` in the bundled set was verified to exist on disk at
authoring time (2026-06-17). When the tree moves, update the gold paths in the
same change-set — a stale gold path silently scores a correct answer as a miss.

## Mining real agent queries

The `real-` category should grow from phrasing agents actually use. The eval tags
**its own** search traffic with `actor = 'benchmark'` (admitted by schema
migration v47) precisely so that organic queries can be mined without the eval's
own traffic polluting the results:

```sql
SELECT query_text, COUNT(*)
FROM search_events
WHERE project_id = ?
  AND op = 'search'
  AND actor != 'benchmark'
GROUP BY query_text
ORDER BY MAX(ts) DESC;
```

Take the recurring, terse phrasings, find the file each one *should* return, and
add them as `real-` entries.

## How it fits together

```
wqm benchmark search-quality
        │
        ▼
quality/mod.rs ── connects daemon + Qdrant once (LiveRunner)
        │
        ├── dataset.rs   load + categorize the gold YAML
        ├── per query × {semantic, hybrid, exact}:
        │       run_search_pipeline / search_exact   (the LIVE pipeline)
        │       └── log_search_event(actor='benchmark')   (fire-and-forget)
        ├── path_match.rs  normalize result paths, match expectedFiles globs
        ├── metrics.rs     hit@k / recall / precision / MRR / dup-rate / verdict
        └── report.rs      shape → terminal + JSON
```

The metric math, categorization, glob matching, dataset loading, and verdict are
all pure functions, unit-tested on synthetic data (no daemon needed). The live
end-to-end run needs the daemon and a real index.
