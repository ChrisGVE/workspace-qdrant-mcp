# Telemetry — what exists, what is decided, and what is open

**Requested by Chris on behalf of `ui-module`, 20260802.** Two questions were asked: *what is the
telemetry surface and how does a client tap into it*, and *do we keep aggregate metrics over a
rolling period of time*.

**What this document is.** Information only: measured facts, what the code contains today, what the
sealed contracts say, and decisions already taken upstream — each with its authority named. It
contains **no recommendations** and nothing that proposes what `ui-module` should build. Where the
corpus has not decided something, this says so and names who owns the decision, rather than
supplying an answer. (Standing rule, Chris 20260802.)

**Authority.** The binding texts are `P02-foundation/GT005-contracts/CONTRACTS.md` §N13/§N31
(sealed) and `CR-006`. If this brief and any of them disagree, they win and this file is the defect.

---

## 1. The short answer on rolling aggregates

**No rolling aggregate exists, and the sealed contract currently commits N13 to the opposite.**

N13's *Guarantees*, verbatim (`CONTRACTS.md`:1540):

> every signal is current-state (a gauge, not a monotonic counter)

That sentence is the FW-10 rule ("a current signal, not a lifetime counter") applied to the whole
observability surface. Under it, a reader receives *the value now*, not a window, a rate, a
percentile, or a history.

**Whether a series may exist alongside the gauges is an open question with a named owner, not a
settled "no".** `CR-006` §3 records it as an observation found while verifying the contract:

> N13's *Guarantees* commit it to gauges, not series … `CR-006` asks for "a real resource series". A
> gauge sampled over time is a series, so these can be reconciled, but **the reconciliation is not
> written anywhere** and the owning engagement should do it deliberately rather than by accident.

and `CR-006` §5.4 lists among what is NOT decided:

> **The resource series' cardinality and retention** — and its reconciliation with N13's gauge-only
> guarantee.

So the three sub-questions inside "do we keep rolling aggregates" — *is there a series at all*,
*at what cardinality*, and *for how long* — are all open, all on `CR-006`, owner **N13**, acceptance
consumer **`P04-GT996`** (`e2e-perf-gates`). `CR-006`'s status in `program_cr` is `carried`.

A separate search of the sealed corpus for retention or windowing language over metrics returns
nothing: every `retention` hit in `CONTRACTS.md` and `ARCHITECTURE.md` belongs to N52's taxonomy
round-GC (`retention = current + previous`), which is unrelated to telemetry.

---

## 2. The sealed N13 contract, verbatim

`CONTRACTS.md`:1533-1543, `### N13 -- Observability  [K-H | M | Observer / metrics-sink Strategy | consolidate]`:

> - **Owns:** metrics / health / queue-depth / tracing surfaced as a **current** signal (not a
>   lifetime counter, FW-10). The kernel produces the signal; CLI/TUI status views render it.
> - **Provides:** `emit(metric)` (kernel side); `snapshot() -> HealthSignal` (current queue depth,
>   DLQ size, embed latency, daemon health -- the render side); the RAG verdict inputs. The metric
>   sink is a seam.
> - **Requires:** N8 (metric names), N7 (thresholds).
> - **Guarantees:** every signal is current-state (a gauge, not a monotonic counter); N13 reports
>   **daemon-level** health (not per-query leg state -- the per-query `missing_legs` is N38's from
>   `N41.available()`, A-NIT2). **I7 (F16): the metric sink scrubs secret-shaped values** -- a
>   resolved credential never appears in a metric. UNIFORM.
> - **Build path:** consolidate -- metrics/health exist per-binary; converge the sink; distinguish
>   from N31.

Four facts follow from that text directly:

- **The surface is two-sided.** `emit(metric)` is the kernel side; `snapshot() -> HealthSignal` is
  named *"the render side"* — the one a status view reads.
- **Metric names are N8's, thresholds are N7's** (*Requires*). A metric name is therefore a guarded
  name in the same sense as a collection name, and any threshold that decides a metric's
  presentation is a config knob, not a constant.
- **N13 is daemon-level.** Per-query leg state is explicitly *not* N13's: `missing_legs` is N38's,
  derived from `N41.available()` (A-NIT2).
- **The sink scrubs secrets (I7/F16), UNIFORM** — a resolved credential never appears in a metric.

**N13 is not N31.** `CONTRACTS.md`:1545-1554 gives logging/tracing to N31: it owns `daemon.jsonl`,
`WQM_LOG_LEVEL`, correlation ids, the `tracing` subscriber setup and `correlation_id()` propagation.
N13's own build path says *"distinguish from N31"*. A latency visible in the log stream and a latency
visible as a metric arrive through different owners.

**One field in the quoted contract is already superseded.** `HealthSignal` names **DLQ size**, and
the DLQ is retired for v0.2 by `CR-010` / `ADR-004`. `CR-006` §3 states that the two CRs
*"touch the identical sealed sentence and must be answered coherently; neither document previously
said so"*; what replaces that field is listed in `CR-006` §5.5 as undecided.

---

## 3. What exists in `dev` today

Measured 20260802 against the working tree.

**The wire carries one RPC.** `src/rust/crates/wqm-proto/proto/wqm/v1/system.proto` declares exactly:

```protobuf
service SystemService {
  rpc Status(StatusRequest) returns (StatusResponse);
}
```

`system.proto` is the only `.proto` file in that directory. **There is no metrics RPC, no stats RPC
and no stream.**

**`StatusResponse` carries four numbers, and they are index state, not telemetry:**

```protobuf
message StatusResponse {
  DaemonHealth daemon = 1;
  optional IndexState index = 2;
}
message DaemonHealth { State state; string detail; int64 since_unix_seconds; string version; }
message IndexState  { uint64 files_tracked; uint64 queue_pending; bool complete; uint64 lag_seconds; }
```

`IndexState` is `optional` and the file's own comment states the convention: fields are *"ABSENT
rather than zero-filled: proto3 message fields carry presence"*. So absence and zero are
distinguishable on this message.

**The client exposes one leg.** `src/rust/crates/wqm-client/src/lib.rs` has two public functions,
`Client::new(address)` and `async fn status(&self, probe: bool) -> Result<DaemonReport, ClientError>`.
The crate is 248 lines and holds no state; its doc records this as deliberate — *"Holds no
connection: each call dials, so that a daemon started or stopped between calls is observed rather
than cached."*

**So the tap that exists today is `status()`**, and what it returns is daemon health plus index
state. Nothing in `dev` emits, aggregates, stores or serves a metric.

---

## 4. What the v0.1 system records — baseline, not the rebuild

⚠️ **These are facts about the system being replaced.** They are included because `CR-006` requires
the v0.1 corpus be carried forward *in shape*, so they describe what the rebuild is measured
against — not what it provides. (An observation of the deployed system is not a fact about the
rebuild.)

From the sealed behavioral census (`BEHAVIORAL-CENSUS.md`, quoted through `CR-006` §1):

- **`state.db.search_events`** — end-to-end MCP search latency, recorded since 2026-05-13.
  **1,430 rows, p50 202 ms, p99 5,008 ms.** It *"times the whole call and nothing inside it"*.
- **`processing_timings`** — write-path only. One row per (queue item, op, phase) with `duration_ms`
  dimensioned by `tenant_id` / `collection` / `language` / `file_type` / `embedding_engine`
  (:1149-1154). **414,456 of 1,145,139 rows carry no `embedding_engine`** (:1156-1158) — recorded in
  the census as the defect worth not repeating.
- **The read path is otherwise unmeasured.** Verbatim (§1.9, :449-454): *"nothing times the stages
  inside it (query embedding, dense, sparse, FTS5, RRF fusion, grep), and the CLI read path records
  nothing at all. `processing_timings` remains entirely write-path."*
- **Telemetry writes are fire-and-forget and lose events silently.** `LogSearchEvent` always returns
  `Ok(())`; the census (§6.3, :1139-1145) carries the pattern forward *"plus a drop counter"*, whose
  semantics `CR-006` §5.3 lists as unspecified.

The stage list `CR-006` requires be decomposed is N38's own *Owns* line (`CONTRACTS.md`:1022):
`query -> embed -> (dense + sparse + FTS5 + grep) -> fuse -> envelope`.

---

## 5. What is undecided, and who owns each

All five are `CR-006` §5, owner **N13**, acceptance consumer **`P04-GT996`**:

| # | Open item |
|---|---|
| 1 | **Where a stage timing is emitted from** — N13 owns the sink, N38 owns the pipeline; whether stages emit through `emit(metric)` or through N31 tracing spans is unstated, and the two have different costs |
| 2 | **How the write-path corpus is carried forward "in shape"** — comparability with `processing_timings` is required for the `P04-GT996` beat-legacy gate, but the schema is not fixed |
| 3 | **The drop counter's contract** — semantics unspecified |
| 4 | **The resource series' cardinality and retention**, and its reconciliation with the gauge-only guarantee — *the rolling-aggregate question* |
| 5 | **What replaces "DLQ size"** in `HealthSignal` — owned by `CR-010` / `ADR-004`, lands on N13's surface |

`CR-006` §6 records that it is *"a build obligation with a named acceptance gate, not a decision
awaiting a ruling"*, and closes when `P04-GT996` accepts the instrumentation — except for items 1
and 4, which `CR-006` §6 says are decisions and would be recorded as an ADR the CR then cites.

**Routing, per the standing protocol:** a request about this surface is raised as a `UIQ` in
`TO-CORPUS.md` and becomes a CR, which is a proposal decided upstream on its merits (Chris,
20260802).

---

## 6. Adjacent and deliberately separate: health is not telemetry

`CR-035` (r03, `proposed`, owner **N48**) defines the *health* surface, and it is a different thing
from a metric on two counts recorded in that CR:

- **Its semantics are an observation record.** Green means *"the last real use succeeded, at T"* —
  not a liveness claim about this instant. Status is a **byproduct of real access**, contributed by
  whichever actor touched a resource; the only poll is a 24 h floor for a resource nothing has
  touched.
- **Its seven entries are keyed by capability, not connection**: `daemon` · `statedb` · `storedb` ·
  `qdrant` · `fts_code` · `fts_grep` · `ladybug`. Queue and embedding provider are deliberately
  excluded. §13.1 of that CR makes these entries the authoritative component vocabulary.

`CR-035` §15 carries one open item, **`degraded`** — deferred by Chris — with two candidate axes,
**freshness** and **latency**, and the constraint that any definition requiring a measurement nobody
was going to take reintroduces the polling the CR removed.

The overlap worth knowing: `HealthSignal` (N13) and the `CR-035` entries both describe daemon
condition, and the N13 contract line naming DLQ size predates `CR-035`. How the two surfaces relate
is not written down in either document.

---

## Maintenance

Registered in the BRIEF REGISTER in the repo-root `handover.md`, and walked before every corpus
seal: if a change-set touches this brief's named sources — `CONTRACTS.md` §N13 or §N31, `CR-006`,
`CR-010`/`ADR-004`, `CR-035`, `system.proto`, or `wqm-client` — this file is updated in the same
change-set (standing, Chris 20260730).

**Binding authority is `CONTRACTS.md` §N13 and `CR-006`.** A brief is a brief, not a specification;
if this file and its authority ever disagree, this file is the defect — say so rather than working
around it.
