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

> ### ⚠️ UPDATED 20260802 (later the same day) — this section's headline is SUPERSEDED
>
> **Chris ruled that the aggregate IS a bounded rolling window** — count, average, median, min and max
> over one uniform window `W`, recorded as **`CR-056`** (owner `N13`, `proposed`). So the answer to
> *"do we keep aggregate metrics over a rolling period of time"* is now **yes**.
>
> **And the second half of the old headline was wrong, not merely outdated.** The sealed contract does
> not commit N13 to the opposite. `CONTRACTS.md`:1540 commits it to **current-state**, and
> `CR-056` §5a records the reconciliation that `CR-006` §3 asked for and nobody had written:
>
> > A bounded rolling aggregate is still current-state, **because it forgets**. FW-10's target is the
> > monotonic lifetime counter — the number that only grows and whose value depends on when the process
> > started. A count/average/median/min/max over the last `W` has neither property.
>
> **N13's guarantee is therefore unchanged and unweakened**, and `CR-006` §3's open reconciliation is
> answered rather than traded away.
>
> Other upstream decisions of the same session, all in `CR-056`, all `proposed`: OpenTelemetry is in
> scope but **off by default**; measurement has **no cadence** — a pack of measures is emitted when an
> operation completes; a metric's status expires from **current** into **last known** when the window
> empties; a metric carries **at most one threshold**, a floor or a ceiling, never both; and an alert is
> a three-state flag (up / down / stale) of which **only the up↔down transitions are transmitted**,
> because stale is derivable by the receiver from the timestamp and `W`.
>
> `W`'s value, every threshold value, and the debounce are **tolerances and remain Chris's** — none is
> fixed in `CR-056`.
>
> The rest of this section is kept as written, because it is the state the ruling changed.

> ### ⚠️ UPDATED AGAIN 20260802 (late) — `CR-056` r02 and r03, and **one statement in the block above is now FALSE**
>
> Two further revisions landed the same evening. They change the mechanism rather than the answer: the
> aggregate is still a bounded rolling window and N13's guarantee is still unweakened.
>
> **The correction first, because it is a statement of fact that has changed.** The block above says
> *"a metric carries **at most one threshold**, a floor or a ceiling, never both"*. That was r01. Under
> **`CR-056` §6a (r02)** a metric carries **one OR two** thresholds, and the invariant is that they are
> **all on the same side** — a metric never mixes a floor with a ceiling. One threshold gives
> `green / red`; two give `green / amber / red`. The values are ordered **by severity, increasing away
> from green** (so ascending for a ceiling, descending for a floor — stated once about severity rather
> than twice about inequalities). A metric with no threshold carries `None`.
>
> **Having a threshold does not make a metric an alarm** (`CR-056` §6d, r02). Alarm-ness is *declared*,
> and it is what buys proactive evaluation. Three tiers are recorded there:
>
> | tier | five statistics | status | evaluated | broadcasts |
> |---|---|---|---|---|
> | plain metric | yes | none | — | no |
> | thresholded, not alarm | yes | yes | **lazily, at read** | no |
> | alarm metric | yes | yes | **proactively** | **up↔down transitions** |
>
> **An alarm metric has exactly ONE threshold** (`CR-056` §6e, r03) — `alarm ⇒ exactly one threshold`,
> so an alarm has no amber, and the three-state flag maps onto a single threshold with nothing left
> over. An alarm declared with two is a registration error.
>
> **Deactivation is encoded by ABSENCE only** (`CR-056` §6f, r03). An alarm with no threshold is off. A
> `0` threshold as a deactivation sentinel was proposed and **withdrawn by Chris** — it duplicates
> absence, and it burns the most useful floor value there is (*throughput fell to zero*).
>
> **Metrics are deactivatable, in two classes** (`CR-056` §6g, r03): **explicit / always-present** —
> queue size, memory, storage — which are **not** deactivatable and exist unconditionally; and
> **on-demand** — stage timings, payload sizes, everything else — which are. Deactivation is a config
> fact in `[observability.telemetry]`, because the cost that matters is the *measurement* and it is paid
> in the emitting process.
>
> **Bidirectional metrics are excluded** (`CR-056` §6h, r03), checked against the metrics named so far
> rather than assumed, with a stated re-entry test: a `Band { floor, ceiling }` variant would extend the
> type without touching the two that exist.
>
> **Ingest is two-stage and the queue is the outbox** (`CR-056` §5e, r02, amended r03). A broadcast is
> accepted, **enriched by the daemon with the always-present block from its own registers**, and
> appended to a raw queue — no payload parse, no per-metric demux. Draining happens on three triggers
> and there is **no periodic process**: a read, the alarm timer, and the daemon's work queue going
> empty. The stated invariant: *a pack is in the queue if and only if it has not yet been **offered** to
> the historian* — offered, not sent, because export is best-effort and the queue is not a delivery
> guarantee. **Local retention is 2W by construction**; anything older leaves fire-and-forget and is
> dropped. **We are not the historian; OTel is**, and remote retention is the operator's.
>
> **Logging shares the pattern and inverts the retention rule** (`CR-056` §7e, r02): accumulate first,
> process later, never on the path being observed — but **never discard on age; flush on a bound**,
> with durability in `daemon.jsonl` (N31's). The flush bound is not inherited from `W`. No new component
> is required: it is N31's already-committed `tracing` facade configured.
>
> **Still tolerances, and still Chris's:** `W`, every threshold value, the debounce, and the log flush
> bound. Nothing here supplies a number.
>
> Open in `CR-056` §9 and named there: losslessness of the broadcast, the drain scope, whether the
> always-present block's memory is the daemon's or the emitter's, whether that block is also sampled at
> read, the ingest size cap, whether the bus is needed for logs at all, whether `schema_id` derives from
> the topic, the burst schema, and **the metric list itself** — `CR-056` fixes the *shape* of a metric,
> not the set. N8 owns metric names and no registry exists yet.

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
| 4 | **The resource series' cardinality and retention**, and its reconciliation with the gauge-only guarantee — *the rolling-aggregate question*. ⚠️ **Answered in `CR-056` (`proposed`)**: the reconciliation is §5a, and retention is §5e — **2W locally by construction**, with OTel as the historian and remote retention the operator's |
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
was not written down in either document.

⚠️ **UPDATED 20260803 — a relation is now PROPOSED, and it is a proposal, not a ruling.** `CR-057`
(`proposed`, owner **N13**) records a measure-point catalog and, in its §10, proposes that a
component's four-state health composes from both surfaces rather than either: **`down`** from
`CR-035` (the last real access failed — an outcome, not a threshold), **`amber`/`red`** from
`CR-056` (the component's duration statistic over `W` crossing its same-side thresholds), and
**`green`** from both holding. `CR-057` §10 also proposes that `degraded` is then the amber band
rather than a separate axis — explicitly left open in its §12.1 as Chris's.

**Nothing in that composition is decided.** `CR-057` is `proposed` and every threshold value and `W`
remain tolerances.

⚠️ **UPDATED again the same day — `CR-057` r02, and two of r01's statements moved.**

- **The `CR-035` entry list DID change.** Chris exercised its revisit clause: **the embedder is an
  availability entry**, along with "all our components and stores" (`CR-035` r04, `CR-057` §9.2).
  The queue keeps its gauges as metrics and its non-entry status as health — both stand.
- **System health is NOT the worst of its components** (`CR-057` §10.1, Chris's ruling): *"the daemon
  being down is the only certainty, but if the embedder is down, the daemon is degraded and the search
  as well because we can still do FTS5, grep and graph."* How components compose into a system state
  is **open**.
- **A rate limit is not an outage** (`CR-057` §9.3) — transient versus indeterminate — and the two are
  measured separately, on the latency side (`rate_limit_wait`, distinct from embedding time) and on the
  availability side (`outage_duration`, plus `down_since` while it is ongoing).
- **The number of participants is a measure** (`CR-057` §6a): daemon, MCP instances, `wqm` CLI/TUI
  instances, from gRPC — because an aggregate over `W` is uninterpretable without knowing how many
  emitters contributed to it.
- **`E2E − parts` is scoped to request/response operations only** (`CR-057` §3 as amended, §7a). It is
  valid where a definite question/answer boundary exists — `query` is the confirmed case — and is
  withdrawn for pipeline, background and lifecycle operations, which have a duration and a progress but
  no "our own latency" because nothing is waiting on them.

⚠️ **UPDATED again — `CR-057` r03.** Two more, and the second is unsettled in a way worth knowing before
anything renders a threshold.

- **Pressure is a named measure family** (`CR-057` §6b), taken from standard practice (RED / USE / the
  four golden signals) rather than invented: arrival rate, service rate, in-flight, saturation. **Write
  pressure is the differential** (arrivals minus completions), not the queue depth — depth is a lagging
  indicator. It fans out per store and carries a **per-participant** dimension, because one client
  hammering and five clients each asking a little are invisibly identical in any aggregate.
- **Qdrant, RULED** (`CR-057` §9.4): **unreachable is `down`**; **slow is amber/red**. This generalizes
  the embedder rule into a law for every dependency — *slow* and *absent* are different states, reached by
  different measurements. Note that by the kind model a `down` Qdrant still leaves the **system**
  degraded, not down, since search continues via FTS5, grep and graph.
- ⚠️ **Thresholds may not be absolute constants** (`CR-057` §10.3). Chris: many of these measures depend
  on the host and are largely unknown to us, so the thresholds are *"a moving target"*.

⚠️ **UPDATED — `CR-057` r04, and this one changes where a threshold LIVES.**

**RULED by Chris:** *"these thresholds are no longer in the configuration file, they belong to the state
database, at least most of them, there might be a few thresholds that can be set by the user as
'preference' but a small number."*

**A threshold is learned state, not configuration.** It follows from `CR-053`'s own rule rather than
cutting across it: configuration is user-written *because a tool writing on the user's behalf still has
the user's agency behind it*, and an adapted threshold has none — nobody wrote it, the system measured it.
Three layers: a **shipped seed** (measured on our machine, good enough to ship), a **learned value** the
daemon adapts continuously, and a **small pinned set** the user may set as a preference.

⚠️ **The consequence for anything rendering configuration:** most thresholds will not appear in the config
file at all. Only the *"small number"* of pinned preferences will. What that set contains is not yet
decided.

**UPDATE 20260803 — "preference" is now a defined store, and it is not the config file.** When the
ruling above was made, *preference* was an informal word. `CR-059` (20260803, `proposed`) gives it a
home: preferences live in **`~/.config/workspace-qdrant/preferences.json`**, a separate, disposable,
**non**-schema-governed file written only by `wqm`, alongside the user's
`~/.config/workspace-qdrant/config.toml`. So a pinned threshold, if the pinnable set ever names one,
lands there rather than in the config file. **Which thresholds are pinnable remains undecided** and is
Chris's (`CR-057` §12.14). Full model in `CR-059`; the channel note is `FROM-CORPUS.md` `NOTE-016`.

**Two collisions are recorded and are Chris's to resolve, not settled:** `FIRST-PRINCIPLES.md` states that
`statedb` holds **only** the watch register and queues, so a home for learned state needs either a
first-principles amendment or a separate store; and the sealed `N13 Requires N7 (thresholds)` edge narrows
to the pinned set.

**Withdrawn from r03, so it is not carried forward as a concern:** an adaptive threshold does **not**
require storing history. Online estimators are O(1) in memory, so the storage added is a handful of floats
per metric rather than a time series — and the five reported statistics stay exact over the raw window,
unchanged, because a threshold is a control parameter and not a reported measurement.

---

## Maintenance

Registered in the BRIEF REGISTER in the repo-root `handover.md`, and walked before every corpus
seal: if a change-set touches this brief's named sources — `CONTRACTS.md` §N13 or §N31, `CR-006`,
**`CR-056` (any revision)**, `CR-010`/`ADR-004`, `CR-035`, `system.proto`, or `wqm-client` — this file
is updated in the same change-set (standing, Chris 20260730).

**Binding authority is `CONTRACTS.md` §N13 and `CR-006`.** A brief is a brief, not a specification;
if this file and its authority ever disagree, this file is the defect — say so rather than working
around it.
