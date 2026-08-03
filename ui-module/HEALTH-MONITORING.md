# Health monitoring — a brief for the status surfaces

Written 2026-07-30, before the status zone is built, so the shape is chosen rather than inherited.
This is a **brief, not a specification**: it records what was measured in v0.1, what exists in the
rebuild today, and which properties the surface should hold out for. The binding decision is
`CR-035` (owner `N49`); if the two ever disagree, `CR-035` wins.

The short version: **a status read should not be able to fail, and should not cost a probe.**

---

## 1. What v0.1 does, measured

Chris noticed the v0.1 TUI log view filling with `Successfully connected to Qdrant server` about
once a second, most visibly when nothing else was happening. Measuring it (22.2 h of
`daemon.jsonl`, 83,665 lines) found this:

```
'Successfully connected to Qdrant server'   894 lines   (1.1% of lines)
mean over the whole window                  0.01 /s     <- misleading
last 5 min while the TUI was open           0.71 /s
```

The per-minute rate steps from ~0 to a flat ~42/min at the moment the TUI is opened, and holds.
Inter-arrival gaps over the last 200 lines:

```
~0.03-0.06 s   99 occurrences   <- a PAIR of probes ~50 ms apart
~3.03-3.07 s   82 occurrences   <- the poll interval
```

A pair of live Qdrant round-trips every ~3 s, for as long as someone is looking.

**Why it happens.** Each `SystemService.health` RPC transitively performs an uncached, live
reachability probe:

```
SystemService.health (gRPC)
  -> get_queue_processor_health()
    -> probe_qdrant_reachable()        <- no cache
      -> StorageClient::test_connection()
        -> Qdrant::health_check()      <- a real round-trip
```

Its sibling probe in the same handler *is* cached — the code comment reads *"embedding provider
health (probe with 3s timeout, TTL cache)"*. One RPC, two probes, one cached and one not.

**Two lessons, and the second is the one that matters.**

1. **The cost is small.** The client is a reused `Arc<Qdrant>`, so there is no reconnect per call,
   the server is local, and the log rotates. At 0.7/s nothing is meaningfully consumed. Health
   monitoring did not make v0.1 slow.
2. **The observer causes the observed.** Health is computed *by whoever asks*, so its cost scales
   with how many surfaces are watching and how fast. In the busiest measured minute the log carried
   42 health lines against 8 lines of real work — the daemon was indexing a file at the time. A log
   that is 84% one unconditional success is not a log anyone can read. That is a UX defect produced
   entirely by the monitoring design.

A contributing detail worth not repeating: the routine **success** is logged at `info!` while the
**attempt** above it is `debug!`. The levels are inverted — the loud line is the one carrying no
information.

---

## 2. What exists in the rebuild today

Less than the plan assumes. Both halves of "watch the value instead of polling" are missing.

**`wqm-client` holds no state.** The `Client` struct carries an `Address` and nothing else, and the
doc states this as a deliberate property:

> Holds no connection: each call dials, so that a daemon started or stopped between calls is
> observed rather than cached.

**The wire has one unary RPC.** `system.proto` declares exactly:

```protobuf
service SystemService {
  rpc Status(StatusRequest) returns (StatusResponse);
}
```

No stream. So there is currently **no maintained value to hook onto and no change signal to
subscribe to** — those have to be built before a TUI can watch anything.

What *does* already exist and is worth leaning on: `DaemonReport` models unreachability as **data**,
not as an error. Its `Unreachable` variant carries a reason and the address tried. The status zone
should render those variants rather than invent a parallel set — and a dropped subscription is
expressible in the same type instead of as a broken stream.

---

## 3. The trap: a watch does not fix this on its own

The instinct "use a watch instead of polling" is right about the surface and insufficient underneath.

A watch changes *who initiates*, not *what a read costs*. If the daemon recomputes health per tick,
the probes continue with a different trigger. With N subscribers a naive implementation is **worse**
than polling, because polling at least had one caller.

The fix that actually removes the cost is on the daemon side: **health becomes maintained state.**
A component refreshes each probe on its own cadence (or on transition) and publishes the current
report; readers never trigger a probe. Then:

- a read is cheap and constant, independent of reader count and refresh rate
- a watch pushes on *change*, so an idle system is genuinely silent — exactly the case where the
  v0.1 noise was most visible
- the TUI may poll or subscribe, and neither choice can hurt the daemon

---

## 4. The tension to hold, not trade away

`wqm-client`'s "observed rather than cached" is sound reasoning for the call it describes. `status`
is what an agent invokes *because something seems wrong*, so a stale answer is a bad answer, and the
sealed MCP surface makes the same point from the other side — `status` is deliberately excluded from
`backend_unavailable` because it exists to answer when the backend is down.

A watch is by definition a cached last-known value. So it is **not** an addition to `status()`; it is
a second call with different semantics:

| call | question it answers | freshness | cost |
|---|---|---|---|
| `status()` | "tell me now" | dials, uncached — unchanged | one dial + one cheap read |
| `watch_status()` | "tell me when it changes" | last known + push on change | one shared stream |

`status()` keeps its documented property verbatim. What changes is that the daemon *reads* its
answer from maintained state instead of *computing* it by probing — cheaper without being staler.

---

## 5. Properties to hold out for

For whoever builds the status zone, in rough priority:

1. **Rendering must never trigger work.** If opening a panel changes what the daemon does, the
   design has the v0.1 shape regardless of transport.
2. **Silence in the steady state.** No unconditional log line on a periodic success path, anywhere
   in the chain.
3. **Coalesce on settled transitions.** A flapping probe pushes one event per flap and reproduces
   the volume this is meant to remove. Debounce; push on change of the settled value.
4. **Unreachable is a render state, not an error path.** It has a variant and a reason — show them.
   The launch screen with no daemon must read as an alarm, never as an empty workspace.
5. **Assume the component set grows.** Queue, embedding provider, Qdrant, and more later. Prefer
   one channel carrying the whole report over per-component subscriptions: the report is small, one
   channel coalesces naturally, and it absorbs new components without a wire change each time.

---

## 5a. ⚠️ ADDED 20260803 — `CR-035` reached r04 and the component set CHANGED

**A factual update, and it touches the vocabulary a status zone renders.** `CR-035` §13's seven
capability-keyed entries — `daemon` · `statedb` · `storedb` · `qdrant` · `fts_code` · `fts_grep` ·
`ladybug` — carried a sentence excluding the embedding provider: *"Deliberately NOT entries: … the
embedding provider. Revisit if either proves to need one."*

**Chris exercised that revisit clause on 20260803** (recorded in `CR-035` r04, raised in `CR-057` §9.2):
*"there is the question of component availability, again the embedder is included but also all our
components and stores."* So **the embedder is an availability entry**, and the principle is broader than
the one name — though "all our components and stores" names a principle, not a finished roster, and the
final membership is not fixed.

**The queue's exclusion stands unchanged.** Its gauges (queue count, in-progress, errored entries) are
always-measured *metrics* under `CR-056`; that never conflicted with its not being a health *entry*.

⚠️ **And the embedder introduces a distinction none of the seven had:** *"we should not conflate rate
limit with inaccessibility, rate limit is transient while non accessibility is not and is of indetermined
duration."* A rate-limited provider is working and refusing us **now** — not `down`. A reachability check
that reports the two identically is wrong in both directions.

**Two new measures recorded in `CR-057` §9.3**, general to every entry: the interval from inaccessible to
available, and — for an outage still in progress — `down_since`, from which elapsed time is derived
locally rather than transmitted. That is the same derive-don't-transmit shape `CR-056` §6c uses for
`stale`.

**One consequence for anything that renders an overall state**, from `CR-057` §10.1, and it is Chris's
ruling rather than a suggestion: **a component being `down` does not make the system `down`.** *"The
daemon being down is the only certainty, but if the embedder is down, the daemon is degraded and the
search as well because we can still do FTS5, grep and graph, searches without the embedder."* How
component states compose into a system state is **open** (`CR-057` §12.0/§12.1) — `CR-057` §10.2 proposes
composing through `CR-035`'s existing `kind` column plus a required/contributing capability table, and
that proposal is not decided.

## 6. What is deliberately open

- **Refresh cadence is a tolerance, and Chris's to set.** Do not infer it from v0.1's accidental
  ~3 s, which was a side effect of the TUI's poll loop rather than a chosen number.
- Whether the subscription is a new `WatchStatus` RPC or a streaming variant of `Status`.
- Whether the CLI ever subscribes, or only ever wants the one-shot call. It is one-shot by nature.
- Where the maintained state lives on the daemon side.
- **Whether an entry can be neither healthy nor unhealthy — `degraded` — and on which axis.**
  Deferred by Chris and still open; `CR-035` §15 holds the two candidates, **freshness** and
  **latency**, and the trap that any definition needing a measurement nobody was going to take
  reintroduces the polling the CR removed. **Added 20260801 (`CR-035` r03 §15.1), because it now
  bears directly on something built here:** your `health::SETTLE_AFTER_RECOVERY` window exists to
  withhold component alarms for a guessed duration after the daemon returns, and it is guessed only
  because a client cannot ask how *old* a reading is. Under the CR's own semantics — green means
  "the last real use succeeded, at T", an observation record and not a liveness claim about this
  instant — a component untouched since a restart has a **stale** reading rather than an unhealthy
  one. If freshness is modelled daemon-side, that window is deleted rather than tuned. So treat the
  5 s as provisional in the strong sense: not "a number awaiting Chris", but a field that may not
  survive the axis being chosen. Nothing here decides it.

Until those are settled, the status zone can be built against `DaemonReport` as it stands: a
one-shot `status()` render path is correct today, and gains a subscription later without the widget
changing shape — which is the point of holding the seam steady.

---

## Maintenance

This brief is **registered and actively maintained**, not a one-off drop. It is listed in the brief
register in the repo-root `handover.md`, and every change-set to the main corpus walks that register
before sealing: if the change touched one of this brief's named sources — `CR-035`, `MCP-SURFACE.md`
§4.3, `wqm-client`'s dial behaviour, `system.proto`'s `Status` RPC, or the `DaemonReport` variants —
this file is updated in the same change-set. The rule is `BEHAVIORS.md` §3.1 (standing, Chris
20260730).

So it is safe to build against. What is **not** safe is to treat it as the authority: `CR-035` is,
and if the two ever disagree the brief is the defect. If you find a disagreement, say so rather than
working around it — that is the signal the maintenance loop missed a hop.
