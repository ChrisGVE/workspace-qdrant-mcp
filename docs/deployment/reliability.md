# Reliability notes (reference compose stack)

This document captures what we've verified about the reference stack's
behaviour under common failure modes. It is not a soak-test report — for
the actual CI coverage see `tests/integration/docker/test-chaos.sh` and
the `docker-integration` workflow in `.github/workflows/`.

## What the chaos test covers

`tests/integration/docker/test-chaos.sh` runs the following scenarios
against `docker/compose/reference.yml`:

1. **Qdrant restart × 3.** `docker compose restart qdrant` — wait for all
   four health probes, then assert the MCP `initialize` RPC still
   produces `serverInfo`.
2. **memexd restart × 3.** Same shape; MCP `/healthz` must stay up during
   the window where the daemon is cycling.
3. **MCP restart × 3.** New `initialize` calls must succeed once the
   container finishes booting.
4. **SIGTERM to memexd.** Container must exit with code 0 within 15s
   (well under the compose stop timeout). Verifies the daemon's cleanup
   path (flush SQLite + close gRPC listeners).

`CHAOS_ITERATIONS` overrides the restart count per scenario.

## Known failure modes & recovery times

Numbers below are typical on a laptop-class host (ubuntu-latest runner,
local Docker Desktop). They are qualitative guides, not SLAs.

| Event | Blast radius | Recovery time | Notes |
|-------|--------------|---------------|-------|
| Qdrant restart | Search and retrieve return transport errors during the outage; store operations are queued by memexd and replay on recovery. | ≤5s (pulled from healthcheck) | memexd's queue writer writes to SQLite first, so no points are lost. |
| memexd restart | MCP `/healthz` stays up. Tool calls that hit the daemon return gRPC `UNAVAILABLE` and the MCP server falls back to SQLite read paths where available (see `wqm_mcp_daemon_fallback_total`). | ≤10s | Sessions initialized through MCP survive; `/mcp` POSTs made mid-restart may receive a 500 that clients must retry. |
| MCP restart | All in-flight sessions are lost. Clients must re-`initialize`. | ≤5s | The MCP Streamable HTTP transport is stateful in memory; restart is equivalent to session rotation. |
| Qdrant SIGTERM | Volume bind-mount keeps storage intact. On cold boot Qdrant replays its WAL. | ≤10s | Verified manually; not in CI because the cold-boot time is too variable. |
| memexd SIGTERM | Drains the queue, flushes SQLite, returns exit 0 within ~2s under typical load. CI asserts ≤15s. | — | The compose `stop_grace_period` is 10s by default; longer-running drains require raising it. |
| MCP SIGTERM | Process flushes pending OTLP batches then exits 0. | — | Handled in `src/rust/daemon/mcp-server/src/observability/`. |

## What the chaos test does **not** cover

- **Concurrent-load failure injection.** The PRD's original step 5 (50
  concurrent MCP HTTP clients with mid-flight restarts) is currently
  out of scope for CI. Running it locally is straightforward — see the
  `load/` sibling folder if we add one — but not justified as a
  per-PR gate given CI runtime cost.
- **Network partition scenarios.** Docker compose's single-host bridge
  makes simulating inter-service partitions awkward; the nearest
  approximation (`docker network disconnect`) is left for manual
  exploration.
- **Long-running soak.** We have no data on degradation over 24h+. The
  daemon's Prometheus metrics (`wqm_queue_depth`, `wqm_mcp_session_count`)
  are the right signal if you want to grow one.

## Operational recommendations

- **Qdrant health check.** Keep it at `interval: 15s` in compose; the
  current value is tuned so chaos tests recover quickly without making
  the healthcheck noisy in normal operation.
- **memexd grace period.** If the daemon is under heavy write load,
  bump `stop_grace_period` on the memexd service to 30s so `compose
  down` does not SIGKILL it mid-drain. The default 10s is fine for
  interactive workloads.
- **MCP reverse-proxy.** Rate limiting is per-process. In multi-node
  deployments terminate the limit at the reverse proxy (Caddy/Traefik)
  — see `docker/compose/reference.tls.yml`.

## Running the chaos test locally

```bash
# Requires a recent release build of wqm on PATH.
cargo build --release --manifest-path src/rust/Cargo.toml --package wqm-cli
export WQM_BIN="$(pwd)/src/rust/target/release/wqm"

export MEMEXD_IMAGE=ghcr.io/chrisgve/memexd:latest
export MCP_IMAGE=ghcr.io/chrisgve/workspace-qdrant-mcp:latest

# 3 iterations per scenario (default). Bump if you're reproducing a flake.
CHAOS_ITERATIONS=10 tests/integration/docker/test-chaos.sh
```

Scratch directories under `/tmp/wqm-e2e-chaos.*` are preserved on
failure; `compose logs` is dumped to `logs/compose.log` in that dir.


## Cross-process single-instance lock

memexd binds a TCP listener to `127.0.0.1:7799` at startup as the
authoritative cross-process single-instance lock (spec 16 §10.1). Only
one daemon — host or docker, arbitrated by the host kernel when the
container publishes the port to `127.0.0.1:7799` — can hold the bind at
a time. Process death releases the socket immediately; no stale-lock
cleanup logic is needed.

### Port-resolution precedence

1. `--control-port <PORT>` CLI flag — highest.
2. `WQM_CONTROL_PORT` env var (compose-generated overrides consume the
   same name).
3. `DaemonConfig.control_port` field in `config.yaml`.
4. Built-in default `7799`.

### Identity stamp

On a successful bind, memexd writes a diagnostic JSON to
`~/.local/share/workspace-qdrant/memexd.lock` (host) or
`/var/lib/wqm/memexd.lock` (docker). Schema:

```json
{
  "mode": "host",
  "pid": 12345,
  "started_at": "2026-05-14T10:30:00+00:00",
  "port": 7799
}
```

The stamp is **diagnostic only** — the bound socket is the authoritative
lock. Stamp-write failure logs a warning but does not block startup.

### Troubleshooting "port in use" errors

If memexd refuses to start with a control-port bind error:

1. Check for a running memexd: `ps -ef | grep memexd`.
2. Check for an unrelated process holding the port:
   `lsof -nP -iTCP:7799 -sTCP:LISTEN`.
3. If running parallel test instances, use `--control-port` to bind
   different ports per instance.

### Deployment-mode detection

memexd detects whether it runs on the host or inside a docker container
via, in order:

1. `WQM_DEPLOYMENT_MODE=host|docker` — explicit override.
2. Presence of `/.dockerenv` — Docker's standard marker file.
3. Default: `host`.

The detected mode is recorded in the identity stamp and selects the
on-disk location (`~/.local/share/workspace-qdrant/` vs `/var/lib/wqm/`).

