# Docker unified-stack integration tests

End-to-end tests for the root `docker-compose.yml`. Each script brings up
the stack (Qdrant + memexd + MCP HTTP server, plus observability),
probes a specific
invariant, and tears everything down again — even on failure. Scripts are
idempotent and use a unique compose project name, so several can run in
parallel on the same host.

## Files

| Script | Covers |
|--------|--------|
| `common.sh` | shared bash helpers (polling, assertions, cleanup trap) — sourced by every test |
| `test-stack-startup.sh` | `compose up` + all three services pass their health probes |
| `test-mcp-http.sh` | `/healthz` is open; bearer auth rejects missing + wrong tokens; `initialize` + `tools/list` succeed |
| `test-cli-connection.sh` | host `wqm` CLI talks to the dockerized daemon via the `docker-local` profile |
| `test-path-transparency.sh` | daemon records the host's absolute project path verbatim (WQM_DEV_ROOT bind-mount invariant) |
| `test-restart-persistence.sh` | `compose down` (without `--volumes`) preserves project registration + Qdrant storage |
| `test-chaos.sh` | restart each service × 3 and SIGTERM memexd; stack recovers, exit codes clean. See [reliability notes](../../../docs/deployment/reliability.md) for full coverage + known failure modes. |
| `run-all.sh` | driver that runs every `test-*.sh` sequentially |

## Running locally

```bash
# From repo root. The host `wqm` CLI is the one piece these tests build natively
# (it exercises the host→dockerized-daemon path); everything else builds inside
# Docker. A native wqm build needs a local Rust toolchain + ONNX Runtime
# (ORT_LIB_LOCATION) — see CLAUDE.md. Alternatively, copy the binary out of the
# already-built image: `docker compose cp memexd:/usr/local/bin/wqm ./wqm`.
cargo build --release --manifest-path src/rust/Cargo.toml --package wqm-cli
export WQM_BIN="$(pwd)/src/rust/target/release/wqm"

# Override image tags if you want to test a local build:
export MEMEXD_IMAGE=ghcr.io/chrisgve/memexd:dev
export MCP_IMAGE=ghcr.io/chrisgve/workspace-qdrant-mcp:dev

tests/integration/docker/run-all.sh
```

Scripts clean up after themselves. If a test fails, the compose project
is torn down, but `TEST_SCRATCH` (container logs, state dirs) is left in
place under `/tmp/wqm-e2e-<test>.XXXXXX` so you can inspect it.

## Environment overrides

| Var | Default | Description |
|-----|---------|-------------|
| `WQM_BIN` | `wqm` on PATH | Host CLI binary used by `test-cli-connection.sh` and `test-path-transparency.sh` |
| `MEMEXD_IMAGE` | `ghcr.io/chrisgve/memexd:latest` | Daemon image |
| `MCP_IMAGE` | `ghcr.io/chrisgve/workspace-qdrant-mcp:latest` | MCP server image |
| `QDRANT_VERSION` | `latest` | Qdrant image tag |
| `MCP_HTTP_TOKEN` | random per-run | Bearer token used by `test-mcp-http.sh` |
| `POLL_TIMEOUT` | `180` | Seconds a test waits for each service to come up |

The scripts never read `docker/.env` — every required variable is
exported in `common.sh`, so tests are reproducible across hosts.

## CI

`.github/workflows/docker-integration.yml` runs the suite on ubuntu-latest
when any of `docker/**`, `src/typescript/**`, `src/rust/**/Dockerfile*`,
or these test scripts change. The workflow:

1. Builds `wqm` natively on the runner (no Docker cross-build, keeps the
   feedback loop short).
2. Pulls the pinned `MEMEXD_IMAGE` / `MCP_IMAGE` tags from GHCR — CI uses
   the same tags the `docker-publish` workflow just pushed, so the images
   under test reflect the PR contents.
3. Invokes `tests/integration/docker/run-all.sh`.
4. Uploads `/tmp/wqm-e2e-*` directories as an artifact on failure so
   compose logs + state are recoverable without re-running the job.
