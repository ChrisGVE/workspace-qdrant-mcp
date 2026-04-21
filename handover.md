# Handover — 2026-04-21 (session 7)

## Current work

Closed two outstanding GitHub issues, then worked through the `docker`
task-master tag. **10/15 docker tasks complete**. MCP HTTP mode is
production-ready: Streamable transport + bearer auth + CORS + rate
limit + optional native TLS + observability. `memexd` image bundles
rust-analyzer, gopls, pyright, typescript-language-server. Reference
compose stack lives at `docker/compose/reference.yml`, TLS overlay at
`docker/compose/reference.tls.yml`, Caddy config at `docker/caddy/`.
Operator docs at `docker/compose/README.md` (quickstart) and
`docs/deployment/docker.md` (long-form reference). `wqm status health`
now probes Qdrant + MCP HTTP alongside the daemon.

All tests green:
- `workspace-qdrant-core` lib: 2238 ✓
- `workspace-qdrant-grpc` full: 150 ✓
- `wqm-common`: 234 ✓
- `wqm-cli` bin: **606** ✓ (was 597, +9 new health probes)
- TypeScript MCP full: 507 ✓ / 2 skipped

## Closed this session

- **GH #60** — Linux-native idle detection (`/proc` heuristic,
  commit `153f06f11`).
- **GH #61** — linux/arm64 in docker-publish (split-matrix native
  runners, commit `c5d03af4d`).

## Docker tag progress (10/15 done)

| # | Title | Commit | Notes |
|---|-------|--------|-------|
| 1 | MCP StreamableHTTPServerTransport | `aba02bf8b` | `MCP_SERVER_MODE=http`, port 6335 default. |
| 2 | Bearer token auth + rate limit + CORS | `78b279cb1` | `timingSafeEqual`, 100 req/min, digest logging. Refuses start without `MCP_HTTP_TOKEN`. |
| 3 | Optional native TLS | `40255b62b` | `MCP_HTTP_TLS_CERT`/`_KEY` env. Default = plain HTTP (reverse-proxy path). |
| 4 | LSP bundle in memexd image | `a19004d38` | rust-analyzer, gopls (go-install), pyright + typescript-language-server (npm -g). |
| 5 | MCP image http-mode env + /healthz probe | `ba890afab` | Dockerfile.mcp comment block; `wget /healthz` HEALTHCHECK. |
| 6 | Reference compose + TLS overlay | `c9808f12d` | `reference.yml`, `reference.tls.yml`, Caddyfile. |
| 7 | Example config + quickstart docs | `86d8af…` (see log) | `docker/config.example.yaml`, `docker/compose/README.md`. |
| 9 | `wqm status health` probes Qdrant + MCP HTTP | `eecbaa…` (see log) | Concurrent probes via `tokio::join!`, 3s timeout, cert-invalid accepted. |
| 12 | Deployment reference doc | `docs/deploy…` (see log) | `docs/deployment/docker.md` — long-form reference. |
| 13 | PR docker build workflow | updated existing `docker-test.yml` | Passes `MCP_HTTP_TOKEN` and probes `/healthz` + `/mcp` 401. |

Run `git log --oneline -14` to see exact hashes — the task-7/9/12/13
commits all shipped after the `c9808f12d` reference-compose commit.

## Docker tag remaining (5/15)

- **8** — `wqm config` profile management (medium). Needs a
  `~/.config/wqm/cli-config.toml`, `config list|use|show` commands,
  and rewiring the DaemonClient/Qdrant client constructors to read the
  active profile. Bigger lift than a single session — recommend
  expanding into sub-tasks with `task-master expand --id=8`.
- **10** — Integrate reference compose with existing observability
  stack (low). Point Prometheus at memexd/mcp metrics endpoints; add
  an optional `-f observability.yml` overlay that adds OTEL collector.
- **11** — Grafana dashboards for http mode (low). Depends on 10.
- **14** — Integration test suite for compose (medium). Real
  `docker compose up` on CI, spin full stack, exercise `wqm project
  activate` + an MCP `initialize` → `search` flow end-to-end.
- **15** — Chaos testing / graceful degradation (low). Sigkill one
  service at a time, assert the others stay up.

Priority if continuing: 8 → 14 → 10 → 11 → 15.

## Decisions locked in (user confirmed)

- **C1 distribution**: Docker Hub + GHCR at no extra cost. Existing
  workflow already does this — no change needed.
- **C2 token rotation**: static bearer from `.env`. Migration to
  per-client admin endpoint deferred.
- **C3 TLS strategy**: reverse-proxy default (Caddy overlay). Native
  TLS is optional fallback.
- **C4 MCP HTTP port**: 6335, configurable via `MCP_HTTP_PORT`.
- **#60 backend**: `/proc` heuristic (option 2).
- **#61 arm64**: split-matrix native runner (free on public repo).

## Resume instructions

1. Read this file; `git log --oneline -14`.
2. `wqm service status`; `launchctl load …plist` if daemon is down.
3. `task-master use-tag docker` → `task-master next` (expect #8).
4. Task 8 is larger than average — consider
   `task-master expand --id=8 --research=false` before starting.
5. Task 14 should run a real `docker compose up` in CI. Gate the
   workflow behind a path filter (`docker/**`, `src/typescript/**`,
   `src/rust/**/Dockerfile*`) so it doesn't run on every PR.
6. `.env.example` manual edit still pending (blocked by sensitive-file
   hook). See `docker/compose/reference.yml`'s header comment for the
   canonical env-var list. Variables to merge in by hand:
   - `MCP_HTTP_TOKEN`, `MCP_HTTP_PORT`, `MCP_HTTP_PATH`,
     `MCP_HTTP_RATE_LIMIT`, `MCP_HTTP_CORS_ORIGINS`
   - `WQM_VERSION`, `WQM_STATE_DIR`, `WQM_DEV_ROOT`, `WQM_CONFIG_FILE`
   - `QDRANT_VERSION`, `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT`
   - `MCP_PUBLIC_HOSTNAME`, `MCP_TLS_EMAIL`
   - `OTEL_EXPORTER_OTLP_ENDPOINT`

## Pending decisions

None blocking the remaining docker tasks.

## Key context

### Commits shipped this session (all on `origin/main`)

| Commit | Summary |
|--------|---------|
| `153f06f11` | `feat(daemon): add linux /proc idle detection backend` — closes #60 |
| `c5d03af4d` | `ci(docker): build linux/arm64 natively via split-matrix manifest merge` — closes #61 |
| `aba02bf8b` | `feat(mcp): add HTTP Streamable transport alongside stdio` — docker #1 |
| `78b279cb1` | `feat(mcp): add bearer token auth, rate limit, and CORS for HTTP mode` — docker #2 |
| `40255b62b` | `feat(mcp): add optional native TLS termination for HTTP mode` — docker #3 |
| `a19004d38` | `build(docker): bundle rust-analyzer, gopls, pyright, typescript-language-server in memexd image` — docker #4 |
| `ba890afab` | `build(docker): document http-mode env vars and add /healthz check for mcp image` — docker #5 |
| `c9808f12d` | `build(docker): reference compose stack with optional TLS overlay` — docker #6 |
| (see log) | `docs(docker): example config file and compose quickstart` — docker #7 |
| (see log) | `ci(docker): update PR build test for the new MCP http-mode contract` — docker #13 |
| (see log) | `docs(deploy): add docs/deployment/docker.md reference guide` — docker #12 |
| (see log) | `feat(cli): extend wqm status health to probe qdrant and mcp http` — docker #9 |

### Architectural invariants

- **MCP HTTP default = plain HTTP**. Terminate TLS at Caddy. Native
  TLS is fallback only.
- **HTTP mode requires a bearer token**. `requireAuth()` enforces
  16-char minimum; refuses to launch without it.
- **`/healthz` is auth-exempt**. Only endpoint bypassing the
  middleware — narrow `GET /healthz` exact match.
- **Rate-limit lives in-process**. Multi-node deployments must
  terminate rate limit at the reverse proxy.
- **Audit logging uses `tokenDigest`** — first 8 hex chars of
  `SHA-256(token)`. Never log the raw secret.
- **Linux idle detection gated behind config**. Default
  `linux_idle_source = "none"`. `"proc"` opt-in.
- **Path transparency in compose**. `WQM_DEV_ROOT` bind-mounted at
  identical path inside `memexd`. Compose refuses to start without
  it.
- **Native binary tier for LSPs**: rust-analyzer from github (pinned
  `RUST_ANALYZER_VERSION`), gopls from-source (pinned
  `GOPLS_VERSION`), pyright + typescript-language-server via npm
  global.
- **Health probes use 3s timeout + `danger_accept_invalid_certs`**
  so native-TLS deployments with self-signed certs don't false-fail
  `wqm status health`.

### Test state

- Rust: core 2238, grpc 150, common 234, cli 606 ✓
- TypeScript: 507 ✓ / 2 skipped
- Dockerfiles: `docker buildx build --check` clean on both files.
  Full build requires CI (ONNX + 15 min).

### Reference files touched this session

**Rust (issue #60)**:
- `src/rust/daemon/core/src/adaptive_resources/idle_detection.rs` —
  `IdleDetector` struct, `linux_idle::LinuxIdleDetector`, trait-based
  `LoadReader`.
- `src/rust/daemon/core/src/adaptive_resources/manager.rs` — passes
  `ResourceLimitsConfig` through to detector.
- `src/rust/daemon/core/src/adaptive_resources/tests.rs` — 2 new
  cross-platform tests.
- `src/rust/daemon/core/src/config/resource_limits.rs` — new fields +
  env overrides + validation.
- `src/rust/common/src/yaml_defaults/infrastructure.rs` — yaml side.
- `src/rust/daemon/core/src/config/mod.rs` — yaml → resource-limits.
- `src/rust/daemon/memexd/src/queue_init.rs` — passes config through.
- `assets/default_configuration.yaml` — docs + defaults.

**CI**:
- `.github/workflows/docker-publish.yml` — fan-out/fan-in rewrite.
- `.github/workflows/docker-test.yml` — PR HTTP probe updated for
  bearer auth.

**TypeScript MCP (docker #1–3)**:
- `src/typescript/mcp-server/src/mcp-http-server.ts` — new.
- `src/typescript/mcp-server/src/auth-middleware.ts` — new.
- `src/typescript/mcp-server/src/server-types.ts` — `ServerMode`,
  `HttpTransportOptions` + `tls`, `ServerOptions.auth`, defaults.
- `src/typescript/mcp-server/src/server.ts` — mode resolution,
  branches.
- `src/typescript/mcp-server/src/index.ts` — env → mode + transport
  + TLS options.
- `tests/server-core.test.ts` — 4 mode tests.
- `tests/server-http-transport.test.ts` — new, 7 tests.
- `tests/auth-middleware.test.ts` — new, 21 tests.
- `tests/server-http-tls.test.ts` — new, 4 TLS tests.

**Docker (#4–7, #12–13)**:
- `docker/Dockerfile.memexd` — +80 lines, 2 new stages, LSP bundle.
- `docker/Dockerfile.mcp` — HTTP-mode env docs, 6335 expose, healthz.
- `docker/compose/reference.yml` — new.
- `docker/compose/reference.tls.yml` — new.
- `docker/caddy/Caddyfile` — new.
- `docker/config.example.yaml` — new.
- `docker/compose/README.md` — new.
- `docs/deployment/docker.md` — new.

**CLI (#9)**:
- `src/rust/cli/src/commands/status/health.rs` — +external probes,
  `tokio::join!`, 9 new tests.

### Gotchas carried forward

- `task-master parse-prd` takes ~3 min per PRD.
- `numTasks` hint, not exact.
- Tag names cannot start with `#`.
- Env-mutating tests must be `#[serial]`.
- Prometheus `METRICS` global singleton; tests use delta asserts.
- `opentelemetry-otlp` 0.14: use `OTEL_EXPORTER_OTLP_HEADERS` env.
- Pre-existing clippy warnings in `graph/algorithms/community.rs`,
  `graph/sqlite_store.rs`, `patterns/detection/detector.rs`,
  `keyword_extraction/semantic_rerank.rs`. Exit 0.
- `git/index.lock` occasional linger → `rm -f .git/index.lock` if
  commits block.
- `ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib` required
  for every `cargo build` / `cargo test`.
- `wqm-cli` ~47 pre-existing compile warnings.
- `unified_queue` has no `priority` column; priority computed in
  `build_dequeue_query`'s `ORDER BY`.
- **Session 7**: `docker/docker-compose.yml` is gitignored as legacy.
  Reference stack lives under `docker/compose/reference*.yml`.
- **Session 7**: `docker/.env.example` edits blocked by
  sensitive-file hook. Document new variables in compose headers;
  user merges into `.env.example` by hand.
- **Session 7**: Dockerfile changes verified with `docker buildx
  build --check`; real builds take 15+ min and need ONNX Runtime.
- **Session 7**: `task-master set-status` only works on IDs in the
  currently selected tag. `use-tag <name>` before commands.
- **Session 7**: `cd` in `Bash` is not persistent — working directory
  resets between calls. Use absolute paths or inline `cd … && …`.
- **Session 7**: `Edit`/`Write` tool blocks writes to `.env*` files.
  Planned additions recorded in compose headers + this handover.
- **Session 7**: After `mv`, a subsequent `Edit` on the moved file
  requires a fresh `Read` of the new path (session state tracks
  per-path).
