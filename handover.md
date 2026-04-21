# Handover — 2026-04-21 (session 7)

## Current work

Closed two outstanding GitHub issues, then moved into the `docker`
task-master tag. 6/15 docker tasks complete. MCP server now has full
HTTP-mode wiring (Streamable transport + bearer auth + CORS + rate
limit + optional native TLS). `memexd` image bundles rust-analyzer,
gopls, pyright, typescript-language-server. Reference compose stack
lives at `docker/compose/reference.yml`, TLS overlay at
`docker/compose/reference.tls.yml`, Caddyfile at `docker/caddy/`.

All Rust + TypeScript tests green:
- `workspace-qdrant-core` lib: 2238 passed (+2 new in task #60)
- `workspace-qdrant-grpc` full: 150 passed
- `wqm-common`: 234 passed
- `wqm-cli` bin: 597 passed
- TypeScript MCP full suite: 507 passed (was 472, +35 across the
  three HTTP-mode commits: 6 mode-resolution + 7 HTTP transport +
  21 auth-middleware + 4 TLS transport + reshuffled)

Release binaries built + deployed to `~/.local/bin/` early in the
session for the #60 fix; daemon restarted via launchctl and came up
healthy. Daemon still healthy at time of writing (PID 29275,
uptime measured in hours — will differ on next restart).

## Closed this session

- **GH #60** — Linux-native idle detection. `/proc` heuristic chosen.
  Commit `153f06f11`. `src/rust/daemon/core/src/adaptive_resources/idle_detection.rs`
  refactored into `IdleDetector` struct; Linux backend reads
  `/proc/loadavg`, treats host as idle when `load_1m/cores < threshold`
  (default 0.1). New config fields `resource_limits.linux_idle_source`
  (`"none"` | `"proc"`) and `resource_limits.linux_idle_load_threshold`.
  Tests: 6 linux-only + 2 cross-platform.
- **GH #61** — linux/arm64 in docker-publish. Commit `c5d03af4d`.
  Split-matrix native-arm64 build (amd64 on `ubuntu-latest`, arm64 on
  `ubuntu-24.04-arm`) merged via `docker buildx imagetools create`.
  Zero incremental cost on this public repo. Full workflow rewrite in
  `.github/workflows/docker-publish.yml`.

## Docker tag progress (6/15 done)

| # | Title | Commit | Notes |
|---|-------|--------|-------|
| 1 | MCP StreamableHTTPServerTransport | `aba02bf8b` | `MCP_SERVER_MODE=http`, port 6335 default. `mcp-http-server.ts`. |
| 2 | Bearer token auth + rate limit + CORS | `78b279cb1` | `auth-middleware.ts`. `timingSafeEqual`, 100 req/min, digest logging. Refuses startup in http mode without `MCP_HTTP_TOKEN`. |
| 3 | Optional native TLS | `40255b62b` | `MCP_HTTP_TLS_CERT`/`MCP_HTTP_TLS_KEY` env. Default = plain HTTP (reverse-proxy path). |
| 4 | LSP bundle in memexd image | `a19004d38` | rust-analyzer (binary download), gopls (go-install build stage), pyright + typescript-language-server (npm -g). |
| 5 | MCP image http-mode env + /healthz probe | `ba890afab` | Dockerfile.mcp comment block lists every env var; `wget /healthz` HEALTHCHECK on `MCP_HTTP_PORT`. |
| 6 | Reference compose + TLS overlay | `c9808f12d` | `docker/compose/reference.yml` (qdrant + memexd + mcp), `reference.tls.yml` (Caddy), `docker/caddy/Caddyfile`. |

## Docker tag remaining (9/15)

Pending (IDs in the `docker` tag):
- **7** — Example config + onboarding docs (medium)
- **8** — `wqm config` profile management (medium)
- **9** — Upgrade `wqm health` to probe MCP HTTP (medium)
- **10** — Integrate compose with existing observability overlay (low)
- **11** — Grafana dashboards for http mode (low)
- **12** — Docker deployment docs (medium)
- **13** — CI workflow to build + test images on PR (medium)
- **14** — Integration test suite for compose (medium)
- **15** — Chaos testing + graceful degradation (low)

Priority: 7 → 13 → 9 → 12. 8 and 10 can wait. 11/14/15 are follow-ups.

## Decisions locked in (user confirmed in session 7)

- **C1 distribution**: publish to both Docker Hub AND GHCR at no extra
  cost. Existing workflow already does this — no change needed.
- **C2 token rotation**: static bearer from `.env`. Option A, simplest.
  Migration to per-client admin endpoint deferred until multi-user
  demand appears (additive: `auth_tokens` table, keep
  `MCP_HTTP_TOKEN` as implicit root token).
- **C3 TLS strategy**: reverse-proxy default (Caddy overlay). Native
  TLS remains optional fallback (task 3).
- **C4 MCP HTTP port**: 6335 confirmed. Configurable via
  `MCP_HTTP_PORT`.
- **#60 backend**: `/proc` heuristic (option 2). Ship simplest, works
  everywhere Linux-native.
- **#61 arm64**: split-matrix native runner (options 1+4). Public repo
  → free `ubuntu-24.04-arm`.

## Resume instructions

1. Read this file; `git log --oneline -12` for recent commits.
2. Verify daemon still healthy: `wqm service status`. If down,
   `launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist`.
3. Continue in `docker` tag: `task-master use-tag docker`, then
   `task-master next`. Task 7 is next (example config + onboarding).
4. Task list intent summary:
   - **7** writes `docker/config.example.yaml` + a short `docker/README.md`
     quickstart pointing at the reference compose. Reuse the env-var
     documentation already in `reference.yml`'s header comment block.
   - **13** adds a PR-triggered GH workflow that builds both Dockerfiles
     on amd64 only (no publish) to catch regressions. The multi-arch
     `docker-publish.yml` already exists for tagged releases.
   - **9** extends `wqm admin health` / `wqm status health` to check
     the MCP HTTP `/healthz` endpoint (respect `MCP_HTTP_PORT`,
     optionally `MCP_HTTP_TLS_*`).
5. `.env.example` manual edit still pending (blocked by sensitive-file
   hook). User should add the block documented at
   `docker/compose/reference.yml:1` — see the env-var list in the
   commit message of `c9808f12d`. Variables to add:
   - `MCP_HTTP_TOKEN`, `MCP_HTTP_PORT`, `MCP_HTTP_PATH`,
     `MCP_HTTP_RATE_LIMIT`, `MCP_HTTP_CORS_ORIGINS`
   - `WQM_VERSION`, `WQM_STATE_DIR`, `WQM_DEV_ROOT`,
     `WQM_CONFIG_FILE`
   - `QDRANT_VERSION`, `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT`
   - `MCP_PUBLIC_HOSTNAME`, `MCP_TLS_EMAIL`
   - `OTEL_EXPORTER_OTLP_ENDPOINT`

## Pending decisions (none blocking docker tag)

All four docker-tag decisions from session 6 are closed. No new
decisions outstanding for the next 4-5 docker tasks.

## Key context

### Commits shipped this session (all on `origin/main`)

| Commit | Summary |
|--------|---------|
| `153f06f11` | `feat(daemon): add linux /proc idle detection backend` — closes #60 |
| `c5d03af4d` | `ci(docker): build linux/arm64 natively via split-matrix manifest merge` — closes #61 |
| `aba02bf8b` | `feat(mcp): add HTTP Streamable transport alongside stdio` — docker task 1 |
| `78b279cb1` | `feat(mcp): add bearer token auth, rate limit, and CORS for HTTP mode` — docker task 2 |
| `40255b62b` | `feat(mcp): add optional native TLS termination for HTTP mode` — docker task 3 |
| `a19004d38` | `build(docker): bundle rust-analyzer, gopls, pyright, typescript-language-server in memexd image` — docker task 4 |
| `ba890afab` | `build(docker): document http-mode env vars and add /healthz check for mcp image` — docker task 5 |
| `c9808f12d` | `build(docker): reference compose stack with optional TLS overlay` — docker task 6 |

### Architectural invariants (carry forward)

- **MCP HTTP default = plain HTTP**. Terminate TLS at Caddy in the
  overlay. Native TLS (`MCP_HTTP_TLS_*` env) is fallback only.
- **HTTP mode requires a bearer token**. `requireAuth()` enforces
  16-char minimum at startup; refuses to launch without it.
- **`/healthz` is auth-exempt**. Only endpoint that bypasses the
  middleware — narrow `GET /healthz` exact match.
- **Rate-limit lives in-process**. Single-process only; multi-node
  deployments must terminate rate limit at the reverse proxy.
- **Audit logging uses `tokenDigest`** — first 8 hex chars of
  `SHA-256(token)`. Never log the raw secret.
- **Linux idle detection gated behind config**. Default
  `linux_idle_source = "none"` preserves macOS-only behaviour on new
  deployments; `"proc"` opt-in via `resource_limits.linux_idle_source`.
- **Path transparency in compose**. `WQM_DEV_ROOT` is bind-mounted at
  the identical path inside `memexd` so daemon-stored absolute paths
  reconcile with host `wqm` CLI state. Compose refuses to start
  without it (`${WQM_DEV_ROOT:?…}` form).
- **Native binary tier for LSPs**: rust-analyzer from github releases
  (pinned `RUST_ANALYZER_VERSION` ARG), gopls from-source via golang
  build stage (pinned `GOPLS_VERSION`), pyright + typescript-language-
  server via npm global install in runtime stage.

### Test state

- Rust:
  - `workspace-qdrant-core` lib: 2238 ✓
  - `workspace-qdrant-grpc` full: 150 ✓
  - `wqm-common`: 234 ✓
  - `wqm-cli` bin: 597 ✓
- TypeScript MCP: 507 ✓ / 2 skipped
- Dockerfiles: `docker buildx build --check` clean on both
  Dockerfile.memexd and Dockerfile.mcp

### Reference files touched this session

**Rust (issue #60)**:
- `src/rust/daemon/core/src/adaptive_resources/idle_detection.rs` —
  `IdleDetector` struct, `linux_idle::LinuxIdleDetector` + trait-based
  testable `LoadReader`.
- `src/rust/daemon/core/src/adaptive_resources/manager.rs` — passes
  `ResourceLimitsConfig` through to detector construction.
- `src/rust/daemon/core/src/adaptive_resources/tests.rs` — 2 new
  cross-platform tests (detector default, linux source validation).
- `src/rust/daemon/core/src/config/resource_limits.rs` — two new fields
  + env overrides + validation.
- `src/rust/common/src/yaml_defaults/infrastructure.rs` — yaml side.
- `src/rust/daemon/core/src/config/mod.rs` — yaml → resource-limits
  wiring.
- `src/rust/daemon/memexd/src/queue_init.rs` — passes config through
  to `AdaptiveResourceManager::start`.
- `assets/default_configuration.yaml` — docs + defaults.

**CI (#61)**:
- `.github/workflows/docker-publish.yml` — rewritten fan-out/fan-in.

**TypeScript MCP (docker tasks 1–3)**:
- `src/typescript/mcp-server/src/mcp-http-server.ts` — new; wraps
  `StreamableHTTPServerTransport` in `http`/`https` server, invokes
  auth middleware, exposes `/healthz`.
- `src/typescript/mcp-server/src/auth-middleware.ts` — new; bearer
  auth, sliding-window rate limiter, CORS.
- `src/typescript/mcp-server/src/server-types.ts` — `ServerMode`,
  `HttpTransportOptions` (+ optional `tls` nested struct),
  `ServerOptions.auth`, defaults consts.
- `src/typescript/mcp-server/src/server.ts` — mode resolution,
  conditional stdio/http/test branches, handle lifecycle.
- `src/typescript/mcp-server/src/index.ts` — env → mode + transport
  + TLS options.
- `src/typescript/mcp-server/tests/server-core.test.ts` — 4 new mode
  tests.
- `src/typescript/mcp-server/tests/server-http-transport.test.ts` —
  new; 7 tests (init, 404, custom path, missing-token startup fail,
  invalid token, rate-limit, CORS preflight).
- `src/typescript/mcp-server/tests/auth-middleware.test.ts` — new;
  21 unit tests.
- `src/typescript/mcp-server/tests/server-http-tls.test.ts` — new;
  4 TLS tests using self-signed cert generated via `openssl req
  -x509` in `beforeAll`.

**Docker (tasks 4–6)**:
- `docker/Dockerfile.memexd` — +80 lines, two new stages
  (`rust-analyzer-download`, `gopls-build`), runtime stage gains
  nodejs/npm/git + global npm installs + LSP binary copies.
- `docker/Dockerfile.mcp` — HTTP-mode env-var documentation block,
  new EXPOSE 6335, HEALTHCHECK against `/healthz`.
- `docker/compose/reference.yml` — new; self-contained three-service
  stack.
- `docker/compose/reference.tls.yml` — new; Caddy overlay.
- `docker/caddy/Caddyfile` — new; Let's Encrypt + reverse proxy to
  `mcp:6335`.

### Gotchas carried forward

- `task-master parse-prd` takes ~3 min per PRD.
- `numTasks` hint, not exact.
- Tag names cannot start with `#`.
- Env-mutating tests must be `#[serial]`.
- Prometheus `METRICS` global singleton; tests use delta asserts.
- `opentelemetry-otlp` 0.14 headers injection: use
  `OTEL_EXPORTER_OTLP_HEADERS` env.
- Clippy pre-existing warnings in `graph/algorithms/community.rs`,
  `graph/sqlite_store.rs`, `patterns/detection/detector.rs`,
  `keyword_extraction/semantic_rerank.rs`. Exit 0.
- `git/index.lock` occasional linger after background task-master
  calls — `rm -f .git/index.lock` if it blocks a commit.
- `ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib` required for
  every `cargo build` / `cargo test`.
- `wqm-cli` ~47 pre-existing compile warnings.
- `unified_queue` has no `priority` column; priority computed in
  `build_dequeue_query`'s `ORDER BY`.
- **New (session 7)**: `docker/docker-compose.yml` is gitignored (as
  "legacy"). Reference stack lives under `docker/compose/reference*.yml`.
  Legacy top-level file was removed locally as part of task 6 but
  remains in the gitignore.
- **New (session 7)**: `docker/.env.example` edits are blocked by the
  sensitive-file hook. Document new variables in compose headers;
  user manually merges additions into `.env.example`.
- **New (session 7)**: Dockerfile changes should be verified with
  `docker buildx build --check`; real builds take 15+ min and need
  ONNX Runtime, so rely on CI for the full rebuild.
- **New (session 7)**: `task-master set-status --id=<N>` only works
  when the ID lives in the currently selected tag. `use-tag docker`
  is required before task-master commands touch docker tasks.
- **New (session 7)**: `cd` in `Bash` is not persistent — working
  directory resets between calls. Use absolute paths or inline
  `cd … && …`.
