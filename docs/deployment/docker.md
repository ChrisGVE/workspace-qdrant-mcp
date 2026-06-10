# Docker deployment

This guide covers running `workspace-qdrant-mcp` as a multi-container
Docker stack. Start here if you want a single-host deployment reachable
from any MCP-speaking client; the canonical entrypoint is the root
`docker-compose.yml`, which builds `memexd` and the MCP server locally
from this checkout when you pass `--build`. Optional profiles enable TLS
and local embeddings.
Additional compose files live under `docker/compose/` for specific
overlays and examples.

All compose files in this repo build images locally from this checkout.
Use `--build` whenever you run them so TypeScript and Rust changes make
it into the container image.

For a one-page quickstart jump to
[`docker/compose/README.md`](../../docker/compose/README.md). This
document is the longer reference — it explains the invariants that make
the stack work and what to touch when things go wrong.

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Docker Engine | 20.10+ | `docker buildx` must be available. Any recent Docker Desktop or server install qualifies. |
| Docker Compose | 2.x | Compose v1 (`docker-compose`) is not supported; the reference files use v2-only syntax (`!reset`, etc.). |
| Disk space | ~10 GB | Rust binaries + Qdrant snapshots add up quickly. |
| Host OS | macOS, Linux, Windows (WSL2) | Linux is strongly recommended for production — see [File watching caveats](#file-watching-caveats). |

Public FQDN optional. Only required when you enable the `tls` profile
and want Let's Encrypt certificates. Local-only deployments can skip it.

## First-run flow

1. **Clone and enter the repo.**

   ```bash
   git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
   cd workspace-qdrant-mcp
   ```

2. **Create `docker/.env`.** Copy `docker/.env.example` and set at least:

   ```bash
   # Generate a token (run this in your shell — not inside .env):
   openssl rand -hex 32
   ```

   Paste the output as a literal value in `docker/.env`:

   ```env
   MCP_HTTP_TOKEN=<paste-openssl-output-here>   # required
   WQM_DEV_ROOT=/Users/you/dev                  # required — see Path transparency
   WQM_VERSION=latest
   WQM_STATE_DIR=./state
   ```

   > **Warning:** Docker Compose `.env` files do not evaluate `$()` command
   > substitution. Paste the literal hex value; do not write
   > `MCP_HTTP_TOKEN=$(openssl rand -hex 32)` inside `.env`.

   The compose file refuses to start if `MCP_HTTP_TOKEN` or
   `WQM_DEV_ROOT` is missing. The full variable list is documented in
   the header comment of `docker-compose.yml`.

3. *(Optional)* Copy the example daemon config if you want to override
   any built-in defaults:

   ```bash
   cp docker/config.example.yaml ~/.config/wqm/config.yaml
   # edit workspace.include_paths, qdrant.url, etc.
   ```

   Then point `WQM_CONFIG_FILE` at that file in `docker/.env`.

4. **Launch.**

   ```bash
   docker compose \
     --env-file docker/.env \
     up -d --build
   ```

5. **Verify services come up healthy.**

   ```bash
   docker compose \
     --env-file docker/.env \
     ps

   # Qdrant liveness (unauthenticated).
   curl -fsS http://127.0.0.1:6333/readyz

   # memexd health + metrics.
   curl -fsS http://127.0.0.1:9091/health
   curl -fsS http://127.0.0.1:9091/metrics | head -5

   # MCP liveness (auth-exempt) and auth enforcement.
   curl -fsS http://127.0.0.1:6335/healthz
   curl -o /dev/null -w "%{http_code}\n" \
        -X POST http://127.0.0.1:6335/mcp     # → 401

   # Full MCP handshake.
   curl -sS \
        -H "Authorization: Bearer ${MCP_HTTP_TOKEN}" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}' \
        http://127.0.0.1:6335/mcp
   ```

6. **Connect a client.** For Claude Desktop / `claude` CLI, point your
   MCP configuration at `http://127.0.0.1:6335/mcp` (or the TLS URL if
   you ran the overlay) with a `Bearer <MCP_HTTP_TOKEN>` header.

7. **(Optional) open the admin UI.** The MCP container also serves a
   browser dashboard at `http://127.0.0.1:6335/admin/`. Paste your
   `MCP_HTTP_TOKEN` to log in. From there you can configure a parent
   directory, scan it for git repositories, and register them with
   the daemon — useful for setups with many sibling repos that you
   don't want to add via `wqm` one-by-one. See [Admin UI](../ADMIN_UI.md).

## Configuration

### `docker/.env`

Everything in the unified stack is parameterised via environment
variables. The header comment in `docker-compose.yml` lists every
variable with its default and purpose. The two it enforces:

- `MCP_HTTP_TOKEN` — bearer secret. Generate with `openssl rand -hex 32`.
  Minimum 16 characters. Rotation = edit `.env`, then
  `docker compose up -d` to restart the MCP container.
- `WQM_DEV_ROOT` — absolute host path bind-mounted into `memexd` at
  the identical path (see [Path transparency](#path-transparency)).

### `config.yaml`

Only the values you want to override need to live in your config file;
everything else falls back to `assets/default_configuration.yaml`
inside the image. The example at `docker/config.example.yaml` covers
the keys that most deployments touch (Qdrant connection, workspace
include paths, log level, OTLP, resource limits).

### Path transparency

The daemon stores **absolute** paths for every watched folder in its
SQLite state. The host `wqm` CLI queries that state over gRPC and
passes paths straight back to the daemon. If the daemon's filesystem
view does not match the host's, `wqm project activate $(pwd)` cannot
find a matching watch folder.

The reference compose solves this with a single volume line:

```yaml
- ${WQM_DEV_ROOT}:${WQM_DEV_ROOT}
```

Set `WQM_DEV_ROOT` to the host directory whose subtrees you want
indexed (e.g. `/Users/you/dev`). Every project under it is reachable
by the daemon at the same absolute path. Without this, activation from
the host fails silently.

### Embedding provider and collection dimension

Since 2026-06-10 the reference deployment runs
`WQM_EMBEDDING_PROVIDER=openai_compatible` with
`intfloat/multilingual-e5-large` (1024d) served by in-stack backend
containers — a GPU one (Infinity, preferred) and a CPU one (TEI, warm
standby), selected via `COMPOSE_PROFILES` with automatic daemon-side
failover between them. **See
[embeddings.md](embeddings.md)** for the full picture: backend selection,
failover semantics, the NVIDIA Container Toolkit requirement (and why),
Blackwell/TEI compatibility, and the model-change/reembed procedure.

`WQM_EMBEDDING_PROVIDER=fastembed` remains the zero-dependency fallback
(in-process ONNX pinned to the 384-dim `AllMiniLM-L6-v2` checkpoint) for
minimal setups without an embedding service.

If you are reusing an older Qdrant state that was created with a
different embedding dimension, startup will fail fast with an embedding
dimension mismatch. To recover, either:

- run `wqm admin reembed --confirm` to rebuild the existing collections
  at the active dimension (start memexd once with `--bootstrap-reembed`
  so the guard lets it boot first), or
- delete the stale collections (`projects`, `libraries`, `rules`,
  `scratchpad`, `images`) and let the daemon recreate them on the next
  start.

If you are starting from a fresh `WQM_STATE_DIR`, no extra action is
needed.

For clarity, this is the containerized `memexd` in the Docker stack.
The host-local FastEmbed helper used by the Windows scripts is a
separate optional path and stays on `55151`; do not mix the two.

## Networking

| Port | Service | Exposed by |
|------|---------|-----------|
| 6333 | qdrant | Qdrant REST |
| 6334 | qdrant | Qdrant gRPC |
| 50051 | memexd | memexd gRPC (`wqm` CLI) |
| 9091 | memexd | memexd Prometheus metrics + `/health` |
| 6335 | mcp | MCP Streamable HTTP (`/mcp`, `/healthz`); admin dashboard under `/admin/` (same port, Bearer-authed REST under `/admin/api/*`) |
| 9092 | mcp | MCP Prometheus metrics |
| 80, 443 | caddy (TLS overlay only) | Reverse proxy to MCP |

All services share the internal bridge network `workspace-network`.
External clients only reach the published ports on the host.

### TLS via Caddy

Enable the `tls` profile to terminate HTTPS with automatic Let's Encrypt
certificates:

```bash
docker compose \
  --env-file docker/.env \
  --profile tls \
  up -d --build
```

Required in `.env`:

```env
MCP_PUBLIC_HOSTNAME=mcp.example.com
MCP_TLS_EMAIL=you@example.com
```

The FQDN must resolve to the host running compose, and ports 80 +
443 must reach it (the HTTP-01 challenge runs on 80). Caddy persists
its ACME state under `${WQM_STATE_DIR}/caddy/` so restarts re-use the
existing certificate.

Client URL becomes `https://${MCP_PUBLIC_HOSTNAME}/mcp`. The MCP
container no longer publishes 6335 to the host — internal traffic
stays on `workspace-network`.

### Alternative reverse proxies

If you already run nginx/Traefik on the host, skip the TLS overlay
and point your proxy at `http://127.0.0.1:6335/mcp`. The MCP server
does not need to know about TLS in that case.

### Native TLS fallback

For deployments that can't accommodate a reverse proxy, the MCP
server can terminate TLS itself. Set `MCP_HTTP_TLS_CERT` +
`MCP_HTTP_TLS_KEY` (absolute paths mounted into the mcp container) and
the listener switches from `http` to `https` on the same port.
Certificate rotation requires a container restart.

## Storage

| Path | Owner | Persisted via |
|------|-------|---------------|
| `${WQM_STATE_DIR}/qdrant/storage` | Qdrant | Bind-mount (host directory) |
| `${WQM_STATE_DIR}/qdrant/snapshots` | Qdrant | Bind-mount |
| `${WQM_STATE_DIR}/memexd` | memexd | Bind-mount (SQLite, queue, rules cache) |
| `${WQM_STATE_DIR}/memexd/cache/workspace-qdrant` | memexd | Bind-mount (tree-sitter grammars, LSP, OCR cache) |
| `${WQM_STATE_DIR}/caddy/data` | caddy (TLS) | Bind-mount (ACME state, certs) |

**Backup strategy.** The bind-mounted directories contain all durable
state. The cache under `${WQM_STATE_DIR}/memexd/cache/workspace-qdrant`
is regenerable, so you can omit it if you want a smaller archive. A
cold backup is just:

```bash
docker compose down
tar czf wqm-backup-$(date +%F).tgz "${WQM_STATE_DIR}"
docker compose up -d
```

For hot backup of Qdrant, use its snapshot API (`POST /snapshots`) and
copy out of `${WQM_STATE_DIR}/qdrant/snapshots`.

## Security

- **Bearer token.** Required, minimum 16 chars. Rotate by editing
  `.env` and restarting the MCP container (`docker compose up -d`
  re-creates only the services with changed config). The server logs
  an 8-char SHA-256 digest on startup so you can confirm a rotation
  took effect without exposing the secret.
- **Rate limit.** Default 100 requests/min/IP, configurable via
  `MCP_HTTP_RATE_LIMIT`. The limiter lives in-process; multi-node
  deployments should rate-limit at the proxy instead.
- **CORS.** Disabled by default (no browser-origin traffic). Set
  `MCP_HTTP_CORS_ORIGINS` to a comma-separated list of origins if you
  need to call the server from a web app.
- **Firewall.** When using Caddy, only expose 80 + 443 to the internet
  and leave 6335 on the loopback interface (it is only used inside
  `workspace-network`). When exposing 6335 directly, front it with
  your host firewall so only trusted networks reach it.
- **Non-root containers.** Both images run as unprivileged users
  (`memexd` UID 1000, `node` UID 1000 in the MCP alpine image).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mcp` crashes with `MCP_HTTP_TOKEN is required` | Token not set in `.env` | Generate one: `openssl rand -hex 32`, put it in `.env`, restart. |
| `mcp` crashes with `MCP_HTTP_TOKEN must be at least 16 characters` | Token too short | Regenerate with the line above. |
| `wqm project activate $(pwd)` on host reports "no matching watch" | Daemon stored a different absolute path | Check `WQM_DEV_ROOT` in `.env` covers the current working directory and is identical on host + container. |
| memexd logs `qdrant: connection refused` | Qdrant not yet healthy | Wait a few seconds; memexd retries. Persistent failure → `docker compose logs qdrant`. |
| `/healthz` returns 200 but `/mcp` returns 401 for valid token | Token mismatch between `.env` and client | Compare `docker exec wqm-mcp printenv MCP_HTTP_TOKEN` to the value your client sends. |
| `/healthz` returns 200 but `/mcp` responses are slow on first call | Cold start — daemon registering the project | Subsequent calls should be fast. Investigate daemon logs if it persists. |
| TLS overlay: Caddy logs `ACME: acme: error 403` | FQDN does not resolve to this host, or port 80 is blocked | Confirm DNS + that 80/TCP is open on the public interface. |
| `docker compose` says `unsupported attribute "!reset"` | Compose v1 in use | Upgrade to Docker Compose v2 (`docker compose`, no dash). |

Log locations:

```bash
  docker compose --env-file docker/.env logs -f memexd
  docker compose --env-file docker/.env logs -f mcp
  docker compose --env-file docker/.env logs -f qdrant
```

## Teardown

Stop the stack but keep all state:

```bash
docker compose --env-file docker/.env down
```

Stop + delete named volumes (grammar cache, LSP cache):

```bash
docker compose --env-file docker/.env down -v
```

The bind-mounted state under `${WQM_STATE_DIR}` is never deleted by
Docker; remove it manually if you want a clean slate.

## File watching caveats

`memexd` uses the OS-native file-event API exposed through
`notify-debouncer-full`. Behaviour depends on the host kernel:

- **Linux**: `inotify`. The default watch limit
  (`fs.inotify.max_user_watches`) is typically 8192; bump it if you
  index large trees.
  ```bash
  sudo sysctl -w fs.inotify.max_user_watches=524288
  ```
- **macOS**: `fsevents`. Events are reliable, but directory renames
  may emit only a coarse parent-level event; the daemon's move-
  detection logic copes with that.
- **Windows (WSL2)**: events for files created on the Windows side
  of a bind mount are not guaranteed to reach inotify inside WSL.
  Either keep your dev-root entirely inside the Linux filesystem, or
  run Docker natively on a Linux host for production workloads.

If watchers look wrong, set `WQM_LOG_LEVEL=DEBUG` in `.env` and look
for `debouncer` or `watch` messages in `docker compose logs memexd`.
