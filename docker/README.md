# Docker deployment — workspace-qdrant-mcp (Rust stack)

Container packaging for the two Rust services:

| Image | Dockerfile | Contents |
|---|---|---|
| `chrisgve/memexd` | `docker/Dockerfile.memexd` | `memexd` daemon — file watching, processing, embeddings (ONNX), single writer to Qdrant + SQLite |
| `chrisgve/workspace-qdrant-mcp` | `docker/Dockerfile.mcp` | Rust MCP server — read-only gRPC/Qdrant client, Streamable HTTP transport. No ONNX, much lighter |

Every compose topology runs the MCP server in **http mode** (`MCP_SERVER_MODE=http`):
stdio mode needs a client attached to stdin, which a detached container lacks, and
the Prometheus `/metrics` endpoint only exists in http mode. stdio remains the
default for host-launched binaries (Claude Code spawning `workspace-qdrant-mcp`
directly).

## Quick start

```bash
cp docker/.env.example docker/.env
# Required: MCP_HTTP_TOKEN (openssl rand -hex 32), WQM_DEV_ROOT (reference stack)
docker compose --env-file docker/.env -f docker/compose/reference.yml up -d
```

See [compose/README.md](compose/README.md) for the full quickstart (token
generation, config overrides, TLS overlay, observability overlay) and
[docs/README.md](docs/README.md) for the deployment decision guide.

## Directory map

| Path | Purpose | Tracked |
|---|---|---|
| `Dockerfile.memexd` | Daemon image (multi-stage, multi-arch, ONNX static) | yes |
| `Dockerfile.mcp` | MCP server image (multi-stage, multi-arch) | yes |
| `memexd-entrypoint.sh` | Daemon entrypoint (config-hash check, signal handling) | yes |
| `compose/` | Ready-to-run topologies — see [compose/README.md](compose/README.md) | yes |
| `docs/` | Deployment guides, telemetry/logging/dashboard references | yes |
| `.env.example` | All environment variables with defaults and comments | yes |
| `config.example.yaml` | Example daemon `config.yaml` | yes |
| `prometheus/` | Scrape config (4 jobs; mcp job bearer-authenticated), alert + recording rules | yes |
| `grafana/` | Dashboards + auto-provisioning | yes |
| `otel/` | OTLP collector config (receives memexd traces/metrics) | yes |
| `loki/`, `promtail/` | Log aggregation for the observability overlay | yes |
| `caddy/` | Reverse proxy for the TLS overlay (`reference.tls.yml`) | yes |
| `secrets/` | Local-only materialized secrets (e.g. `mcp_token` for Prometheus) | **gitignored** |
| `k8s/`, `nginx/`, `integration-tests/`, `deploy.sh`, `build-and-push.sh`, `docker-compose.*.yml`, `entrypoint.sh` | Legacy Python/TS-era artifacts retained locally | **gitignored** |

## Topologies

| Compose file | Shape |
|---|---|
| `compose/reference.yml` | Qdrant + memexd + MCP (recommended single-host stack) |
| `compose/reference.tls.yml` | Overlay: Caddy + Let's Encrypt in front of the MCP HTTP endpoint |
| `compose/minimal.yml` | memexd + MCP, external Qdrant |
| `compose/observability.yml` | Overlay: Prometheus + Grafana + otel-collector + Loki/promtail |
| `compose/standalone-memexd.yml` / `compose/standalone-mcp.yml` | Single-service |
| `compose/qdrant.yml` | Qdrant only |
| `compose/full-stack.yml` | Author-specific overlay onto the local `main-docker` stack |

## Service endpoints

| Port | Service | Endpoint |
|---|---|---|
| 6333/6334 | qdrant | REST / gRPC |
| 50051 | memexd | gRPC (`wqm` CLI + MCP server) |
| 6337 | memexd | `/health`, `/metrics` (unauthenticated) |
| 6335 | mcp | Streamable HTTP — `/mcp` (bearer), `/healthz` (open) |
| 9092 | mcp | `/metrics` — bearer-authenticated when bound non-loopback |

The MCP metrics token reuses `MCP_HTTP_TOKEN` in the local stack
(`MCP_METRICS_TOKEN=${MCP_HTTP_TOKEN}` in compose); hosted deployments should
issue a dedicated secret. Prometheus reads it from a mounted file — see
[docs/telemetry.md](docs/telemetry.md).

## Building images

```bash
# MCP server (no ONNX, fast build)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.mcp -t chrisgve/workspace-qdrant-mcp:latest .

# Daemon (ONNX static link, slower)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.memexd -t chrisgve/memexd:latest .
```

Both Dockerfiles use the repository root as build context. CI builds and
smoke-tests both images (`.github/workflows/`).

## Smithery

`smithery.yaml` (repo root) deploys the MCP server container via Smithery,
built from `docker/Dockerfile.mcp`. The container is only the MCP server —
it points at your separately running memexd + Qdrant.
