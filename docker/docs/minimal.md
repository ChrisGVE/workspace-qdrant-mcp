# Minimal deployment

Runs memexd and the MCP server. Qdrant must already be reachable — it is not
started by this compose file.

## When to use

- You run Qdrant on the Docker host, in another container, or in the cloud.
- You do not run main-docker.
- You want to add observability later (or not at all).

## Prerequisites

- Docker Engine 24+ (Linux) or Docker Desktop 4+ (macOS/Windows).
- Qdrant reachable at a known URL. For a local instance use
  `http://host.docker.internal:6333` (macOS/Windows) or
  `http://172.17.0.1:6333` (Linux bridge default).

## Setup

### 1. Copy and edit the environment file

```bash
cp docker/.env.example docker/.env
```

Open `docker/.env` and set `QDRANT_URL` to the address of your Qdrant instance:

```bash
QDRANT_URL=http://host.docker.internal:6333
```

Leave `QDRANT_API_KEY` empty for unauthenticated local Qdrant. Set it for
Qdrant Cloud or a secured instance.

### 2. Start the stack

```bash
docker compose -f docker/compose/minimal.yml --env-file docker/.env up -d
```

### 3. Verify

```bash
# memexd health endpoint
curl -s http://localhost:6337/health

# memexd Prometheus metrics
curl -s http://localhost:6337/metrics | head -20
```

Both should respond without error. The MCP server does not expose an HTTP
health endpoint in `stdio` mode; verify it via the MCP client logs or
`docker logs workspace-qdrant-mcp`.

## Add self-contained observability

Layer `observability.yml` on top to get Prometheus, Grafana, and the
OpenTelemetry Collector without main-docker:

```bash
docker compose \
  -f docker/compose/minimal.yml \
  -f docker/compose/observability.yml \
  --env-file docker/.env \
  up -d
```

Services started by this combined stack:

| Container | Default port | Purpose |
|---|---|---|
| `memexd` | 50051 (gRPC), 6337 (metrics) | Rust daemon |
| `workspace-qdrant-mcp` | 9092 (metrics) | MCP server |
| `wqm-prometheus` | 9090 | Metrics collection |
| `wqm-grafana` | 3000 | Dashboards |
| `wqm-otel-collector` | 4317 (gRPC), 4318 (HTTP) | OTLP receiver |

Access Grafana at `http://localhost:3000`. Default credentials: `admin` / `admin`.
Change the password on first login.

### Reload Prometheus config (if needed)

```bash
curl -X POST http://localhost:9090/-/reload
```

## Environment variables reference

All variables are defined in `docker/.env.example` with defaults. Key variables:

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://host.docker.internal:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | _(empty)_ | API key for authenticated Qdrant |
| `WQM_LOG_LEVEL` | `INFO` | Log level: `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE` |
| `MEMEXD_GRPC_PORT` | `50051` | Host port for memexd gRPC |
| `MEMEXD_METRICS_PORT` | `6337` | Host port for memexd metrics/health |
| `MEMEXD_GRPC_URL` | `http://memexd:50051` | gRPC URL the MCP server uses to reach memexd |
| `MCP_SERVER_MODE` | `stdio` | Transport: `stdio` (Claude Code) or `http` |
| `MCP_METRICS_PORT` | `9092` | Host port for MCP server metrics |
| `PROMETHEUS_PORT` | `9090` | Prometheus host port (observability overlay) |
| `GRAFANA_PORT` | `3000` | Grafana host port (observability overlay) |
| `OTEL_GRPC_PORT` | `4317` | OTLP gRPC port (observability overlay) |
| `OTEL_HTTP_PORT` | `4318` | OTLP HTTP port (observability overlay) |

## Volume mounts

Both services mount host directories into the container so data persists across
restarts and the daemon can watch the developer's working tree:

| Host path | Container path | Notes |
|---|---|---|
| `~/.workspace-qdrant` | `/home/memexd/.workspace-qdrant` | SQLite database |
| `~/.config/workspace-qdrant` | `/home/memexd/.config/workspace-qdrant` | Runtime config |
| `~/dev` | `/home/memexd/dev` | Source tree (read-only) |
| `~/.workspace-qdrant` | `/home/node/.workspace-qdrant` | MCP server state |
| `~/.config/workspace-qdrant` | `/home/node/.config/workspace-qdrant` | MCP server config |

The `~/dev` mount is read-only. The daemon watches files under this path and
indexes them. Adjust the path in `docker/compose/minimal.yml` if your code lives
elsewhere.

## Stopping

```bash
docker compose -f docker/compose/minimal.yml --env-file docker/.env down
```

Add `--volumes` to also remove named volumes (Prometheus/Grafana data if using
the observability overlay).

_workspace-qdrant-mcp v0.1.3 — documentation updated 2026-04-18_
