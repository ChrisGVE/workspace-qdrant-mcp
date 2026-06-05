# Standalone containers

Run individual containers without Docker Compose. Use this when you want a
single service and do not need the full compose orchestration.

## Standalone memexd

Start the daemon only. Qdrant must be reachable at `QDRANT_URL`.

```bash
docker run -d \
  --name memexd \
  --restart unless-stopped \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e QDRANT_API_KEY= \
  -e WQM_LOG_LEVEL=INFO \
  -v "${HOME}/.local/share/workspace-qdrant:/home/memexd/.local/share/workspace-qdrant" \
  -v "${HOME}/.config/workspace-qdrant:/home/memexd/.config/workspace-qdrant" \
  -v "${HOME}/dev:/home/memexd/dev:ro" \
  -p 50051:50051 \
  -p 6337:6337 \
  chrisgve/memexd:v0.1.3
```

Verify:

```bash
curl -s http://localhost:6337/health
curl -s http://localhost:6337/metrics | head -10
```

## Standalone MCP server

Start the MCP server only. A running memexd instance must be reachable at
`WQM_DAEMON_ENDPOINT`. This can be the standalone-memexd container above, or
any other memexd instance on the network.

A containerized MCP server always runs http mode: stdio mode needs a client
attached to stdin, which a detached container lacks. Clients connect with
`Authorization: Bearer $MCP_HTTP_TOKEN` at `http://<host>:6335/mcp`.

```bash
docker run -d \
  --name workspace-qdrant-mcp \
  --restart unless-stopped \
  -e WQM_DAEMON_ENDPOINT=host.docker.internal:50051 \
  -e WQM_LOG_LEVEL=INFO \
  -e MCP_SERVER_MODE=http \
  -e MCP_HTTP_HOST=0.0.0.0 \
  -e MCP_HTTP_TOKEN="$(openssl rand -hex 32)" \
  -e MCP_METRICS_HOST=0.0.0.0 \
  -e MCP_METRICS_TOKEN="<same-or-dedicated-token>" \
  -v "${HOME}/.local/share/workspace-qdrant:/home/wqm/.local/share/workspace-qdrant" \
  -v "${HOME}/.config/workspace-qdrant:/home/wqm/.config/workspace-qdrant" \
  -p 6335:6335 \
  -p 9092:9092 \
  chrisgve/workspace-qdrant-mcp:v0.1.3
```

## Running both standalone containers

The simplest way to connect them is a shared user-defined bridge network:

```bash
# Create a network once
docker network create wqm-net

# Start memexd on the network
docker run -d \
  --name memexd \
  --network wqm-net \
  --restart unless-stopped \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e WQM_LOG_LEVEL=INFO \
  -v "${HOME}/.local/share/workspace-qdrant:/home/memexd/.local/share/workspace-qdrant" \
  -v "${HOME}/.config/workspace-qdrant:/home/memexd/.config/workspace-qdrant" \
  -v "${HOME}/dev:/home/memexd/dev:ro" \
  -p 50051:50051 \
  -p 6337:6337 \
  chrisgve/memexd:v0.1.3

# Start MCP server on the same network, referencing memexd by container name
docker run -d \
  --name workspace-qdrant-mcp \
  --network wqm-net \
  --restart unless-stopped \
  -e WQM_DAEMON_ENDPOINT=memexd:50051 \
  -e WQM_LOG_LEVEL=INFO \
  -e MCP_SERVER_MODE=http \
  -e MCP_HTTP_HOST=0.0.0.0 \
  -e MCP_HTTP_TOKEN="$(openssl rand -hex 32)" \
  -e MCP_METRICS_HOST=0.0.0.0 \
  -e MCP_METRICS_TOKEN="<same-or-dedicated-token>" \
  -v "${HOME}/.local/share/workspace-qdrant:/home/wqm/.local/share/workspace-qdrant" \
  -v "${HOME}/.config/workspace-qdrant:/home/wqm/.config/workspace-qdrant" \
  -p 6335:6335 \
  -p 9092:9092 \
  chrisgve/workspace-qdrant-mcp:v0.1.3
```

On the same user-defined network, container names resolve as hostnames, so
`http://memexd:50051` works directly.

## Compose-file equivalents

The `docker run` commands above are documented verbatim in the compose files for
reference:

- `docker/compose/standalone-memexd.yml` — memexd only
- `docker/compose/standalone-mcp.yml` — MCP server only

Use the compose files when you want `--env-file` support, auto-restart, and
health check wiring without repeating flags.

```bash
docker compose -f docker/compose/standalone-memexd.yml --env-file docker/.env up -d
docker compose -f docker/compose/standalone-mcp.yml   --env-file docker/.env up -d
```

## Ports

| Service | Port | Protocol | Purpose |
|---|---|---|---|
| `memexd` | 50051 | gRPC | MCP server connection, wqm CLI |
| `memexd` | 6337 | HTTP | `/health` health check, `/metrics` Prometheus |
| `workspace-qdrant-mcp` | 6335 | HTTP | MCP Streamable HTTP (`/mcp`, `/healthz`) |
| `workspace-qdrant-mcp` | 9092 | HTTP | `/metrics` Prometheus (bearer-authenticated) |

The metrics endpoint exists only in http mode and, bound non-loopback,
requires `Authorization: Bearer $MCP_METRICS_TOKEN` on every scrape.

## Data persistence

Both containers write to paths inside the mounted host directories. If you
remove a container and re-create it with the same volume mounts, state is
preserved.

To start fresh, remove the host directories:

```bash
rm -rf "${HOME}/.local/share/workspace-qdrant"
```

This also removes the SQLite database used by memexd. Qdrant data stored
remotely is unaffected.

_workspace-qdrant-mcp v0.1.3 — documentation updated 2026-04-18_
