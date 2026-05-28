# workspace-qdrant-mcp — Docker Deployment

The root `docker-compose.yml` is the canonical entrypoint for day-to-day use.
The files under `docker/compose/` remain available as overlays and reference
examples, but they are not the primary path.

All of the compose-based paths build the MCP image locally from this
checkout, so use `--build` when you run them.

## Deployment modes

| Mode | Use case | Compose file | Qdrant | Observability |
|---|---|---|---|---|
| **Minimal** | You already run Qdrant and want a local MCP build | `docker/compose/minimal.yml` | External | None |
| **Minimal + observability** | Self-contained stack, no main-docker, with a local MCP build | `docker-compose.yml` + `observability.yml` | External | Self-hosted (Prometheus, Grafana, otel-collector) |
| **Standalone** | Single container, local MCP image | `docker run` one-liners | External | None |

## Decision guide

```text
Do you already have Qdrant running?
  Yes → Use minimal.md
  No  → Run minimal.yml + observability.yml (see minimal.md)
```

Need just the daemon or just the MCP server without Docker Compose?  
Use the `docker run` one-liners in [standalone.md](standalone.md).

## Quick start (canonical root stack)

```bash
cp docker/.env.example docker/.env
# Fill MCP_HTTP_TOKEN and WQM_DEV_ROOT, then start the root stack
docker compose --env-file docker/.env up -d --build
```

Verify:

```bash
curl -s http://localhost:9091/health   # memexd health
curl -s http://localhost:9091/metrics  # Prometheus metrics endpoint
```

## File reference

| File | Purpose |
|---|---|
| `docker/compose/minimal.yml` | memexd + MCP server, external Qdrant |
| `docker/compose/observability.yml` | Prometheus + Grafana + otel-collector overlay |
| `docker/compose/standalone-memexd.yml` | Daemon only |
| `docker/compose/standalone-mcp.yml` | MCP server only |
| `docker/.env.example` | All environment variables with defaults |
| `docker/prometheus/prometheus.yml` | Prometheus scrape config (4 jobs) |
| `docker/prometheus/alerts.yml` | 6 alerting rules |
| `docker/grafana/dashboards/*.json` | 6 pre-built Grafana dashboards |
| `docker/grafana/provisioning/` | Grafana auto-provisioning |
| `docker/otel/otel-collector-config.yml` | OTLP → Prometheus bridge |

## Detailed guides

- [minimal.md](minimal.md) — minimal and minimal + observability setup
- [standalone.md](standalone.md) — single-container docker run commands
- [telemetry.md](telemetry.md) — metrics reference and Prometheus queries
- [dashboards.md](dashboards.md) — Grafana dashboard panel catalog

_workspace-qdrant-mcp v0.1.3 — documentation updated 2026-04-18_
