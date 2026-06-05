# workspace-qdrant-mcp — Docker Deployment

Compose configurations cover the common deployment scenarios. Choose the one
that matches your existing infrastructure. All containerized MCP deployments
run Streamable HTTP and require `MCP_HTTP_TOKEN` in `docker/.env`.

## Deployment modes

| Mode | Use case | Compose file | Qdrant | Observability |
|---|---|---|---|---|
| **Reference** | New deployment, nothing running yet | `docker/compose/reference.yml` | In-stack | Optional overlay |
| **Minimal** | You already run Qdrant | `docker/compose/minimal.yml` | External | None |
| **Minimal + observability** | Self-contained stack, no main-docker | `minimal.yml` + `observability.yml` | External | Self-hosted (Prometheus, Grafana, otel-collector, Loki) |
| **Full-stack** | Integrated with main-docker | `docker/compose/full-stack.yml` | `main-docker` | Owned by `main-docker` |
| **Standalone** | Single container | `docker run` one-liners | External | None |

## Decision guide

```text
Do you run the main-docker stack?
  Yes → Use full-stack.md
  No  → Do you already have Qdrant running?
          Yes → Use minimal.md (add observability.yml if wanted)
          No  → Use reference.yml (see ../compose/README.md)
```

Need just the daemon or just the MCP server without Docker Compose?  
Use the `docker run` one-liners in [standalone.md](standalone.md).

## Quick start (minimal)

```bash
cp docker/.env.example docker/.env
# Edit QDRANT_URL to point at your Qdrant instance and set MCP_HTTP_TOKEN
# (generate with: openssl rand -hex 32)
docker compose -f docker/compose/minimal.yml --env-file docker/.env up -d
```

Verify:

```bash
curl -s http://localhost:6337/health   # memexd health
curl -s http://localhost:6337/metrics  # memexd Prometheus metrics
curl -s http://localhost:6335/healthz  # MCP server liveness (unauthenticated)
```

## File reference

| File | Purpose |
|---|---|
| `docker/compose/reference.yml` | Qdrant + memexd + MCP server (recommended) |
| `docker/compose/minimal.yml` | memexd + MCP server, external Qdrant |
| `docker/compose/observability.yml` | Prometheus + Grafana + otel-collector + Loki overlay |
| `docker/compose/full-stack.yml` | Overlay that attaches to main-docker network |
| `docker/compose/standalone-memexd.yml` | Daemon only |
| `docker/compose/standalone-mcp.yml` | MCP server only |
| `docker/.env.example` | All environment variables with defaults |
| `docker/prometheus/prometheus.yml` | Prometheus scrape config (4 jobs; mcp job bearer-authenticated) |
| `docker/prometheus/alerts.yml` | Alerting rules |
| `docker/grafana/dashboards/*.json` | Pre-built Grafana dashboards |
| `docker/grafana/provisioning/` | Grafana auto-provisioning |
| `docker/otel/otel-collector-config.yml` | OTLP collector (memexd traces/metrics) |

## Detailed guides

- [minimal.md](minimal.md) — minimal and minimal + observability setup
- [full-stack.md](full-stack.md) — integration with main-docker
- [standalone.md](standalone.md) — single-container docker run commands
- [telemetry.md](telemetry.md) — metrics reference and Prometheus queries
- [logging.md](logging.md) — structured logging, Loki aggregation, LogQL queries
- [dashboards.md](dashboards.md) — Grafana dashboard panel catalog

_workspace-qdrant-mcp v0.1.3 — documentation updated 2026-04-18_
