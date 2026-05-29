# Compose deployments

The canonical one-file entrypoint lives at the repository root as
`docker-compose.yml`. The files in `docker/compose/` are reference
overlays and examples; they are useful for tests and special setups,
but they are not the primary way we start the stack.

All compose files in this repo build images locally from this checkout.
Use `--build` with `docker compose up` so TypeScript and Rust changes
are included in the container image.

| File | Status |
|------|---------|
| `reference.yml` | Reference stack for Qdrant, memexd, MCP, Prometheus, Grafana, Loki, and Promtail. |
| `reference.tls.yml` | TLS overlay for `reference.yml`. |
| `minimal.yml` | Legacy minimal stack for host-local Qdrant. |
| `standalone-memexd.yml` | Single-service memexd example. |
| `standalone-mcp.yml` | Single-service MCP example. |
| `qdrant.yml` | Qdrant-only example. |
| `observability.yml` | Observability overlay for local Prometheus/Grafana/OTel. |

## Quickstart

Prefer the root `docker-compose.yml` for day-to-day use:

```bash
docker compose --env-file docker/.env up -d --build
```

1. **Clone + prep state directory.**

   ```bash
   cd /path/to/workspace-qdrant-mcp
   mkdir -p state
   ```

2. **Populate `docker/.env`.** Copy `.env.example` and fill in:

   ```bash
   # Generate a token (run this in your shell):
   openssl rand -hex 32
   ```

   Paste the output as a literal value in `docker/.env`:

   ```env
   # Required
   MCP_HTTP_TOKEN=<paste-openssl-output-here>
   WQM_DEV_ROOT=/Users/your-user/dev
   WQM_VERSION=latest

   # Optional
   WQM_STATE_DIR=./state
   WQM_CONFIG_FILE=/absolute/path/to/wqm-config.yaml
   ```

   > **Warning:** Docker Compose `.env` files do not evaluate `$()` command
   > substitution. You must paste the literal hex value — do not write
   > `MCP_HTTP_TOKEN=$(openssl rand -hex 32)` inside `.env`.

   See the header comment in `reference.yml` for the full list of variables.
   Without `MCP_HTTP_TOKEN` or `WQM_DEV_ROOT` compose refuses to start
   with a clear error message.

3. **Copy the example config** if you want to override defaults:

   ```bash
   cp docker/config.example.yaml ~/.config/wqm/config.yaml
   # edit include_paths etc.
   ```

   Point `WQM_CONFIG_FILE` at that host path. Leave it unset to run with the
   built-in defaults.

4. **Launch the canonical stack.**

   ```bash
   docker compose \
     --env-file docker/.env \
     up -d --build
   ```

5. **Verify.**

   ```bash
   docker compose \
     --env-file docker/.env \
     ps

   # Probe the MCP HTTP endpoint (bearer auth required on /mcp; /healthz is open).
   curl http://127.0.0.1:6335/healthz
   curl -H "Authorization: Bearer $MCP_HTTP_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}' \
        http://127.0.0.1:6335/mcp

   curl -H "Authorization: Bearer $MCP_HTTP_TOKEN" \
        http://127.0.0.1:9092/metrics | head
   ```

## Add local observability (Prometheus + Grafana + OTLP)

Compose in `observability.yml` to attach a self-hosted Prometheus +
Grafana + OpenTelemetry Collector to the canonical stack. The overlay
joins `workspace-network`, so Prometheus reaches the memexd (`:9091`)
and MCP (`:9092`) metrics endpoints by service DNS without publishing
extra host ports.

Extra `.env` values (all optional):

```
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change-me
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

Launch:

```bash
docker compose \
  --env-file docker/.env \
  -f docker-compose.yml \
  -f docker/compose/observability.yml up -d --build
```

Prometheus scrape targets live in `docker/prometheus/prometheus.yml`
(already wired to `memexd`, `mcp`, `qdrant`, `otel-collector`).
The MCP scrape reuses `MCP_HTTP_TOKEN`; the Prometheus container writes it
to an in-memory secret file before launching.
Grafana picks up `docker/grafana/provisioning/` on first boot.

## Add Let's Encrypt TLS

Extra `.env` values:

```
MCP_PUBLIC_HOSTNAME=mcp.example.com
MCP_TLS_EMAIL=you@example.com
```

The FQDN must resolve to the host running compose. Open TCP 80 + 443
(Let's Encrypt HTTP-01 challenge happens over port 80).

```bash
docker compose \
  --env-file docker/.env \
  -f docker/compose/reference.yml \
  -f docker/compose/reference.tls.yml up -d --build
```

Caddy takes over ports 80 and 443; the MCP server no longer exposes 6335 to
the host. Traffic path:

```
client → https://$MCP_PUBLIC_HOSTNAME → caddy (:443) → mcp (:6335)
```

First boot provisions a certificate automatically; subsequent boots reuse
the certificate data persisted in `${WQM_STATE_DIR}/caddy/`.

## Path transparency (`WQM_DEV_ROOT`)

The daemon stores **absolute paths** for every watched folder in its SQLite
state. The host `wqm` CLI queries that state over gRPC and passes paths
back to the daemon. If the daemon's filesystem view does not match the
host's, `wqm project activate $(pwd)` cannot find a matching watch folder.

The reference compose solves this with an identical-path bind mount:

```yaml
volumes:
  - ${WQM_DEV_ROOT}:${WQM_DEV_ROOT}
```

Set `WQM_DEV_ROOT` to the host directory whose subtrees you want indexed
(e.g. `/Users/your-user/dev`). Every project rooted below that path is
reachable by the daemon at the same absolute path. Without it, activation
from the host would fail silently.

## Ports

| Port | Service | Purpose |
|------|---------|---------|
| 6333 | qdrant | Qdrant REST |
| 6334 | qdrant | Qdrant gRPC |
| 50051 | memexd | memexd gRPC (`wqm` CLI connection) |
| 9091 | memexd | memexd Prometheus metrics + `/health` |
| 6335 | mcp | MCP Streamable HTTP (`/mcp`, `/healthz`) |
| 9092 | mcp | MCP Prometheus metrics |
| 80, 443 | caddy (TLS overlay only) | Reverse proxy |
| 9090 | prometheus (observability overlay) | Prometheus UI |
| 3000 | grafana (observability overlay) | Grafana UI |
| 4317 | otel-collector (observability overlay) | OTLP gRPC |
| 4318 | otel-collector (observability overlay) | OTLP HTTP |
