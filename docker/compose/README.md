# Compose deployments

The files in this directory are ready-to-run Docker Compose stacks for
different deployment shapes. Pick the one that matches how you want to run
the system.

| File | Purpose |
|------|---------|
| `reference.yml` | Self-contained three-service stack (Qdrant + memexd + MCP over HTTP). Start here for a new deployment. |
| `reference.tls.yml` | Overlay for `reference.yml` — adds a Caddy reverse proxy with automatic Let's Encrypt TLS. Requires a public FQDN. |
| `minimal.yml` | Host-local stack that reaches a Qdrant running on the Docker host via `host.docker.internal`. Useful when Qdrant is already installed outside Docker. |
| `standalone-memexd.yml` | memexd only. Run this alongside an existing Qdrant + MCP. |
| `standalone-mcp.yml` | MCP server only. Run this alongside existing memexd + Qdrant. |
| `qdrant.yml` | Qdrant only. |
| `full-stack.yml` | Chris-specific: attaches memexd + MCP to the author's local `main-docker` observability stack. Not intended as a general template. |
| `observability.yml` | Prometheus + Grafana + OTLP collector overlay for local debugging. |

## Quickstart (reference stack)

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

4. **Launch.**

   ```bash
   docker compose \
     --env-file docker/.env \
     -f docker/compose/reference.yml up -d
   ```

5. **Verify.**

   ```bash
   docker compose \
     --env-file docker/.env \
     -f docker/compose/reference.yml ps

   # Probe the MCP HTTP endpoint (bearer auth required on /mcp; /healthz is open).
   curl http://127.0.0.1:6335/healthz
   curl -H "Authorization: Bearer $MCP_HTTP_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}' \
        http://127.0.0.1:6335/mcp
   ```

## Add local observability (Prometheus + Grafana + OTLP)

Compose in `observability.yml` to attach a self-hosted Prometheus +
Grafana + OpenTelemetry Collector to any of the primary stacks. The
overlay joins `workspace-network`, so Prometheus reaches the memexd
(`:6337`) and MCP (`:9092`) metrics endpoints by service DNS without
publishing extra host ports.

The MCP `/metrics` endpoint binds non-loopback inside the container and
therefore requires a bearer token (the local stack reuses
`MCP_HTTP_TOKEN`; hosted deployments should issue a dedicated secret).
Materialize the token file Prometheus mounts **before** `up`:

```bash
mkdir -p docker/secrets
printf '%s' "<your MCP_HTTP_TOKEN value>" > docker/secrets/mcp_token
```

`docker/secrets/` is gitignored; override the file location with
`MCP_METRICS_TOKEN_FILE` in `.env` if you keep it elsewhere. memexd
metrics (`:6337`) are unauthenticated and need no token.

Extra `.env` values (all optional):

```
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change-me
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
MCP_METRICS_TOKEN_FILE=../secrets/mcp_token
```

Launch:

```bash
docker compose \
  --env-file docker/.env \
  -f docker/compose/reference.yml \
  -f docker/compose/observability.yml up -d
```

Prometheus scrape targets live in `docker/prometheus/prometheus.yml`
(already wired to `memexd`, `mcp` — bearer auth via
`/etc/prometheus/mcp_token` — `qdrant`, and `otel-collector`).
Grafana picks up `docker/grafana/provisioning/` on first boot.

Already running `main-docker`? Use `full-stack.yml` instead — it
attaches memexd + mcp to the main-docker observability stack and
omits Prometheus/Grafana here.

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
  -f docker/compose/reference.tls.yml up -d
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
| 6337 | memexd | memexd Prometheus metrics + `/health` |
| 6335 | mcp | MCP Streamable HTTP (`/mcp`, `/healthz`) |
| 9092 | mcp | MCP Prometheus metrics |
| 80, 443 | caddy (TLS overlay only) | Reverse proxy |
| 9090 | prometheus (observability overlay) | Prometheus UI |
| 3000 | grafana (observability overlay) | Grafana UI |
| 4317 | otel-collector (observability overlay) | OTLP gRPC |
| 4318 | otel-collector (observability overlay) | OTLP HTTP |
