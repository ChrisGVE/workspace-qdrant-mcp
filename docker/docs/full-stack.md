# Full-stack deployment (main-docker integration)

`docker/compose/full-stack.yml` is an overlay that attaches memexd and the MCP
server to a running main-docker stack. It does not start Qdrant, Prometheus,
Grafana, or the otel-collector -- those services are owned by main-docker.

## How the overlay works

The file defines two services (`memexd` and `mcp`) and declares one external
network:

```yaml
networks:
  main:
    external: true
    name: main-docker_default
```

Both services join `main-docker_default`, so they can reach other containers by
service name (e.g. `qdrant`, `prometheus`). Ports are **not** published to the
host -- Prometheus scrapes the metrics endpoints over the shared network.

## Prerequisites

1. main-docker stack is running and healthy:

   ```bash
   docker compose -f $HOME/dev/tools/main-docker/docker-compose.yml up -d
   ```

2. The external network `main-docker_default` exists. It is created automatically
   the first time main-docker starts. Verify:

   ```bash
   docker network ls | grep main-docker
   ```

3. `docker/.env` is configured. In full-stack mode `QDRANT_URL` must use the
   main-docker service name:

   ```bash
   cp docker/.env.example docker/.env
   # Set:
   QDRANT_URL=http://qdrant:6333
   ```

## Step-by-step setup

### 1. Prepare environment file

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env`:

```bash
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=                     # empty unless Qdrant is secured
WQM_LOG_LEVEL=INFO
MCP_HTTP_TOKEN=<openssl rand -hex 32>   # required -- mcp runs Streamable HTTP
```

Other variables use the defaults defined in `.env.example` and are safe to leave
unchanged for a first run.

### 2. Generate the mount override

memexd reads host directories declared in `config.yaml`'s `mounts:` section
(spec 16 §5). The mappings are rendered into a Compose override file that
layers on top of `full-stack.yml`. Run this whenever you edit `mounts:` or
move between machines:

```bash
wqm docker generate-compose                 # write docker-compose.override.yaml
wqm docker generate-compose --check         # exit 1 if config has drifted
wqm docker generate-compose --clean         # delete the override
```

The override:

- emits one `volumes:` entry per declared mount,
- bind-mounts your `config.yaml` read-only at `/etc/wqm/config.yaml`,
- bind-mounts `~/.local/share/workspace-qdrant` and `~/.local/share/qdrant`
  to their canonical container locations (spec §9.2),
- publishes `127.0.0.1:7799:7799` so the memexd control-port lock (spec
  §10.1) arbitrates correctly between host and Docker daemons,
- embeds a `# wqm-config-hash:` header so the entrypoint and `--check`
  catch stale overrides.

`network_mode: none` or any custom network that omits the control-port
publish is rejected at generation time -- see spec §10.1.

### 3. Start the overlay

```bash
docker compose \
  -f docker/compose/full-stack.yml \
  -f docker-compose.override.yaml \
  --env-file docker/.env up -d
```

### 4. Wait for memexd to pass its health check

```bash
docker compose -f docker/compose/full-stack.yml ps
```

The `mcp` service will not start until `memexd` reports `healthy`. The health
check polls `http://localhost:6337/health` every 30 seconds with a 30-second
start period.

### 5. Merge Prometheus scrape targets

Prometheus in main-docker reads its config from
`$HOME/dev/tools/main-docker/prometheus.yml`. Add the workspace-qdrant
scrape jobs to that file.

Copy the three job blocks from `docker/prometheus/prometheus.yml` -- `memexd`,
`mcp`, and (optionally) `otel-collector` -- into the `scrape_configs:` section
of main-docker's prometheus.yml.

The `mcp` job needs a bearer token: the MCP `/metrics` endpoint binds
non-loopback and rejects unauthenticated scrapes (this stack reuses
`MCP_HTTP_TOKEN` as `MCP_METRICS_TOKEN`). Write the token to a file readable
by main-docker's prometheus container, mount it (e.g. at
`/etc/prometheus/mcp_token`), and keep the job's `authorization` block:

```yaml
authorization:
  type: Bearer
  credentials_file: /etc/prometheus/mcp_token
```

Then reload Prometheus without restarting it:

```bash
curl -X POST http://localhost:9090/-/reload
```

The `memexd` job scrapes `memexd:6337` (no auth) and the `mcp` job scrapes
`workspace-qdrant-mcp:9092` (bearer). Both container names are reachable over
`main-docker_default`.

### 6. Import Grafana dashboards

If Grafana is not already provisioned from this project:

1. Open Grafana at `http://localhost:3000`.
2. Navigate to **Dashboards -> Import**.
3. Upload each JSON file from `docker/grafana/dashboards/`:
   - `system-overview.json` -- service up/down, queue depth, error events
   - `memexd.json` -- daemon queue, latency, sessions, watch errors
   - `claude-mcp.json` -- MCP tool rates, durations, sessions, fallbacks
   - `qdrant.json` -- collections, vector counts, REST/gRPC latency

Alternatively, configure provisioning so Grafana loads dashboards automatically
on startup. See `docker/grafana/provisioning/` for the datasource and dashboard
provider YAML.

### 7. Verify metrics are flowing

```bash
# From within a container on the same network, or via main-docker's Prometheus:
curl http://localhost:9090/api/v1/query?query=memexd_uptime_seconds
```

Or open Grafana -> **WQM -- System Overview** and check that `memexd` and
`MCP Server` show **UP**.

## Ports (full-stack mode)

No ports are published to the host. Container-internal ports:

| Container | Internal port | Protocol | Consumer |
|---|---|---|---|
| `memexd` | 50051 | gRPC | `workspace-qdrant-mcp` |
| `memexd` | 6337 | HTTP | Prometheus scrape job `memexd` |
| `workspace-qdrant-mcp` | 6335 | HTTP | MCP clients on the shared network (`/mcp`, bearer auth) |
| `workspace-qdrant-mcp` | 9092 | HTTP | Prometheus scrape job `mcp` (bearer auth) |

## Stopping

```bash
docker compose -f docker/compose/full-stack.yml --env-file docker/.env down
```

This stops only the workspace-qdrant services. The main-docker stack is
unaffected.

## Troubleshooting

**`network main-docker_default not found`**  
Start main-docker first: `docker compose -f .../main-docker/docker-compose.yml up -d`.

**`memexd` stays unhealthy**  
Check logs: `docker logs memexd`. Common cause: `QDRANT_URL` is wrong or Qdrant
is not yet started in main-docker.

**Prometheus shows `memexd` as DOWN after reload**  
Confirm the container is on the `main-docker_default` network:
`docker inspect memexd | grep -A5 Networks`. Confirm the scrape target matches
the container name (`memexd:6337`).

**MCP server never starts**  
It waits for `memexd` healthy. Check `docker compose -f docker/compose/full-stack.yml ps`.
If memexd health check fails, resolve that first.

_workspace-qdrant-mcp v0.1.3 -- documentation updated 2026-04-18_
