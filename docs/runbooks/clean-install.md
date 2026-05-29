# Clean Install — Zero to First Search

This runbook walks a brand-new user from an empty machine to a first
successful search. Follow the steps in order. Each command has been
cross-checked against `wqm 0.1.3`, the root `docker-compose.yml`, and the
docs under `docs/reference/`.

There are two supported paths:

- **Path A — Native binaries (Homebrew / installer).** Run `wqm` + `memexd`
  directly on the host, point them at a local Qdrant container. Best for
  macOS/Linux and the simplest onboarding.
- **Path B — Full Docker stack.** Run Qdrant, the daemon, and the MCP server
  (plus observability) from the repo's `docker-compose.yml`. Best when you
  want everything containerized, including on Windows.

Pick one path and follow it through to the end. The "First search" and
"Troubleshooting" sections at the bottom apply to both.

---

## 1. Prerequisites

| Requirement | Why | Path |
|---|---|---|
| **Docker / Docker Desktop** | Runs Qdrant (and, for Path B, the daemon + MCP server). | A and B |
| **A C compiler/toolchain** | Tree-sitter grammars ship as C source and are compiled locally on first use. macOS: `xcode-select --install`. Debian/Ubuntu: `apt install build-essential`. Windows: Visual Studio Build Tools with the C++ workload. | A and B |
| **Homebrew** (macOS/Linux) | Easiest way to install the `wqm` CLI and `memexd` daemon. | A |
| **`git`** | Project detection is Git-based — the daemon detects and scopes projects by repository. | A and B |

> A first run will compile Tree-sitter grammars and (with the default
> FastEmbed provider) download an embedding model. The first indexing pass is
> therefore slower than later ones — this is expected, not a hang.

---

## Path A — Native binaries

### A.2. Bring up Qdrant + the daemon

**Start Qdrant** (vector database) as a local container:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Qdrant exposes REST on `6333` and gRPC on `6334`. Confirm it is up:

```bash
curl -s http://localhost:6333/healthz
```

**Install `wqm` + `memexd`:**

```bash
# Option 1 — Homebrew (macOS & Linux, recommended)
brew install ChrisGVE/tap/workspace-qdrant

# Option 2 — pre-built binaries
#   macOS / Linux
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash
#   Windows (PowerShell)
irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
```

The installer places `wqm` and `memexd` on your PATH (`~/.local/bin` on
Linux/macOS, `%LOCALAPPDATA%\wqm\bin` on Windows). Verify:

```bash
wqm --version
```

**Start the daemon.** `memexd` watches files, generates embeddings, and serves
the gRPC API that both the CLI and the MCP server connect to:

```bash
wqm service install   # one-time: register memexd as a system service
wqm service start     # start the background daemon
wqm service status    # confirm it is running
```

By default the daemon connects to Qdrant at `http://localhost:6334` (gRPC). If
your Qdrant is elsewhere or secured, set `QDRANT_URL` (and `QDRANT_API_KEY` for
Qdrant Cloud) before starting — see
[Configuration](../reference/configuration.md).

Confirm the daemon and Qdrant are both healthy:

```bash
wqm status health
```

This reports daemon connectivity, Qdrant availability, collection sizes, and
active projects. You want it to read healthy before continuing.

→ Now jump to **3. Configure the MCP server**.

---

## Path B — Full Docker stack

The root `docker-compose.yml` is the canonical entrypoint. It builds the
daemon and MCP server from your checkout and starts them alongside Qdrant and
the observability stack (Prometheus, Grafana, Loki, OTel). The MCP server is
served over Streamable HTTP on port `6335`.

### B.2. Configure and bring up the stack

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp
cp docker/.env.example docker/.env
```

Edit `docker/.env` and set the two required variables for the root stack:

```bash
# Bearer token for the MCP HTTP endpoint (and reused by Prometheus for /metrics)
MCP_HTTP_TOKEN=<generate with: openssl rand -hex 32>

# Absolute host path to the directory containing the repos you want indexed.
# It is bind-mounted into the daemon and MCP containers at the SAME path, so
# recorded file paths stay identical between host and container.
WQM_DEV_ROOT=/absolute/path/to/your/dev/root
```

The stack defaults to the **FastEmbed** embedding provider
(`WQM_EMBEDDING_PROVIDER=fastembed`) and caches the model under
`./.fastembed_cache`, so no API key is required for a local run.

Bring everything up:

```bash
docker compose --env-file docker/.env up -d --build
```

The `mcp` service waits for `memexd` to pass its health check, which in turn
waits for Qdrant to be healthy. Watch the startup:

```bash
docker compose ps
```

Verify the daemon is healthy and Qdrant is reachable:

```bash
curl -s http://localhost:9091/health    # memexd health endpoint
curl -s http://localhost:6333/healthz    # Qdrant (published from the qdrant service)
```

For other Docker layouts (minimal, standalone, or integration with an existing
`main-docker` stack), see [`docker/docs/README.md`](../../docker/docs/README.md).

> **Qdrant corruption.** If Qdrant fails to start after an unclean shutdown,
> consult the Qdrant corruption runbook if present under `docs/runbooks/`. At
> the time of writing this repo does not yet ship one — see the *Known
> onboarding gaps* note at the end. As a stopgap, inspect the Qdrant container
> logs (`docker logs wqm-qdrant`) and the storage volume mounted at
> `${WQM_STATE_DIR:-./state}/qdrant/storage`.

→ Now continue to **3. Configure the MCP server**.

---

## 3. Configure the MCP server

> **Do not create a `.mcp.json` inside this repository.** The MCP server is
> wired up at the client level (Claude Desktop / Claude Code), not via an
> in-project config file.

### Path A clients (stdio transport)

Point your client at the bundled TypeScript MCP server entrypoint.

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "node",
      "args": ["<install-path>/src/typescript/mcp-server/dist/index.js"],
      "env": { "QDRANT_URL": "http://localhost:6333" }
    }
  }
}
```

**Claude Code:**

```bash
claude mcp add workspace-qdrant -- node <install-path>/src/typescript/mcp-server/dist/index.js
```

See the [README "Configure MCP" section](../../README.md) and
[MCP Tools Reference](../reference/mcp-tools.md) for the full client wiring and
tool list.

### Path B clients (HTTP transport)

The Docker stack serves the MCP server over Streamable HTTP at
`http://localhost:6335` (path `/mcp`), authenticated with the
`MCP_HTTP_TOKEN` you set in `docker/.env`. The compose default
`MCP_HTTP_TRUST_LOCALHOST=1` skips the bearer check for loopback peers, so a
host-side client on the same machine connects without the token. Point your
MCP client's HTTP transport at that URL; do **not** also launch a local stdio
server, or you will run two competing instances.

---

## 4. Register and index a project

The daemon detects projects from Git repositories. You point it at a repo
once; from then on it watches the directory and ingests changes automatically.

```bash
cd /path/to/your/project        # must be a Git repository
wqm project register .          # register the current directory
```

`wqm project register` accepts an explicit path and an optional label:

```bash
wqm project register /path/to/repo --name my-project
wqm project register . -y        # skip the confirmation prompt
```

Confirm registration and watch indexing progress:

```bash
wqm project list      # the project should now appear
wqm queue stats       # watch the queue drain toward 0 as files are indexed
wqm status health     # active-project count should include your repo
```

The first indexing pass compiles Tree-sitter grammars and (with FastEmbed)
loads the embedding model, so give it time. When `wqm queue stats` shows no
pending or in-progress items, indexing is complete.

> On the **Docker** path, `wqm` running on the host talks to the containerized
> daemon over gRPC. If the host `wqm` cannot reach it, point it at the daemon's
> published gRPC port with `--daemon-addr http://127.0.0.1:50051` (the compose
> default) or run the equivalent command inside the container.

---

## 5. Run your first search

Searches run against indexed content, so only run them **after** the queue has
drained (step 4).

### From the MCP `search` tool (the primary path)

The `search` tool does hybrid semantic + keyword search across your indexed
content. A minimal call from an MCP client:

```json
{
  "query": "authentication middleware",
  "scope": "project",
  "limit": 10
}
```

Each result includes an `id`, a `score` (0.0–1.0), the matched `content`, and
`metadata` (file path, language, branch, component, concept tags). Getting a
non-empty array back confirms the full path — daemon → Qdrant → MCP server —
is working. See [MCP Tools Reference](../reference/mcp-tools.md#search) for all
parameters (`mode`, `pathGlob`, `component`, `exact`, `includeGraphContext`,
etc.).

> Over HTTP (Path B) the server cannot observe your working directory. Pass
> your absolute working directory in the `cwd` argument (or an explicit
> `projectId`) on each call, or you may get *"Could not detect project"*.

### From the CLI (quick sanity check)

You can confirm indexing worked without an MCP client:

```bash
wqm project search 'authentication middleware'      # full-text search, current project
wqm search project 'handleRequest' --limit 5        # equivalent under the search subcommand
```

> The bare top-level form `wqm search "query"` is **deprecated**; use
> `wqm project search '<query>'` (full-text/regex) or
> `wqm search project '<query>'`. See *Known onboarding gaps* below.

---

## 6. Troubleshooting / common first-timer gaps

**"Could not detect project" (especially on Windows / mounted paths).**
The MCP server detects the project from your working directory. Over HTTP it
can't see your cwd, and on Windows the daemon may register a container-mount
path (`/run/desktop/mnt/host/c/...`) while the client sends `C:\...`. Pass the
absolute `cwd` argument or an explicit `projectId` on each `search`/`grep`/
`list`/`retrieve`/`rules` call to bypass detection.

**Daemon not running.** `wqm status health` shows the daemon disconnected, or
commands hang/refuse the connection. Check `wqm service status` (Path A) or
`docker compose ps` / `docker logs wqm-memexd` (Path B). On Path A, start it
with `wqm service start`. On Path B, the `mcp` service deliberately won't start
until `memexd` reports healthy.

**Empty search results.** Almost always means indexing hasn't finished (or the
project isn't registered). Confirm with `wqm project list` and wait for
`wqm queue stats` to reach zero pending/in-progress items. The first pass is
slow because grammars compile and the embedding model loads on demand.

**Qdrant unreachable.** `wqm status health` flags Qdrant as down, or
`curl http://localhost:6333/healthz` fails. Confirm the container is running
(`docker ps`), the port mapping matches (`6333` REST / `6334` gRPC), and
`QDRANT_URL` points at the right host/port. For Qdrant Cloud, set
`QDRANT_API_KEY`. More cases in [Troubleshooting](../TROUBLESHOOTING.md).

**Project must be a Git repo.** `wqm project register` requires the directory
to be (or live inside) a Git repository — detection and project scoping are
Git-based. Run `git init` first if needed.

**Recently added files not indexed (Windows / bind mounts).** Host Git events
can be missed by the watcher across a 9P/bind mount, so a freshly created
branch may stay unindexed until a Git hook fires or the daemon restarts.
Restart the daemon, or install the project's Git hooks, if a new branch's
content doesn't appear.

---

## Known onboarding gaps (surfaced by this runbook)

These are real discrepancies a first-time user will hit. They are documented
here so they can be fixed separately:

1. **Deprecated search command in onboarding docs.** `README.md` and
   `docs/quick-start.md` advertise `wqm search "query"`, but `wqm 0.1.3`
   reports that top-level form as deprecated in favor of
   `wqm project search` / `wqm search project`.
2. **Wrong project command in quick-start.** `docs/quick-start.md` uses
   `wqm project add /path` and `wqm admin health`, but the actual CLI commands
   are `wqm project register <path>` and `wqm status health`.
3. **Missing Qdrant corruption runbook.** Internal notes reference
   `docs/runbooks/qdrant-corruption.md` and a
   `docker/qdrant-quarantine-wrapper.sh`, but neither exists in this branch.
   The corruption-handling guidance above is therefore a stopgap.

---

_workspace-qdrant-mcp v0.1.3 — clean-install runbook_
