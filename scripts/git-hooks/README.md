# Git hooks for the dockerized MCP stack

Companion to `scripts/windows/indexed-projects-hooks.ps1`. These POSIX-shell
hooks talk to the dockerized MCP HTTP server (`workspace-qdrant-mcp:local`)
instead of running PowerShell registry logic on the host, so the same setup
works on Linux, macOS, and Windows (Git Bash / WSL).

The hook itself stays on the host — git always fires it where you run `git`.
Only the logic moves into the container.

## What gets installed

Five hooks in the target repo's git hooks directory:

- `post-checkout` (branch switches only; file checkouts are ignored)
- `post-commit`
- `post-merge`
- `post-rewrite`
- `post-worktree-add`

Each one execs `wqm-sync-branch.sh`, which:

1. Reads the current branch, commit, remote, and worktree state via `git`.
2. Opens an MCP HTTP session (`initialize` → `notifications/initialized`).
3. Calls `workspace_index` with `action: "sync_current_branch"`.

The MCP server's new TypeScript handler forwards the call to the daemon
(`RegisterProject` with `register_if_new=true, priority=high`). The daemon
detects the worktree, computes the `tenant_id`, and either reactivates an
existing watch folder or enqueues a new one.

## Install

```sh
# From the workspace-qdrant-mcp checkout:
export MCP_HTTP_TOKEN="<same token as docker/.env>"

# Install into the current repo (auto-detects git common dir, so worktrees work):
scripts/git-hooks/install.sh

# Or into a specific repo:
scripts/git-hooks/install.sh --repo /path/to/my-project
```

Options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--repo <path>` | `git rev-parse --show-toplevel` | Target repo |
| `--hooks-dir <path>` | `<git-common-dir>/hooks` | Override hook install dir |
| `--mcp-url <url>` | `http://localhost:6335/mcp` | MCP HTTP endpoint |
| `--token <value>` | `$MCP_HTTP_TOKEN` | Bearer token |
| `--log <path>` | `<repo>/.wqm-fork/logs/git-hooks.jsonl` | Log file |
| `--wqm-script <path>` | sibling `wqm-sync-branch.sh` | Override sync script |
| `--uninstall` | — | Remove only the hooks the installer wrote |

The installer refuses to overwrite a hook it didn't create (no marker
present). Remove or rename existing hooks before re-installing.

## Verify

After install:

```sh
# Trigger a hook manually
git checkout -b test-wqm-sync && git checkout -

# Tail the log
tail -f .wqm-fork/logs/git-hooks.jsonl

# Or run the script directly
WQM_HOOK_LOG=/tmp/wqm-hook.log scripts/git-hooks/wqm-sync-branch.sh test-manual
```

If the MCP container is up and the token matches, the log line shows the
daemon's `RegisterProject` response, including the resolved `project_id`,
`watch_path`, and `is_worktree` flag.

## Observability

Every `sync_current_branch` call the MCP server services emits Prometheus
metrics (scraped from `mcp:9092/metrics`):

| Metric | Type | Labels | Notes |
|--------|------|--------|-------|
| `wqm_git_hook_invocations_total` | counter | `hook`, `result`, `is_worktree` | `result` is `success` / `error` / `bad_request` |
| `wqm_git_hook_duration_seconds`  | histogram | `hook`, `result` | Server-side; excludes the client's session init |

These power three panels on the **WQM — System Overview** Grafana dashboard
(`docker/grafana/dashboards/system-overview.json`):

- *Git Hook Activity* — stacked rate by `hook` × `result`, red for `error`, orange for `bad_request`.
- *Git Hook Latency (P50 / P95 / P99)* — per-hook tail latency, useful for spotting LSP-startup pauses.
- *Git Hook Error Ratio* — single-stat 5-minute error share, green ≤5% / yellow / red ≥20%.

If the panels stay flat at 0 after running `git checkout -b test-wqm-sync && git checkout -`,
either the hook isn't reaching the MCP server (check `WQM_MCP_URL`/`WQM_MCP_TOKEN` and
the host integration card in the admin UI) or Prometheus isn't scraping the MCP target
(check the *MCP Target Health* stat on the same dashboard).

## How it differs from the PowerShell hooks

| Aspect | PowerShell hook | This (POSIX) |
|--------|----------------|--------------|
| Runs where | Host (PS only) | Host (any sh) |
| Writes to | `.wqm-fork/indexed-projects.json` | (none — daemon owns state) |
| Calls | `indexed-projects-registry.ps1` (host) | `workspace_index` over MCP HTTP (container) |
| Side effects | Local JSON registry + `wqm project register` | `RegisterProject` gRPC via MCP |
| Daemon target | Local (no Docker) | Containerized (`memexd` over gRPC) |

If you use both fork registries (`indexed-projects.json` for `workspace_index`
list/observe actions, and SQLite `watch_folders` for the daemon), keep the
PowerShell installer for the JSON-side and use this one for the daemon-side.
They are not mutually exclusive — they just write to different stores.

## Environment overrides (script-level)

The sync script reads these env vars (set by the installed hooks, but useful
for manual invocation too):

- `WQM_MCP_URL` — MCP endpoint (default `http://localhost:6335/mcp`)
- `WQM_MCP_TOKEN` — bearer token (falls back to `MCP_HTTP_TOKEN`)
- `WQM_HOOK_TIMEOUT` — curl `--max-time` seconds (default `5`)
- `WQM_HOOK_LOG` — append-log path (silent if unset)
- `WQM_HOOK_NAME` — label recorded in the response (default `manual` or `$1`)

The script always `exit 0` so a slow or unreachable MCP never blocks `git`.
