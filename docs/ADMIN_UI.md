# Admin UI

A browser-based dashboard for the workspace-qdrant MCP server. Lives at
`http://<mcp-host>:<mcp-port>/admin/` (default `http://localhost:6335/admin/`).

## What it does

- **Discover** git repositories under a configurable parent directory
  by scanning for `.git` markers (recursive, depth-limited).
- **Register** discovered candidates with the daemon — same effect as
  calling `RegisterProject` via gRPC, but one click and no CLI.
- **Monitor** queue depth, daemon health, indexed-document totals, and
  registered watch folders in real time (polling every 5 seconds).
- **Persist** the discovery configuration (`devRoot`, `scanDepth`,
  approved-projects list) in a JSON file under
  `$WQM_DATA_DIR/admin-settings.json` — survives container restarts.

## Access

The UI is mounted on the same HTTP port as the MCP transport (default
`6335`). Static assets (`/admin/`, `/admin/app.js`, `/admin/style.css`)
are served **without authentication** because a browser cannot inject a
Bearer header on the initial GET. Once the SPA loads, it prompts for
the same `MCP_HTTP_TOKEN` that protects the MCP transport, stores it in
the page's `sessionStorage`, and attaches it to every `/admin/api/*`
request from then on.

| Path | Auth | Why |
|---|---|---|
| `/admin/` | none | HTML/CSS/JS — public assets, no data |
| `/admin/app.js`, `/admin/style.css` | none | Same |
| `/admin/api/*` | Bearer (`MCP_HTTP_TOKEN`) | Reads daemon state, runs scans, mutates registrations |

The static handler refuses any path containing `..` or that escapes the
static root, so the unauthenticated surface is strictly the three
files above.

## Quickstart

1. `docker compose --env-file docker/.env up -d` (or your equivalent).
2. Open `http://localhost:6335/admin/` in a browser.
3. Paste your `MCP_HTTP_TOKEN` into the login field. (Same value as in
   `docker/.env`. Find it with
   `grep ^MCP_HTTP_TOKEN= docker/.env | cut -d= -f2-`.)
4. In **Discovery settings**, set `Parent dev root` to the directory
   containing your repositories. With the dockerized stack this is the
   container-visible path (typically
   `/run/desktop/mnt/host/c/Users/<you>/dev` on Windows + Docker
   Desktop, or `/home/<you>/dev` on Linux). Must match
   `WQM_DEV_ROOT` from `docker/.env`.
5. Click **Save**, then **Scan now**.
6. The **Discovered candidates** table lists every git repo (main or
   worktree) under the parent root, with its current branch, remote,
   and depth.
7. Click **Register** next to each project you want indexed. The
   daemon picks it up via gRPC and begins watching/ingesting.
8. **Registered projects** lists everything the daemon currently
   tracks; click **Deactivate** to deprioritize one.
9. Leave the page open — every 5 seconds the dashboard re-polls
   `/admin/api/snapshot` and refreshes daemon status, queue stats, and
   the registered list.

## Sections explained

### Header stats (top of page)

- **Daemon** — pill: `healthy` / `unhealthy`. Detail: active project
  count + total collections.
- **Queue** — current pending count (large number) and the
  pending/in-progress/failed breakdown beneath. Pulled from the
  daemon's `getStatus` gRPC.
- **Indexed** — total documents in Qdrant, total collections, and the
  number of registered watch folders.

### Discovery settings

- **Parent dev root** — absolute path to the directory that contains
  one or more git repositories as direct children. The scan walks down
  from here.
- **Depth** — how many directory levels to descend. 1 = direct
  children only (recommended). 5 = max (clamped server-side).

Clicking **Save** writes the new values to
`$WQM_DATA_DIR/admin-settings.json` immediately. Clicking **Scan now**
triggers a fresh scan with the current values.

### Discovered candidates

Each row shows a repository the scan found but that is not yet
registered with the daemon:

- **Project** — full path and basename.
- **Branch** — current branch (or `HEAD` if detached).
- **Remote** — `remote.origin.url` if configured.
- **Depth** — distance from the parent root.
- **Type** — `repo` (main worktree) or `worktree` (linked, `.git` is a
  file).

The **Register** button POSTs to `/admin/api/projects/register` which
forwards a `RegisterProject` gRPC to the daemon. On success the row
moves to the **Registered projects** table on the next poll cycle.

### Registered projects

Every watch folder the daemon knows about, regardless of activity
state:

- **Project** — path the daemon is watching.
- **Tenant ID** — the project's `tenant_id` (e.g. `local_5288aa13ad6c`
  for path-derived or a 12-char hex for remote-derived). Stable
  across worktrees of the same repo when a remote is configured.
- **Active** — pill `active` (one or more live MCP sessions) or
  `idle`.
- **Last activity** — relative time since `last_activity_at`.

The **Deactivate** button sends a `DeprioritizeProject` gRPC. Note
this only decrements the session counter; the watch folder persists
in the database and the daemon keeps the index intact.

### Debug snapshot

A collapsed `<details>` pane showing the raw JSON of the latest
`/admin/api/snapshot` response. Useful when something doesn't render
the way you expect — pop the disclosure open and compare what the
server actually returned to what the UI displayed.

## REST API

The UI is purely a consumer of the JSON API. Anything the UI does, you
can do from the command line. All endpoints require
`Authorization: Bearer $MCP_HTTP_TOKEN`.

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/admin/api/snapshot` | — | Consolidated state: `settings`, `daemon`, `queue`, `projects.registered` |
| `GET` | `/admin/api/settings` | — | The persisted settings JSON |
| `PUT` | `/admin/api/settings` | `{ devRoot?, scanDepth? }` | Updated settings |
| `POST` | `/admin/api/projects/scan` | `{ devRoot?, scanDepth? }` (override; defaults to persisted values) | `{ scan: { root, maxDepth, visited, skipped, candidates[], finishedAt }, settings }` |
| `POST` | `/admin/api/projects/register` | `{ path, registerIfNew? }` | `{ ok, projectId, created, newlyRegistered, isActive, isWorktree, watchPath }` |
| `POST` | `/admin/api/projects/deregister` | `{ projectId, path? }` | `{ ok, isActive, newPriority }` |

### Example: scan + register from `curl`

```sh
TOKEN="$(grep ^MCP_HTTP_TOKEN= docker/.env | cut -d= -f2-)"

# Scan
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"devRoot":"/run/desktop/mnt/host/c/Users/me/dev","scanDepth":1}' \
  http://localhost:6335/admin/api/projects/scan

# Register a specific candidate
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"path":"/run/desktop/mnt/host/c/Users/me/dev/my-project"}' \
  http://localhost:6335/admin/api/projects/register
```

## Implementation notes

- **No build pipeline**: the frontend is vanilla HTML + CSS + JS. To
  modify it, edit files in `src/typescript/mcp-server/src/admin/static/`
  and run `npm run build` — `copy:admin-static` ships them into `dist/`.
- **Backend** lives in `src/typescript/mcp-server/src/admin/`:
  `routes.ts` (REST dispatcher), `handler.ts` (static + admin
  request router), `discovery.ts` (recursive `.git` scan),
  `settings-store.ts` (JSON file persistence).
- **gRPC, not SQLite, for project lists**: the snapshot route asks the
  daemon via `ListProjects` rather than reading `state.db` directly.
  Reason: on Docker Desktop the SQLite file lives behind a 9P bind
  mount that doesn't implement the shared-memory locks SQLite uses to
  coordinate with the writer — direct opens fail with
  `SQLITE_CANTOPEN`. The gRPC path sidesteps the issue and keeps the
  daemon as the canonical reader.
- **Polling, not streaming**: refresh interval is 5 s and there is no
  SSE/WebSocket. Keeps the server single-protocol (Streamable HTTP for
  MCP, plain REST for admin). Cost is negligible — `/admin/api/snapshot`
  takes ~10 ms.

## Security posture

- The Bearer token is the **only** auth mechanism. Same scope as the
  MCP transport: anyone with the token can list/register/deactivate
  watch folders.
- Token is stored in the browser's `sessionStorage` — cleared when
  the tab closes; never written to `localStorage`.
- HTTPS is the operator's responsibility. The reference `caddy`
  profile in `docker-compose.yml` (`--profile tls`) terminates TLS in
  front of the MCP container; behind it everything else is unchanged.
- The static handler never serves anything outside
  `dist/admin/static/`. Anti-traversal: any URL with `..` after
  normalization is rejected before the file is opened.
- No write access to the daemon's SQLite from this surface. Mutations
  go through gRPC where the daemon's own validation runs.

## Limitations

- The discovery scan walks the filesystem synchronously inside the
  request. On very large directories with depth > 1 the scan can
  block the connection for a second or two. Default depth is 1.
- Settings are stored in a single JSON file with no concurrency
  control. If you have multiple admin tabs open and submit settings
  changes at the same time, last write wins. Acceptable for an
  operator UI.
- The candidates table hides candidates whose path already appears in
  the registered list — but matching is by exact path string. If the
  daemon stored a canonicalized form different from what the scan
  produced, the same repo can show up in both tables. Spec 16 keeps
  this rare; report if you see one.
- No editing of individual `RegisterProject` fields (name, priority).
  The UI always sends `register_if_new: true, priority: high`. Use
  the CLI (`wqm project register`) for finer-grained control.

## Related

- [`scripts/git-hooks/README.md`](../scripts/git-hooks/README.md) —
  POSIX git hooks that call the same `workspace_index` action
  programmatically.
- [`docs/specs/19-branch-worktree-audit.md`](specs/19-branch-worktree-audit.md)
  §7 — bugs in the branch/worktree flow that this UI helps observe.
- [`docs/deployment/docker.md`](deployment/docker.md) — full Docker
  stack reference (the admin UI is part of the same MCP container).
