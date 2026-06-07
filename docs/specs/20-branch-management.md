# 20 — Branch Management

This specification defines the authoritative design for branch-aware content management in workspace-qdrant-mcp. It supersedes the informal branch handling described in `02-collection-architecture.md §Branch Handling` and expands on the lifecycle detector introduced in `19-branch-worktree-audit.md`.

---

## 1. Overview and Motivation

The current implementation stores a single `branch` string field on each Qdrant point and in `file_metadata.branch` in the search database. This model has three compounding problems:

**No content deduplication across branches.** When a file is identical on two branches (common for most of a repository), the daemon embeds the file twice, stores two Qdrant points with identical vectors, and occupies double the storage. At five active branches, every unchanged file is embedded five times.

**No automatic branch scoping.** All MCP search surfaces — semantic search, FTS5 grep, graph queries — return results from all branches unless the caller explicitly filters. A user working on `feature/auth` receives results from `main`, `fix/login`, and every other branch simultaneously, generating noise proportional to branch count.

**No resilient branch discovery.** The current model depends on creation-time events. If the daemon is down during a `git checkout -b` or `git worktree add`, the new branch is never indexed with the correct lineage information.

This specification resolves all three problems through:

1. Content-hash deduplication via a `branches` array payload field replacing the scalar `branch` field.
2. Branch-scoped search as the default on all surfaces, with an explicit opt-out.
3. Discovery-based branch population that is resilient to daemon downtime and self-healing.

---

## 2. Data Model Changes

### 2.1 Qdrant Payload: `branch` → `branches`

The `branch: string` field in the `projects` collection payload is replaced by `branches: string[]`.

**Current payload (before this spec):**

```json
{
  "project_id": "a1b2c3d4e5f6",
  "branch": "main",
  "base_point": "...",
  "relative_path": "src/auth.rs",
  "file_hash": "sha256:abc123..."
}
```

**New payload (this spec):**

```json
{
  "project_id": "a1b2c3d4e5f6",
  "branches": ["main", "feature/auth"],
  "base_point": "...",
  "relative_path": "src/auth.rs",
  "file_hash": "sha256:abc123..."
}
```

A Qdrant point now represents a unique file content version shared by one or more branches. The `branches` array is the set of branches for which this exact file content is the current version.

**Qdrant index configuration change:**

The `branch` keyword index is replaced by a `branches` keyword index with `is_tenant: false`. Because `branches` is an array, Qdrant's array field matching (`match: { any: [...] }`) applies. The existing `project_id` index is unchanged.

**Filtering syntax:**

```json
// Branch-scoped filter (any of the requested branches)
{
  "must": [
    { "key": "project_id", "match": { "value": "a1b2c3d4e5f6" } },
    { "key": "branches",   "match": { "any": ["feature/auth"] } }
  ]
}

// Cross-branch filter (no branches constraint)
{
  "must": [
    { "key": "project_id", "match": { "value": "a1b2c3d4e5f6" } }
  ]
}
```

### 2.2 Base Point Identity

The `base_point` formula changes to remove the branch component, because a point is now content-addressed rather than branch-addressed:

**Old formula:**

```
base_point = hash(tenant_id, branch, relative_path, file_hash)
```

**New formula:**

```
base_point = hash(tenant_id, relative_path, file_hash)
```

This change is the foundation of deduplication: identical file content at the same relative path within a project now produces the same `base_point` regardless of which branch is being ingested.

**Consequence for `tracked_files`:** The `UNIQUE(watch_folder_id, file_path, branch)` constraint is loosened. Multiple branches can reference the same `file_id` / `base_point` row. See §2.3.

### 2.3 SQLite `tracked_files` Table Changes

The `tracked_files` table in `state.db` is the authoritative file inventory. Changes:

**Unique constraint change:**

```sql
-- Old: one row per (watch_folder_id, file_path, branch)
UNIQUE(watch_folder_id, file_path, branch)

-- New: one row per (watch_folder_id, file_path, file_hash) -- content-versioned
UNIQUE(watch_folder_id, file_path, file_hash)
```

The `branch` column is retained but changes meaning: it becomes the **primary branch** that most recently wrote this row. A separate `branches` column (TEXT, JSON array) tracks all branches sharing this exact content. Application-level consistency keeps `branches` in sync with the Qdrant point's `branches` array.

**Column additions:**

```sql
ALTER TABLE tracked_files ADD COLUMN branches TEXT NOT NULL DEFAULT '[]';
-- JSON array of branch names sharing this exact file_hash at this relative_path.
-- Invariant: branch (primary) is always an element of branches[].

ALTER TABLE tracked_files DROP COLUMN branch;         -- remove scalar
ALTER TABLE tracked_files ADD COLUMN primary_branch TEXT; -- most recent writer
```

Because this project has no released users, no migration compatibility is required. The schema migration (schema version N+1) drops the old `branch` column and introduces `primary_branch` and `branches`.

**Updated `tracked_files` schema (relevant columns only):**

```sql
CREATE TABLE tracked_files (
    file_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,
    file_path     TEXT NOT NULL,           -- RELATIVE path
    relative_path TEXT,                    -- alias
    primary_branch TEXT,                   -- branch that last wrote this row
    branches      TEXT NOT NULL DEFAULT '[]', -- JSON: ["main","feature/auth"]
    collection    TEXT NOT NULL DEFAULT 'projects',
    base_point    TEXT,
    file_type     TEXT,
    language      TEXT,
    file_mtime    TEXT NOT NULL,
    file_hash     TEXT NOT NULL,
    chunk_count   INTEGER DEFAULT 0,
    chunking_method TEXT,
    lsp_status    TEXT DEFAULT 'none',
    treesitter_status TEXT DEFAULT 'none',
    last_error    TEXT,
    incremental   INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, file_path, file_hash) -- content-versioned uniqueness
);

CREATE INDEX idx_tracked_files_watch      ON tracked_files(watch_folder_id);
CREATE INDEX idx_tracked_files_path       ON tracked_files(file_path);
CREATE INDEX idx_tracked_files_base_point ON tracked_files(base_point);
CREATE INDEX idx_tracked_files_branches   ON tracked_files(watch_folder_id, branches);
-- Note: branches is a JSON column; range queries use json_each() in SQLite.
```

**Branch membership queries:**

```sql
-- Find all files on a given branch for a project
SELECT tf.*
FROM tracked_files tf
JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
WHERE wf.tenant_id = :tenant_id
  AND EXISTS (
      SELECT 1 FROM json_each(tf.branches)
      WHERE json_each.value = :branch
  );

-- Remove a branch from all files in a project
UPDATE tracked_files
SET branches = json_remove(
    branches,
    '$[' || (
        SELECT key FROM json_each(branches)
        WHERE value = :branch
    ) || ']'
)
WHERE watch_folder_id IN (
    SELECT watch_id FROM watch_folders WHERE tenant_id = :tenant_id
);
```

### 2.4 SQLite `file_metadata` Table Changes (search.db)

The `file_metadata` table in `search.db` drives FTS5 branch scoping. The scalar `branch` column is **retained as-is** — `file_metadata` uses a per-row model where each `(file_id, branch)` pair gets its own row. This preserves fast FTS5 filtering via `fm.branch = ?3` (simple indexed equality, no `json_each()` in the hot path).

**No schema change to `file_metadata` itself.** The change is in insertion/deletion logic:

When a file is shared across branches (e.g., `branches: ["main", "feature/auth"]` in `tracked_files`), the ingestion pipeline inserts one `file_metadata` row per branch. The `code_lines` rows are shared (same `file_id`).

When a branch is removed from a file's membership, the corresponding `file_metadata` row is deleted.

**FTS5 branch-scoped query (`FTS5_SEARCH_BY_PROJECT_BRANCH_SQL`) — unchanged:**

```sql
SELECT cl.line_id, cl.file_id, cl.seq, cl.content, fm.tenant_id, fm.file_path, fm.branch
FROM code_lines cl
JOIN code_lines_fts fts ON cl.line_id = fts.rowid
JOIN file_metadata fm ON cl.file_id = fm.file_id
WHERE fts.content MATCH ?1 AND fm.tenant_id = ?2 AND fm.branch = ?3
ORDER BY cl.file_id, cl.seq
```

This query uses a simple equality predicate on `fm.branch` — indexed, fast, no `json_each()`.

**Cross-branch grep** omits the `fm.branch = ?3` predicate entirely (returns results from all branches).

**`UPSERT_FILE_METADATA_SQL` change:**

The existing UPSERT targets `ON CONFLICT(file_id)`. For multi-branch support, the conflict target changes to `ON CONFLICT(file_id, branch)` since there can now be multiple rows per file:

```sql
INSERT INTO file_metadata (file_id, tenant_id, branch, file_path, base_point, relative_path, file_hash)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
ON CONFLICT(file_id, branch) DO UPDATE SET
    tenant_id     = excluded.tenant_id,
    file_path     = excluded.file_path,
    base_point    = excluded.base_point,
    relative_path = excluded.relative_path,
    file_hash     = excluded.file_hash
```

Where `?3` is a single branch string (e.g., `'main'`).

### 2.5 Graph Schema Changes (graph.db)

The `graph_nodes` and `graph_edges` tables in `graph.db` currently carry no branch column. Graph queries are scoped only by `tenant_id`.

**Changes:**

```sql
ALTER TABLE graph_nodes ADD COLUMN branches TEXT NOT NULL DEFAULT '["main"]';
-- JSON array of branches for which this symbol exists in its current form.

ALTER TABLE graph_edges ADD COLUMN branch TEXT;
-- The branch on which this edge (call relationship) was observed.
-- NULL means "inferred globally" (e.g., cross-branch reference).
```

**Index additions:**

```sql
CREATE INDEX idx_nodes_branches ON graph_nodes(tenant_id, branches);
CREATE INDEX idx_edges_branch   ON graph_edges(tenant_id, branch);
```

**Branch-scoped graph query pattern:**

```sql
-- Nodes visible on a given branch
SELECT * FROM graph_nodes
WHERE tenant_id = :tenant_id
  AND EXISTS (
      SELECT 1 FROM json_each(branches)
      WHERE json_each.value = :branch
  );

-- Edges for a given branch
SELECT * FROM graph_edges
WHERE tenant_id = :tenant_id
  AND (branch = :branch OR branch IS NULL);
```

---

## 3. Branch Discovery Algorithm

Discovery-based population is the mechanism by which an encountered branch is fully populated in the index without relying on creation-time events.

### 3.1 Trigger Conditions

Discovery runs whenever the queue processor encounters a file event (add, update, or scan) carrying a branch name that has zero existing `tracked_files` entries for the project. This covers:

- Daemon startup after downtime during which `git checkout -b` occurred.
- `git worktree add` while daemon was not running.
- Database wipe followed by daemon restart.
- First-ever registration of a project that already has multiple branches.

### 3.2 Algorithm

```
fn discover_branch(tenant_id, new_branch, project_root):

  1. HASH SCAN: Walk all files on new_branch (from filesystem).
     For each file: compute file_hash = SHA256(content).
     Result: set(B) = { (relative_path, file_hash) }

  2. LOAD KNOWN HASHES: Query tracked_files for all (relative_path, file_hash, branches)
     tuples where watch_folder_id corresponds to tenant_id.
     Result: known_files = { (relative_path, file_hash) → branches[] }

  3. CLASSIFY FILES:
     For each (path, hash) in set(B):
       If (path, hash) in known_files:
         → SHARED: add new_branch to existing tracked_files.branches[] and
                   to the Qdrant point's branches[]. No re-embedding.
       Else:
         → NOVEL: queue as normal file/ingest for new_branch.

  4. INFER PARENT (optional, for lineage tracking):
     For each known branch K:
       diff(K, B) = files in K not in B  (by hash)
       parent = argmin_K( |diff(K, B)| )
     Store parent branch name in watch_folders metadata (informational only).

  5. MARK DISCOVERED:
     Insert a row in branch_registry (see §8.3) for new_branch.
     Update watch_folders.known_branches JSON field.
```

**Complexity:** Step 1 is O(F·S) where F is file count and S is SHA256 cost (fast). Step 2 is a single SQLite query returning an in-memory map. Step 3 is O(F) with hash map lookups. Step 4 is optional and O(B·F). The expensive path (embedding) is proportional to only the novel files.

**Resilience properties:**
- Works with no prior history (empty `tracked_files`).
- Works after DB wipe: all files are classified as NOVEL and re-embedded.
- Self-healing: re-running discovery on an already-discovered branch is a no-op (all files are SHARED).
- Correct for worktrees: same algorithm, same project, different filesystem path.

### 3.3 Deduplication Invariant

Two tracked_files rows with the same `(watch_folder_id, file_path)` but different `file_hash` values represent **different versions** of the same file (on different content states). Each has its own Qdrant point(s). A branch is associated with the row whose `file_hash` matches the file's current content on that branch.

Two branches sharing the same `file_hash` for a given path share the **same Qdrant point(s)** — they are both listed in `tracked_files.branches` for that row, and both listed in the Qdrant point's `branches` payload field.

---

## 4. Ingestion Pipeline Changes

### 4.1 Branch Detection at Queue-Processing Time

The current model detects the branch at file-event-enqueue time. This is unreliable because queue lag can be seconds to minutes.

**New rule:** The queue processor detects the branch by running:

```rust
git -C <project_root> rev-parse --abbrev-ref HEAD
```

at the moment it dequeues and processes a `file` queue item. This is the same `detect_branch()` function already present in `src/rust/cli/src/commands/ingest/detect.rs`, moved to a shared location callable by the queue processor.

For worktrees, the project root is the worktree's checkout path. `git rev-parse --abbrev-ref HEAD` correctly returns the worktree's branch.

The detected branch is stored in the queue item's processing context; it is not re-read on retry (the stored branch from first processing attempt is used).

**Non-git projects:** Branch is `"default"` (unchanged from current behavior).

**Detached HEAD:** Branch is the short SHA of HEAD (8 characters), consistent with `detect_git_status` in `git/types.rs:63`.

### 4.2 Content-Hash Deduplication During Ingest

When processing a `file/ingest` or `file/update` queue item:

```
BEGIN TRANSACTION;

1. Compute file_hash = SHA256(file_content)
2. Detect branch = git rev-parse --abbrev-ref HEAD (at project_root)
3. Query tracked_files WHERE watch_folder_id = ? AND file_path = ? AND file_hash = ?
   (matches UNIQUE(watch_folder_id, file_path, file_hash) constraint)
   - If row exists:
       → SHARED: add branch to tracked_files.branches[] if not present.
                 Enqueue a Qdrant set_payload to add branch to existing point's branches[].
                 Insert file_metadata row for (file_id, branch) if not exists.
                 Mark queue item done. COMMIT. RETURN.
   - If no row with this hash:
       → NEW or DIVERGED content for this branch. Proceed to step 4.

4. (New or diverged content)
   A tracked_files row may exist for the same (watch_folder_id, file_path) with a
   DIFFERENT file_hash (the old version, referenced by other branches). That row is
   untouched — it remains valid for those branches.

5. Chunk file content. Generate embeddings.
6. Upsert Qdrant points with branches = [branch].
7. INSERT tracked_files row with branches = JSON([branch]), file_hash, primary_branch = branch.
   (UNIQUE(watch_folder_id, file_path, file_hash) allows this — different hash = different row.)
8. Insert file_metadata row for (file_id, branch).
9. Mark queue item done.

COMMIT;
```

**Qdrant set_payload for branch addition (step 3, shared path):**

```rust
client.set_payload(
    "projects",
    SetPayload {
        payload: hashmap! {
            "branches" => json!([...existing_branches, new_branch])
        },
        filter: Some(Filter::must(vec![
            FieldCondition::new_match("project_id", tenant_id.into()),
            FieldCondition::new_match("base_point", base_point.into()),
        ])),
        ..Default::default()
    },
).await?;
```

This targets all chunks of the file (all points sharing the same `base_point`) atomically.

### 4.3 File Delete

When a file is deleted on a branch:

```
1. Detect branch at processing time.
2. Load tracked_files row for this (watch_folder_id, file_path).
3. Remove branch from tracked_files.branches[].
4. If tracked_files.branches[] is now empty:
     → No branch references this content version anymore.
     → Delete all Qdrant points for this base_point.
     → Delete tracked_files row.
     → Delete file_metadata row.
5. Else:
     → Other branches still reference this content.
     → Update tracked_files.branches[] (remove this branch).
     → Enqueue Qdrant set_payload to remove branch from existing point's branches[].
     → Update file_metadata.branches[] accordingly.
6. Mark queue item done.
```

### 4.4 File Update

A file update on branch B means B's version of the file has changed content:

```
1. Detect branch B at processing time.
2. Load tracked_files row for (watch_folder_id, file_path).
3. Compute new file_hash.
4. If file_hash unchanged: mark done, no-op.
5. Remove B from tracked_files.branches[] of the old row (step 4.3 logic).
6. Check if new file_hash matches any existing tracked_files row for this path (another branch).
   - If yes: shared path (step 4.2 step 3).
   - If no: novel content (step 4.2 step 4 onward).
```

---

## 5. Search Scoping

### 5.1 Default Behavior

Branch-scoped search is the default on all surfaces. The current branch is auto-detected from the project's CWD using `git rev-parse --abbrev-ref HEAD` called by the MCP server at session start (see §6.2).

**Opt-out:** Pass `branch: "*"` to any search surface to bypass the branch filter and search across all branches.

### 5.2 Qdrant Semantic Search

The MCP server constructs the Qdrant filter as follows:

```typescript
function buildBranchFilter(projectId: string, branch: string | undefined): Filter {
  const projectFilter = { key: "project_id", match: { value: projectId } };

  if (!branch || branch === "*") {
    // Cross-branch: no branch constraint
    return { must: [projectFilter] };
  }

  return {
    must: [
      projectFilter,
      { key: "branches", match: { any: [branch] } }
    ]
  };
}
```

This applies to all calls to `search`, `retrieve`, and any internal Qdrant queries in the MCP server.

### 5.3 FTS5 Full-Text Search (Grep Tool)

The `grep` tool routes through the search database. Branch scoping uses the updated `FTS5_SEARCH_BY_PROJECT_BRANCH_SQL` (see §2.4).

Two SQL variants are maintained:

- `FTS5_SEARCH_BY_PROJECT_BRANCH_SQL` — scoped to one branch (current default)
- `FTS5_SEARCH_ALL_BRANCHES_SQL` — no branch filter (for `branch: "*"`)

The router selects the variant based on the branch parameter:

```rust
let sql = if branch == "*" {
    FTS5_SEARCH_ALL_BRANCHES_SQL
} else {
    FTS5_SEARCH_BY_PROJECT_BRANCH_SQL
};
```

### 5.4 Graph Queries

Graph queries add a branch predicate to node and edge lookups:

```sql
-- Branch-scoped symbol lookup
SELECT * FROM graph_nodes
WHERE tenant_id = :tenant_id
  AND EXISTS (SELECT 1 FROM json_each(branches) WHERE value = :branch);

-- Cross-branch symbol lookup
SELECT * FROM graph_nodes
WHERE tenant_id = :tenant_id;
```

The `wqm graph query` CLI command and the MCP `search` tool's `includeGraphContext` option both accept `branch` as a parameter, defaulting to the current branch if omitted.

### 5.5 List Tool

The `list` MCP tool queries `tracked_files`. With branch scoping:

```sql
SELECT tf.*
FROM tracked_files tf
JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
WHERE wf.tenant_id = :tenant_id
  AND EXISTS (SELECT 1 FROM json_each(tf.branches) WHERE value = :branch)
```

Cross-branch list (when `branch = "*"`) omits the `EXISTS` clause.

---

## 6. MCP API Changes

### 6.1 `search` Tool — `branch` Parameter Semantics

The `branch` parameter on the `search` tool changes from "optional filter" to "always applied, defaults to current branch":

```typescript
search({
  query: string,
  branch?: string,  // Default: auto-detected current branch.
                    // Pass "*" for cross-branch search.
  ...
})
```

**Default behavior (branch omitted):** The MCP server applies the current branch (detected at session start, see §6.2) as a filter.

**Explicit branch:** Filter to that specific branch.

**Cross-branch (`branch: "*"`):** No branch filter. Returns results from all branches.

The same semantics apply to `grep`, `retrieve`, and `list`.

### 6.2 Branch Auto-Detection in MCP Server

At `RegisterProject` time, the MCP server detects the current branch:

```typescript
async function detectCurrentBranch(projectRoot: string): Promise<string> {
  try {
    const result = await execGit(["-C", projectRoot, "rev-parse", "--abbrev-ref", "HEAD"]);
    const branch = result.stdout.trim();
    if (branch && branch !== "HEAD") {
      return branch;
    }
    // Detached HEAD: use short SHA
    const sha = await execGit(["-C", projectRoot, "rev-parse", "--short=8", "HEAD"]);
    return sha.stdout.trim();
  } catch {
    return "default"; // Non-git project
  }
}
```

This value is cached per session in the MCP server's in-memory state and used as the default `branch` filter on all subsequent search calls.

**Cache invalidation:** The branch is re-detected when:
- `RegisterProject` is called again (session reconnect).
- A `BranchEvent.Switched` is received from the daemon (future: gRPC push notification; for now, re-read on each tool invocation via a lightweight `.git/HEAD` check).

The `.git/HEAD` check on each tool invocation is a file read (nanoseconds) using `read_current_branch(git_dir)` from `git/reflog.rs`, not a subprocess.

### 6.3 `search` Tool — Response Metadata

Search results include the `branch` field in each result's metadata, reflecting which branches the result is present on:

```json
{
  "content": "...",
  "metadata": {
    "branches": ["main", "feature/auth"],
    "relative_path": "src/auth.rs",
    ...
  }
}
```

When searching with a branch filter, `branches` in results always includes the requested branch. When searching cross-branch, `branches` shows the full membership.

---

## 7. Branch Lifecycle

### 7.1 Branch Creation

Branch creation is handled entirely by the discovery algorithm (§3). No special creation-time event is required. When a file event arrives for an unknown branch, the queue processor runs discovery before processing the event.

The `BranchEvent::Created` from `BranchLifecycleDetector` triggers an immediate discovery scan of the new branch, front-running the file events that follow. This is an optimization (reduces latency to first searchable result) but not a correctness requirement.

### 7.2 Branch Deletion

Branch deletion follows the truth table established by the confirmed design decisions:

| Local branch exists | Remote branch exists | Action |
|--------------------|--------------------|--------|
| yes | yes | keep |
| yes | no | keep |
| yes | no remote configured | keep |
| no | yes | keep |
| no | no | **delete immediately** |
| no | no remote configured | **delete immediately** |

**"Delete immediately"** means: upon receiving `BranchEvent::Deleted`, check both local refs and remote refs. Delete only when both are absent.

**Deletion procedure:**

```
fn handle_branch_deleted(tenant_id, deleted_branch):

  1. Verify: git branch --list <deleted_branch> → empty?
  2. Verify: git ls-remote origin refs/heads/<deleted_branch> → empty?
     (Skip step 2 if no remote configured.)
  3. If branch still exists anywhere: return (no action).

  4. For all tracked_files WHERE tenant_id matches:
     a. Remove deleted_branch from branches[].
     b. If branches[] becomes empty:
          → Queue file/delete for this base_point (full Qdrant point deletion).
          → Delete tracked_files row and file_metadata row.
     c. Else:
          → Enqueue Qdrant set_payload to remove branch from point's branches[].
          → Update tracked_files.branches[] and file_metadata.branches[].

  5. Remove deleted_branch from graph_nodes.branches[] for all nodes in this project.
  6. Remove graph_edges WHERE branch = deleted_branch for this tenant.
  7. Remove deleted_branch from branch_registry (see §8.3).
```

**Remote check implementation:**

```rust
enum RemoteCheckResult {
    Exists,
    NotFound,
    Error(String),
}

fn branch_exists_remotely(project_root: &Path, branch: &str) -> RemoteCheckResult {
    let output = std::process::Command::new("git")
        .args(["-C", project_root.to_str().unwrap(),
               "ls-remote", "--heads", "origin",
               &format!("refs/heads/{}", branch)])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            if o.stdout.is_empty() {
                RemoteCheckResult::NotFound
            } else {
                RemoteCheckResult::Exists
            }
        }
        Ok(o) => RemoteCheckResult::Error(
            format!("git ls-remote failed: {}", o.status)
        ),
        Err(e) => RemoteCheckResult::Error(e.to_string()),
    }
}
// When remote check returns Error: DEFER deletion, do not delete.
// Log warning and retry on next BranchEvent or periodic reconciliation.
// Rationale: network outage + local branch delete must not destroy index data.
```

**search.db cleanup (#102):** `delete_file_metadata_for_branch` removes the
branch's `file_metadata` rows and, in the same transaction, the `code_lines`
(+ incremental FTS5 index entries) of any file left with no `file_metadata`
rows at all. This prune runs even when `tracked_files` no longer references
the branch — stale `file_metadata` rows duplicate every unfiltered grep match.

**Periodic reconciliation (#102):** `reconcile_stale_branches`
(`branch_cleanup::reconcile`) covers deletions missed while the daemon was
down (and deferred cleanups). memexd runs it 5 minutes after startup and then
daily: for every active git-tracked project watch folder it diffs the branches
recorded in `tracked_files.branches[]` ∪ `file_metadata.branch` against the
repository's local refs and routes each stale branch through
`cleanup_deleted_branch` (which re-checks local + remote existence before
deleting). A final pass prunes `code_lines` whose file has neither a
`file_metadata` row nor a `tracked_files` row; files still tracked but missing
metadata are left alone and logged (re-indexing restores them).

### 7.3 Branch Rename

Git branch renames (`git branch -m old new`) are detected as `BranchEvent::Renamed` by `BranchLifecycleDetector` after the fix applied in issue #69.

**Rename procedure:**

```
fn handle_branch_renamed(tenant_id, old_name, new_name):

  1. For all tracked_files WHERE tenant_id matches:
     a. In branches[] JSON array: replace old_name with new_name.
     b. If primary_branch = old_name: set primary_branch = new_name.
     c. Update file_metadata.branches[] in search.db accordingly.

  2. Enqueue Qdrant set_payload: for all points WHERE project_id = tenant_id
     AND branches contains old_name:
     → Replace old_name with new_name in branches[].

  3. In graph_nodes.branches[]: replace old_name with new_name.
  4. In graph_edges.branch: replace old_name with new_name.
  5. Update branch_registry: rename entry.
```

The Qdrant `set_payload` with a filter targeting `branches: any: [old_name]` plus a transform is not natively supported; implementation uses scroll + bulk set_payload in batches of 1000 points.

### 7.4 Default Branch Change

`BranchEvent::DefaultChanged` does not require data migration. The default branch is just a named branch; no special treatment is needed in the data model.

The `watch_folders` table stores the detected default branch for display purposes:

```sql
ALTER TABLE watch_folders ADD COLUMN default_branch TEXT;
-- Populated on project registration, updated on DefaultChanged event.
```

---

## 8. Worktree Integration

### 8.1 Worktree = Branch at Different Path

A git worktree is a branch checked out at a different filesystem path. From the daemon's perspective, a worktree is registered as a distinct `watch_folders` entry with:
- The same `tenant_id` as the main clone (same remote URL → same hash).
- A different `path` (the worktree checkout directory).
- A different `primary_branch` / `branches`.

The `is_worktree = true` flag (detected via `.git` file vs. directory, as confirmed in spec 19 §1.2) is stored in `watch_folders` and used to:
1. Force `tenant_id` to match the main clone's `tenant_id` at registration time (per spec 19 ADR note B: registration-time check in `register_project_with_disambiguation`).
2. Skip disambiguation: worktrees do not need path-based disambiguation since they share the main clone's identity.

### 8.2 MCP Server Branch Detection for Worktrees

The MCP server's `detectCurrentBranch()` (§6.2) works identically for worktrees. `git rev-parse --abbrev-ref HEAD` run from the worktree path returns the worktree's branch, not the main clone's branch.

No special-casing is needed: a worktree session sees `branch = "feature/X"` and scopes all searches to `feature/X`, while a main clone session sees `branch = "main"`.

### 8.3 Branch Registry

A `branch_registry` table in `state.db` tracks known branches per project:

```sql
CREATE TABLE branch_registry (
    registry_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,
    branch_name  TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    is_worktree_branch INTEGER DEFAULT 0,
    worktree_path TEXT,   -- NULL unless this branch was discovered via a worktree
    file_count   INTEGER DEFAULT 0,   -- cached count, updated on discovery
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, branch_name)
);

CREATE INDEX idx_branch_registry_folder ON branch_registry(watch_folder_id);
```

This table powers `wqm branch list` and helps the daemon avoid re-running full discovery on already-known branches.

---

## 9. Migration Path

Because this project has no released users and backward compatibility is not required, the migration is a clean schema upgrade.

### 9.1 Schema Version Bump

A new schema version (e.g., v38) is introduced. The migration runs at daemon startup via the existing `schema_version` migration framework.

**Migration steps (state.db):**

```sql
-- Step 1: Add branches column
ALTER TABLE tracked_files ADD COLUMN branches TEXT NOT NULL DEFAULT '[]';

-- Step 2: Populate branches from existing branch column
UPDATE tracked_files SET branches = json_array(branch) WHERE branch IS NOT NULL;
UPDATE tracked_files SET branches = '["default"]' WHERE branch IS NULL;

-- Step 3: Add primary_branch
ALTER TABLE tracked_files ADD COLUMN primary_branch TEXT;
UPDATE tracked_files SET primary_branch = branch;

-- Step 4: Drop old branch column (SQLite requires recreate)
-- Because SQLite does not support DROP COLUMN before v3.35, use the
-- standard recreate-rename pattern or leave branch in place as a
-- deprecated alias. Given the project's minimum SQLite version (see
-- docs/specs/12-configuration.md), use conditional DROP or accept the
-- extra column.

-- Step 5: Drop and recreate UNIQUE constraint
-- (requires table recreate in SQLite)

-- Step 6: Create branch_registry table
CREATE TABLE IF NOT EXISTS branch_registry (...);

-- Step 7: Populate branch_registry from tracked_files
INSERT OR IGNORE INTO branch_registry (watch_folder_id, branch_name, discovered_at, file_count)
SELECT DISTINCT watch_folder_id, primary_branch, datetime('now'), COUNT(*)
FROM tracked_files
WHERE primary_branch IS NOT NULL
GROUP BY watch_folder_id, primary_branch;
```

**Migration steps (search.db):**

```sql
-- Step 1: Add branches, keep old branch for transition
ALTER TABLE file_metadata ADD COLUMN branches TEXT NOT NULL DEFAULT '[]';

-- Step 2: Populate
UPDATE file_metadata SET branches = json_array(branch) WHERE branch IS NOT NULL;
UPDATE file_metadata SET branches = '["default"]' WHERE branch IS NULL;

-- Step 3: Drop branch column (or leave as alias; same SQLite version caveat)
```

**Migration steps (Qdrant):**

Qdrant payload migration is performed via scroll + bulk `set_payload`:

```
for each point in projects collection:
    if "branch" in payload and "branches" not in payload:
        set_payload({ branches: [payload["branch"]] })
```

This runs as part of the v38 migration, initiated inline in the daemon's startup sequence. Large collections are batched (1000 points per Qdrant request). The migration is idempotent: re-running sets `branches` to the same value if `branch` is still present, and is a no-op if `branches` already exists.

**Migration steps (graph.db):**

```sql
ALTER TABLE graph_nodes ADD COLUMN branches TEXT NOT NULL DEFAULT '["main"]';
ALTER TABLE graph_edges ADD COLUMN branch TEXT;
```

No data back-fill is needed for graph nodes: the `["main"]` default is conservative and correct for existing data (all graph data was indexed against `main`). Graph edges default to `NULL` (global/inferred).

### 9.2 Post-Migration Verification

After migration, the daemon logs:

```
Branch migration v38 complete:
  tracked_files rows migrated: N
  file_metadata rows migrated: M
  Qdrant points migrated: P
  graph_nodes rows migrated: G
  branch_registry entries created: B
```

A CLI diagnostic command is provided:

```bash
wqm admin verify-branch-migration
```

This checks that no `tracked_files` row has an empty `branches[]` array and that every Qdrant point in the `projects` collection has a `branches` array (not a scalar `branch` field).

---

## 10. Invariants and Constraints

These invariants hold at all times after migration is complete. Violations indicate bugs that must be fixed at root cause.

**I1 — Branches array non-empty:** Every `tracked_files` row has `json_array_length(branches) >= 1`. A file with no branches is an orphan and must be garbage-collected.

**I2 — Qdrant branches non-empty:** Every Qdrant point in the `projects` collection has `branches` as a non-empty array. Points with empty `branches` arrays are orphaned vectors that must be deleted.

**I3 — Primary branch membership:** `primary_branch` is always an element of `branches[]` in `tracked_files`. Violation indicates a bookkeeping error in the branch-addition or branch-removal path.

**I4 — Cross-store consistency:** For any `(tenant_id, relative_path, file_hash)` triple, the `branches[]` in `tracked_files`, in Qdrant point payloads, and in `file_metadata` must be identical sets. Temporary divergence during in-flight queue processing is permitted; convergence must occur within one queue drain cycle.

**I5 — Base point determinism:** `base_point = hash(tenant_id, relative_path, file_hash)`. Two files with the same `(tenant_id, relative_path, file_hash)` always produce the same `base_point` and therefore share the same Qdrant points. Different `file_hash` always produces a different `base_point` and therefore different Qdrant points.

**I6 — Worktree tenant identity:** A worktree's `tenant_id` equals its main clone's `tenant_id`. Verified at registration time in `register_project_with_disambiguation`. Violation causes cross-worktree results to appear as separate projects.

**I7 — Branch scoping by default:** All MCP tool surfaces (`search`, `grep`, `retrieve`, `list`) apply the current branch as a filter by default. The only way to see cross-branch results is to explicitly pass `branch: "*"`.

**I8 — No re-embedding for shared content:** When a branch is added to an existing Qdrant point's `branches[]`, no embedding is generated. The only Qdrant operation is `set_payload`. Embeddings are generated only for genuinely novel `(relative_path, file_hash)` combinations not yet present in `tracked_files`.

**I9 — Delete safety:** A Qdrant point is deleted only when its `branches[]` becomes empty (all branches referencing that content version have been removed). No single-branch delete operation may remove a point still referenced by another branch.

**I10 — Deletion truth table:** A branch is deleted from the index only when it no longer exists locally AND it no longer exists on any configured remote. The check uses `git branch --list` and `git ls-remote origin` as authoritative sources.

---

## Implementation Notes

**Affected components:**

| Component | Change scope |
|-----------|-------------|
| `daemon/core/src/unified_queue_processor.rs` | Branch detection at dequeue time; dedup logic in ingest/delete/update handlers |
| `daemon/core/src/code_lines_schema.rs` | Updated `CREATE_FILE_METADATA_SQL`, `UPSERT_FILE_METADATA_SQL`, `FTS5_SEARCH_BY_PROJECT_BRANCH_SQL` |
| `daemon/core/src/fts_batch_processor/processor.rs` | Updated `upsert_file_metadata` to use `branches` JSON column |
| `daemon/core/src/graph/schema.rs` | `branches` column on `graph_nodes`, `branch` column on `graph_edges` |
| `daemon/core/src/git/branch_lifecycle/mod.rs` | Hook `BranchEvent::Deleted` → deletion truth table check; `BranchEvent::Renamed` → rename procedure |
| `daemon/core/src/schema_version/v38.rs` | New migration module |
| `src/typescript/mcp-server/src/` | `detectCurrentBranch()`, branch filter in all Qdrant queries, `branch` parameter semantics update |
| `src/rust/common/src/` | Updated `fieldBranch` constant → `fieldBranches`; Qdrant payload schema update |

**Testing requirements (per new behavior):**

- Discovery algorithm: given a project with two branches sharing 80% of files, verify that 80% of files produce `set_payload` operations (no re-embedding) and 20% produce full embed+upsert.
- Deletion truth table: test each of the six cases in §7.2.
- Branch rename: verify `branches[]` arrays are updated in `tracked_files`, Qdrant, `file_metadata`, and `graph_nodes` atomically.
- FTS5 branch scoping: verify that `grep` on branch A does not return results present only on branch B.
- Graph branch scoping: verify that `wqm graph query --branch A` excludes symbols only defined on branch B.
- Cross-branch opt-out: verify that `branch: "*"` returns results from all branches.
- MCP auto-detection: verify that the detected branch changes correctly when switching branches between sessions.
- Worktree isolation: verify that a worktree session on `feature/X` does not see results from the main clone's `main` branch unless `branch: "*"` is passed.
