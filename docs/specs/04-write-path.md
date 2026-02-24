## Write Path Architecture

**Reference:** [ADR-002](../adr/ADR-002-daemon-only-write-policy.md)

### Core Rules

1. **Daemon-only Qdrant writes**: The Rust daemon (memexd) is the ONLY component that writes to Qdrant. No exceptions.
2. **Queue-only content writes**: ALL content writes from MCP/CLI go through SQLite queue. No direct gRPC for content.
3. **Direct reads**: MCP and CLI read from Qdrant directly (no daemon intermediary).

### Write Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WRITE PATH                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MCP Server                CLI                                       │
│      │                      │                                        │
│      └──────────┬───────────┘                                        │
│                 │                                                    │
│                 ▼                                                    │
│        ┌────────────────┐                                           │
│        │  SQLite Queue  │  ← ALL content writes go here             │
│        │  (unified_     │                                           │
│        │   queue)       │                                           │
│        └───────┬────────┘                                           │
│                │                                                    │
│                ▼                                                    │
│        ┌────────────────┐                                           │
│        │  Rust Daemon   │  ← Polls queue, processes items           │
│        │  (memexd)      │                                           │
│        └───────┬────────┘                                           │
│                │                                                    │
│                ▼                                                    │
│        ┌────────────────┐                                           │
│        │    Qdrant      │                                           │
│        └────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Read Flow

```
MCP Server ──→ Qdrant (direct, no daemon)
CLI (wqm) ───→ Qdrant (direct, no daemon)
```

### Session Management (Direct gRPC)

Only session lifecycle messages go directly to daemon via gRPC:

| Message                     | Direction    | Purpose                     |
| --------------------------- | ------------ | --------------------------- |
| `RegisterProject(path)`     | MCP → Daemon | Project is now active       |
| `DeprioritizeProject(path)` | MCP → Daemon | Project is no longer active |

**All other operations use the queue.**

### Collection Ownership

- **Daemon owns all collections**: Creates the 4 canonical collections on startup
- **No collection creation via MCP/CLI**: Only `projects`, `libraries`, `rules`, `scratchpad` exist
- **No user-created collections**: The 4-collection model is fixed

### gRPC Methods (Reserved)

The daemon exposes gRPC methods for content ingestion (`IngestText`, etc.) but these are **reserved for administrative/diagnostic use only**. Production code paths (MCP, CLI) must NOT use them directly - all content goes through the queue.

### Unified Queue

**ALL writes go through the SQLite queue.** The queue serves as the transaction log for daemon processing. This includes daemon file watcher events - the daemon queues its own events for centralized processing.

**All database operations MUST be enclosed in transactions** to ensure integrity with concurrent read/write access from MCP server, CLI, and daemon.

#### Priority System

Priority is **calculated at query time**, not stored in the queue:

| Item Type                            | Priority | Calculation                            |
| ------------------------------------ | -------- | -------------------------------------- |
| `rules` (any scope)                  | 1 (high) | Always high priority                   |
| `file`/`folder` for active project   | 1 (high) | JOIN with `watch_folders.is_active`    |
| `file`/`folder` for inactive project | 0 (low)  | JOIN with `watch_folders.is_active`    |
| `library`                            | 0 (low)  | Background processing                  |

**Anti-starvation mechanism (asymmetric batching):** The fairness scheduler alternates between priority directions with different batch sizes:

- **High-priority batch** (default 10): `ORDER BY priority DESC, created_at ASC` (active projects first)
- **Low-priority batch** (default 3): `ORDER BY priority ASC, created_at ASC` (inactive projects get a turn)

The asymmetric sizes (10:3) ensure ~77% of processing capacity goes to active projects while still preventing starvation. Equal batch sizes would neutralize priority advantages when library files are significantly larger than source code files.

#### Project Activity Tracking

The `watch_folders` table tracks activity state:

| Field              | Purpose                              |
| ------------------ | ------------------------------------ |
| `is_active`        | Boolean, true when project is active |
| `last_activity_at` | Timestamp, updated on any activity   |

**Activation:** MCP server sends `RegisterProject` → `is_active=true`, `last_activity_at=now()`

**Reactivation:** If already active, just update `last_activity_at=now()`

**Deactivation triggers:**

1. MCP server sends `DeprioritizeProject` (explicit sign-out)
2. Timeout from `last_activity_at` (default: 12 hours, configurable)

**Keep-alive:** MCP server checks at `timeout/4` interval (default: every 3 hours) if current project was wrongly deactivated by timeout, and reactivates it.

#### Queue Schema

```sql
CREATE TABLE unified_queue (
    -- Identity
    queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    idempotency_key TEXT NOT NULL UNIQUE,  -- SHA256 hash for deduplication

    -- Item classification
    item_type TEXT NOT NULL CHECK (item_type IN (
        'text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'
    )),
    op TEXT NOT NULL CHECK (op IN ('add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset')),
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,            -- projects|libraries|rules

    -- Processing control
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 0 AND priority <= 10),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'done', 'failed'
    )),

    -- Per-destination state machine
    qdrant_status TEXT DEFAULT 'pending' CHECK (qdrant_status IN (
        'pending', 'in_progress', 'done', 'failed'
    )),
    search_status TEXT DEFAULT 'pending' CHECK (search_status IN (
        'pending', 'in_progress', 'done', 'failed'
    )),
    decision_json TEXT,                  -- Stored decision: action, old_base_point,
                                         -- new_base_point, delete_old (boolean)

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- Lease-based crash recovery
    lease_until TEXT,                    -- Expiration timestamp for current lease
    worker_id TEXT,                      -- ID of worker holding lease

    -- Payload and metadata
    payload_json TEXT NOT NULL DEFAULT '{}',
    branch TEXT DEFAULT 'main',
    metadata TEXT DEFAULT '{}',

    -- Error handling
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    last_error_at TEXT
);

-- Primary dequeue index (pending items by priority)
CREATE INDEX idx_unified_queue_dequeue
    ON unified_queue(status, priority DESC, created_at ASC)
    WHERE status = 'pending';

-- Idempotency enforcement (unique constraint creates implicit index)
CREATE UNIQUE INDEX idx_unified_queue_idempotency
    ON unified_queue(idempotency_key);

-- Stale lease detection for crash recovery
CREATE INDEX idx_unified_queue_lease_expiry
    ON unified_queue(lease_until)
    WHERE status = 'in_progress';

-- Tenant-based queries
CREATE INDEX idx_unified_queue_collection_tenant
    ON unified_queue(collection, tenant_id);
```

**Robust design features:**

- **status column:** Derived value — tracks overall item lifecycle (pending -> in_progress -> done/failed). A queue item is `done` only when BOTH `qdrant_status = 'done'` AND `search_status = 'done'`.
- **qdrant_status / search_status:** Per-destination state machines enabling parallel execution. Qdrant and search DB execute independently with no ordering dependency between them.
- **decision_json:** Stores the keep/delete decision (computed once during the decision phase) before execution. On retry, only the failed destination is re-executed using the stored decision — no re-analysis needed.
- **lease_until/worker_id:** Enables crash recovery by detecting stale leases
- **idempotency_key:** SHA256 hash of `item_type|op|tenant_id|collection|payload_json` - prevents duplicate processing even for content items without file paths
- **priority column:** Allows different priority levels for different item types (e.g., MCP content = 8, file watch = 5)
- **updated_at:** Tracks when item status last changed
- **branch:** Preserves branch context for project items

**Per-destination processing flow:**

```
1. Decision phase (state.db transaction):
   - Evaluate reference count for old base_point
   - Record decision in decision_json (delete_old? which base points?)
   - Set qdrant_status = pending, search_status = pending
2. Qdrant execution (parallel):
   - Create/delete chunk points per decision
   - Set qdrant_status = done (or failed)
3. Search DB execution (parallel):
   - Create/delete code_lines + file_metadata per decision
   - Set search_status = done (or failed)
4. Completion:
   - Queue item status = done when BOTH destinations complete
5. Retry on failure:
   - Re-execute only the failed destination using stored decision_json
   - No re-analysis needed — decision is idempotent
```

**Decision staleness on retry:** If the decision was "keep old" because another instance referenced it, but by retry time that instance has also changed, the stale "keep" means old points linger slightly longer. The other instance's queue item handles its own cleanup. No data corruption, just delayed garbage collection.

**Idempotency key calculation:**

```
idempotency_key = SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
```

**Crash recovery:** On daemon startup, scan for `status='in_progress'` with `lease_until < now()` and reset to `pending`.

#### Folder Move Detection Strategy

**Problem:** Absolute `file_path` as unique key breaks when folders move.

**Solution:** Use `notify-debouncer-full` with `FileIdMap` + periodic validation.

**notify-debouncer-full capabilities:**

- Correlates rename events via filesystem IDs
- Memory: O(n) where n = watched files (acceptable for typical projects)
- CPU: minimal (hashmap lookups)

**Platform behavior for root folder moves:**

| Platform | Event     | Watch Follows | Paths Correct |
| -------- | --------- | ------------- | ------------- |
| macOS    | RENAME    | ❌            | ⚠️            |
| Linux    | MOVE_SELF | ⚠️            | ❌ (bug #555) |
| Windows  | None      | ⚠️            | ❌            |

**Handling strategy:**

1. **Rename detection (within same filesystem):**
   - `notify-debouncer-full` correlates MOVED_FROM + MOVED_TO via cookie
   - Daemon updates queue entries and Qdrant metadata with new paths

2. **Cross-filesystem moves:**
   - Appear as unrelated delete + create
   - Treated as deletion (unavoidable limitation)

3. **Root folder move recovery:**
   - Detect via MOVE_SELF/RENAME event or path validation failure
   - Unwatch old path, watch new path
   - Update `watch_folders` table
   - Update queue entries with new paths
   - Update Qdrant metadata via bulk `set_payload`

4. **Periodic path validation (projects only):**
   - Every hour (configurable), validate watched paths exist
   - Clock resets when folder operation notification received
   - If path doesn't exist:
     - Delete tenant from Qdrant
     - Remove all queue entries for that tenant
     - Remove entry from `watch_folders`
   - Prevents orphaned data accumulation

#### Queue Error Handling

Single daemon process - no need for complex status states.

**On processing failure:**

1. Increment `retry_count`
2. Append timestamped error to `error_message` (accumulated log)
3. Update `last_error_at` with current timestamp
4. If `retry_count >= max_retries`: set `status = 'failed'`

**Error message format:** `error_message` accumulates across retries as a newline-separated log:

```
2026-02-06T12:00:00Z Qdrant connection refused: timeout after 30s
2026-02-06T12:05:00Z Qdrant connection refused: server unavailable
2026-02-06T12:15:00Z Qdrant upsert failed: collection not found
```

Update SQL:
```sql
UPDATE unified_queue SET
  retry_count = retry_count + 1,
  error_message = COALESCE(error_message || char(10), '') || strftime('%Y-%m-%dT%H:%M:%fZ', 'now') || ' ' || :error_text,
  last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
  status = CASE WHEN retry_count + 1 >= max_retries THEN 'failed' ELSE status END,
  updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE queue_id = :queue_id;
```

**Failed items:**

- Stay in queue with `status = 'failed'`
- Skipped by normal processing (query: `WHERE status = 'pending'`)
- CLI displays failed items with full error history

**CLI commands for failed items:**

```bash
wqm queue list --failed          # List all failed items with error messages
wqm queue show <queue_id>        # Show single item with full error history
```

**Reset mechanism:** TBD once failure patterns are better understood. The important requirement is visibility - CLI must show what failed and why.

**Retry backoff:** Optional exponential backoff based on `retry_count`

#### Batch Processing Flow

**Daemon state:** Maintains `sort_ascending` flag (boolean, flips every batch)

```
1. Start SQL transaction
2. Read up to 10 elements:
   - WHERE failed = 0
   - JOIN watch_folders for priority calculation
   - ORDER BY priority [DESC|ASC based on sort_ascending], created_at ASC
3. Flip sort_ascending flag for next batch
4. Group items by (op, collection) for efficient Qdrant batching
5. For each group:
   a. Build Qdrant batch request
   b. Execute Qdrant batch (atomic within batch)
   c. On success: DELETE items from queue
   d. On failure: UPDATE retry_count++, append to errors, set failed if max reached
6. Create new queue entries (folder scans from processed items)
7. Commit SQL transaction
```

**Sort alternation (asymmetric):** Daemon alternates with different batch sizes per direction:

- **High-priority batch** (default 10): `ORDER BY priority DESC` (active projects first)
- **Low-priority batch** (default 3): `ORDER BY priority ASC` (inactive projects get a turn)

This prevents starvation while maintaining effective priority differentiation.

**Qdrant atomicity:** Each batch request to Qdrant is atomic. Grouping by (op, collection) leverages this for efficiency.

**Idempotency:** All operations are idempotent - retries are safe (delete non-existent = no-op, upsert = replace).

#### File Operation Transactions

All file operations follow a consistent transaction pattern. The SQL transaction opens
**before** the Qdrant operation. On success, the same transaction records the file
tracking state and marks the queue item done. On failure, the transaction only records
the retry information with accumulated error log.

##### File Ingest (new file)

```
BEGIN TRANSACTION;
  1. Read file from filesystem
  2. Compute file_hash (SHA256), read file_mtime
  3. Check ingestion gates: allowlist → exclusion → size limit (skip if rejected → mark done, COMMIT)
  4. Chunk file (tree-sitter or fallback overlap)
  5. Generate embeddings for all chunks
  6. Upsert all points to Qdrant (batch)
  7. If Qdrant succeeds:
       INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_type,
         language, file_mtime, file_hash, chunk_count, chunking_method,
         lsp_status, treesitter_status, created_at, updated_at) VALUES (...);
       INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash,
         chunk_type, symbol_name, start_line, end_line, created_at) VALUES (...);
         -- repeated for each chunk
       UPDATE unified_queue SET status = 'done', updated_at = ... WHERE queue_id = ?;
  8. If Qdrant fails:
       UPDATE unified_queue SET retry_count++, error_message append, last_error_at...;
       -- If retry_count >= max_retries: also SET status = 'failed'
COMMIT;
```

##### File Delete

```
BEGIN TRANSACTION;
  1. Look up file in tracked_files by (watch_folder_id, file_path, branch)
  2. Read all point_ids from qdrant_chunks for that file_id
  3. Delete points from Qdrant by point_ids (batch)
  4. If Qdrant succeeds:
       DELETE FROM qdrant_chunks WHERE file_id = ?;  -- CASCADE handles this too
       DELETE FROM tracked_files WHERE file_id = ?;
       UPDATE unified_queue SET status = 'done', updated_at = ... WHERE queue_id = ?;
  5. If Qdrant fails:
       Update queue with retry info + accumulated error.
  6. If file not found in tracked_files:
       Attempt Qdrant delete by filter (file_path + tenant_id) as fallback.
       Mark queue item done.
COMMIT;
```

##### File Update (delete + reingest)

```
BEGIN TRANSACTION;
  1. Look up existing file in tracked_files
  2. Read file from filesystem, compute new file_hash and file_mtime
  3. If file_hash unchanged → mark queue item done, COMMIT (skip processing)
  4. Read old point_ids from qdrant_chunks
  5. Chunk new file content, generate embeddings
  6. Delete old points from Qdrant (batch by point_ids)
  7. Upsert new points to Qdrant (batch)
  8. If both Qdrant operations succeed:
       DELETE FROM qdrant_chunks WHERE file_id = ?;
       UPDATE tracked_files SET file_mtime = ?, file_hash = ?, chunk_count = ?,
         chunking_method = ?, lsp_status = ?, treesitter_status = ?,
         last_error = NULL, updated_at = ... WHERE file_id = ?;
       INSERT INTO qdrant_chunks (...) VALUES (...);  -- for each new chunk
       UPDATE unified_queue SET status = 'done', updated_at = ... WHERE queue_id = ?;
  9. If Qdrant fails:
       Update queue with retry info + accumulated error.
COMMIT;
```

##### File Update — Surgical (future development)

An optimization enabled by `qdrant_chunks.content_hash`. Instead of full delete + reingest:

```
1. Read old qdrant_chunks with content_hashes for the file
2. Chunk new file content, compute content_hash for each chunk
3. Compare old vs new by content_hash:
   - Unchanged (same hash, same index): skip entirely
   - Modified (same index, different hash): upsert to Qdrant, update qdrant_chunks
   - New (index > old count): upsert to Qdrant, insert qdrant_chunks
   - Removed (old index > new count): delete from Qdrant, delete qdrant_chunks
4. Update tracked_files metadata, mark queue done
```

This reduces Qdrant operations when only part of a file changes (e.g., one function
edited in a large file).

#### Item Types

##### `rules` - Behavioral Rules

**Purpose:** LLM behavioral rules that persist across sessions.

**Writers:** MCP server (`rules` tool), CLI (`wqm rules add/update/remove`)

**Target collection:** `rules`

**Priority:** 1 (high) - always processed with active project priority

**Valid operations:**

| Operation | Description                  |
| --------- | ---------------------------- |
| `ingest`  | Add new rule                 |
| `update`  | Modify existing rule content |
| `delete`  | Remove rule                  |

**Queue fields:**

- `tenant_id`: `"global"` for global scope, or `<project_id>` for project scope
- `collection`: `"rules"`

**Payload structure:**

```json
{
  "label": "prefer-uv",
  "content": "Use uv instead of pip for Python packages",
  "scope": "global"
}
```

**Payload fields:**

- `label`: Human-readable identifier, unique within scope
- `content`: The actual rule text
- `scope`: `"global"` or `"project"` (mirrors tenant_id for validation)

**For `delete` operation:**

```json
{
  "label": "prefer-uv",
  "scope": "global"
}
```

**Idempotency key:** `SHA256(rules|<op>|<tenant_id>|rules|<payload_json>)[:32]`

##### `library` - Reference Documentation

**Purpose:** Reference documentation (books, papers, API docs, websites) - NOT programming libraries (use context7 MCP for those).

**Writers:**

- MCP server (`store` tool) - single file/webpage
- CLI (`wqm library ingest`) - single file/webpage to global library
- CLI (`wqm library add`) - register library folder → writes to `watch_folders` table
- Daemon - watches registered library folders, queues individual file operations

**Target collection:** `libraries`

**Priority:** 0 (low) - background processing

**Valid operations:**

| Operation | Description               | Writer                  |
| --------- | ------------------------- | ----------------------- |
| `ingest`  | Add new document/content  | MCP, CLI, Daemon        |
| `update`  | Replace existing document | Daemon (on file change) |
| `delete`  | Remove document           | Daemon (on file delete) |

**Tenant ID structure:**

| Context            | Format                       | Example                    | Use Case                             |
| ------------------ | ---------------------------- | -------------------------- | ------------------------------------ |
| Registered library | `folder.subfolder.filename`  | `rust-book.chapter1.intro` | Fine-grained search by hierarchy     |
| Project-specific   | `<project_id>.<payload_ref>` | `a1b2c3d4e5f6.design-spec` | Non-tracked files related to project |
| Global (catch-all) | `"global"`                   | `global`                   | Content without clear categorization |

**Queue fields:**

- `tenant_id`: See structure above
- `collection`: `"libraries"`

**Payload structure (MCP `store` / CLI single file):**

```json
{
  "content": "The actual text content...",
  "source": "user_input|web|file",
  "url": "https://...",
  "file_path": "/original/path.pdf"
}
```

**Note:** Title is not stored in payload - derived from tenant_id (filename/path) or extracted from content (first heading) during processing.

**Payload structure (Daemon from watched folder):**

```json
{
  "file_path": "/path/to/library/folder/chapter1/intro.md",
  "library_name": "rust-book",
  "relative_path": "chapter1/intro.md"
}
```

**Notes:**

- MCP/CLI provide content directly in payload (for single items)
- Daemon reads file content during processing (for watched folders)
- Dot-delimited tenant_id enables hierarchical search: `tenant_id LIKE 'rust-book.chapter1.%'`

**Idempotency key:** `SHA256(library|<op>|<tenant_id>|libraries|<payload_json>)[:32]`

##### `file` - Project/Library Source Files

**Purpose:** Individual files from watched folders (projects or libraries)

**Writers:** Daemon only (from file watcher or folder scan)

**Target collection:** `projects` or `libraries` (depending on watch type)

**Priority:** Calculated from `watch_folders.is_active` (for projects), always 0 for libraries

**Valid operations:**

| Operation | Description             | Trigger                  |
| --------- | ----------------------- | ------------------------ |
| `ingest`  | Add/update file content | File created or modified |
| `delete`  | Remove file from index  | File deleted             |

**Queue fields:**

- `tenant_id`: `<project_id>` or library tenant format
- `collection`: `"projects"` or `"libraries"`

**Payload structure:**

```json
{
  "file_path": "/absolute/path/to/file.rs",
  "relative_path": "src/main.rs"
}
```

**Notes:**

- Full path is unique in queue (first debounce level)
- Daemon computes metadata during processing (branch, file_type, language, symbols via LSP/tree-sitter)
- At processing time, daemon adapts to current state:
  - File doesn't exist but in collection → remove from collection, pop queue
  - File doesn't exist and not in collection → just pop queue
  - File exists → ingest/update as normal

##### `folder` - Directory Scan

**Purpose:** Trigger recursive scanning of a folder's contents

**Writers:** Daemon only (from folder creation event or initial registration)

**Target collection:** N/A (expands into `file` and `folder` entries)

**Priority:** Same as parent project/library

**Valid operations:**

| Operation | Description                         | Trigger                                |
| --------- | ----------------------------------- | -------------------------------------- |
| `scan`    | List folder contents and queue them | Folder created or initial registration |

**Queue fields:**

- `tenant_id`: `<project_id>` or library tenant format
- `collection`: `"projects"` or `"libraries"`

**Payload structure:**

```json
{
  "folder_path": "/absolute/path/to/folder",
  "relative_path": "src/utils"
}
```

**Processing behavior:**

- Single-level `read_dir` → immediate children only (for project collection)
- Files: check exclusion + allowlist → queue as `file` with `op=add`
- Directories: check exclusion → queue as `folder` with `op=scan`
- Excluded directories (.git, node_modules, target, etc.) are skipped entirely
- Transaction encompasses all additions + pop of scan entry
- Full path uniqueness prevents duplicate queueing

##### `tenant` - Project Lifecycle

**Purpose:** Project registration, deletion, scanning, and uplift operations

**Writers:** gRPC handlers (RegisterProject, DeleteProject), queue processor (cascade)

**Target collection:** `projects`

**Valid operations:**

| Operation | Description                        | Trigger                      |
| --------- | ---------------------------------- | ---------------------------- |
| `add`     | Register new project               | RegisterProject gRPC         |
| `delete`  | Delete project and all data        | DeleteProject gRPC           |
| `scan`    | Scan project root directory        | After tenant/add completes   |
| `uplift`  | Re-process all tenant's files      | Collection uplift cascade    |

**Processing:**
- `(Tenant, Add)`: Create collection → INSERT watch_folder → enqueue `(Tenant, Scan)`
- `(Tenant, Delete)`: Delete Qdrant points → SQLite cascade (qdrant_chunks, tracked_files, watch_folders)
- `(Tenant, Scan)`: Call `scan_directory_single_level()` on project root
- `(Tenant, Uplift)`: Query tracked_files → enqueue `(Doc, Uplift)` for each file

##### `collection` - Collection-Level Operations

**Purpose:** Bulk operations across all tenants in a collection

**Writers:** Admin operations, CLI

**Target collection:** The named collection

**Valid operations:**

| Operation | Description                              | Trigger          |
| --------- | ---------------------------------------- | ---------------- |
| `uplift`  | Re-process all content in collection     | Admin/CLI        |
| `reset`   | Delete all data, preserve configuration  | Admin/CLI        |

**Processing:**
- `(Collection, Uplift)`: Query watch_folders for all tenants → enqueue `(Tenant, Uplift)` for each
- `(Collection, Reset)`: For each tenant in collection: delete Qdrant points, DELETE qdrant_chunks + tracked_files in SQLite transaction. Watch_folder entries are preserved.

##### `website` - Website Crawl

**Purpose:** Progressive website crawling with link extraction

**Writers:** MCP server, CLI

**Target collection:** `projects` or `libraries`

**Valid operations:**

| Operation | Description                        | Trigger               |
| --------- | ---------------------------------- | --------------------- |
| `add`     | Start crawling a website           | User request          |
| `scan`    | Fetch page and extract links       | After website/add     |
| `update`  | Re-crawl website                   | User request          |
| `delete`  | Remove all website content         | User request          |

**Processing:**
- `(Website, Add)`: Validate URL → enqueue `(Website, Scan)` for root URL
- `(Website, Scan)`: Fetch HTML → extract same-domain links → enqueue `(Url, Add)` for each. Tracks visited URLs in payload metadata to prevent cycles. Respects `max_depth` and `max_pages` limits.
- `(Website, Update)`: Re-enqueue as `(Website, Scan)` for re-crawl
- `(Website, Delete)`: Delete all Qdrant points matching the website's base URL pattern

##### `url` - Individual URL Content

**Purpose:** Fetch and ingest content from a single URL

**Writers:** Website crawler, MCP server, CLI

**Target collection:** `projects` or `libraries`

**Valid operations:**

| Operation | Description             | Trigger                      |
| --------- | ----------------------- | ---------------------------- |
| `add`     | Fetch and ingest URL    | Website scan, user request   |

##### `doc` - Document-Level Operations

**Purpose:** Operations on individual tracked documents (uplift, delete)

**Writers:** Queue processor (cascade from tenant/collection operations)

**Target collection:** `projects` or `libraries`

**Valid operations:**

| Operation | Description                 | Trigger              |
| --------- | --------------------------- | -------------------- |
| `uplift`  | Re-process tracked document | Tenant uplift        |
| `delete`  | Delete tracked document     | Tenant/file deletion |

---

### Adaptive Resource Management

The daemon dynamically adjusts processing resources based on user activity and queue state.

#### Resource Modes

```
Normal → Active → RampingUp(step) → Burst
  ↑        ↑           ↑              |
  |        |           |              |
  +--------+-----------+--------------+
           (user returns)
```

| Mode | Condition | Embeddings | Delay | Description |
|------|-----------|-----------|-------|-------------|
| **Normal** | Queue empty or user active | Baseline (from config) | 50ms | Default operation |
| **Active** | Queue has work, user present | 1.5x baseline | 25ms | +50% boost for active processing |
| **RampingUp(n)** | User idle > threshold, queue has work | Interpolated | Interpolated | Gradual ramp over N steps |
| **Burst** | User idle, ramp complete | Maximum (from config) | Minimum | Full resource utilization |

#### State Transitions

- **Normal → Active**: Queue gets work while user is active
- **Active → Normal**: Queue empties while user is active
- **Active → RampingUp**: User goes idle while queue has work
- **RampingUp → Burst**: All ramp steps completed
- **RampingUp/Burst → Active**: User returns while queue has work
- **RampingUp/Burst → Normal**: User returns and queue is empty

#### Configuration

```yaml
resource_limits:
  active_concurrency_multiplier: 1.5  # +50% embeddings during active processing
  active_inter_item_delay_ms: 25      # Half of normal delay during active processing
```

#### Heartbeat Logging

Every ~60 seconds (12 polls at 5s interval), the adaptive resource manager logs:
```
Adaptive resources heartbeat: mode=Active, idle_secs=0, cpu_high=false, embeddings=3, delay=25ms
```

---

### Daemon Processing Phases

#### Phase 1: Initial Registration (Progressive Single-Level Scan)

When a new project is registered via `RegisterProject` gRPC:

```
1. gRPC handler enqueues (Tenant, Add) with project payload
2. Queue processor handles (Tenant, Add):
   a. Create collection if needed
   b. INSERT OR IGNORE watch_folder entry (with is_active from payload)
   c. Enqueue (Tenant, Scan) for the project root
3. Queue processor handles (Tenant, Scan):
   - Call scan_directory_single_level(root):
     - std::fs::read_dir for IMMEDIATE children only (not recursive)
     - Files: check exclusion + allowlist → enqueue (File, Add)
     - Directories: check exclusion → enqueue (Folder, Scan)
     - Excluded dirs (.git, node_modules, target) are skipped
4. Queue processor handles each (Folder, Scan):
   - Repeat step 3 for each subdirectory (single level)
5. Queue processor handles each (File, Add):
   - Ingest with LSP/tree-sitter metadata
   - Record in tracked_files + qdrant_chunks
   - Pop queue entry on success
6. Continues until queue empty (progressive growth, not burst)
```

**Progressive design:** Unlike recursive WalkDir which enqueues ALL files at once,
single-level scanning only enqueues immediate children per directory. This produces
gradual queue growth and avoids overwhelming the system with large projects.

**Atomic unit:** One folder level (all its direct contents queued in one transaction)

#### Phase 2: Ongoing Watching (File Changes)

Once initial scan complete, daemon watches for changes:

| Event          | Action                                                                 |
| -------------- | ---------------------------------------------------------------------- |
| File created   | Queue `file/ingest` (uniqueness prevents duplicates)                   |
| File modified  | Queue `file/update` (atomic: delete existing points + reingest)        |
| File deleted   | Queue `file/delete`                                                    |
| Folder created | Queue `folder/scan`                                                    |
| Folder deleted | Remove all files in collection with path prefix                        |

**Update atomicity:** A `file/update` is processed as a single atomic operation:
delete all existing points for the file path, then reingest current file content.
This ensures no window of inconsistency where a file has partial data.

**Queue deduplication:** The `file_path UNIQUE` constraint ensures only one operation
per file can be queued at a time. If a file already has a pending `ingest` and gets
modified, the `update` is silently dropped — the pending ingest will read the file's
current content at processing time, achieving the same result.

**Debouncing:**

1. **Queue uniqueness:** Full path can't be queued twice (first level)
2. **External debounce:** Configurable delay before queueing (second level)

**Incremental libraries:** Ignore delete events (additions and updates only)

**Processing adaptation:** At ingestion time, daemon checks current state:

- File gone but in collection → remove, pop
- File gone and not in collection → just pop
- File exists → ingest normally

#### Phase 3: Removal (Project/Library Deletion)

When a project is deleted via `DeleteProject` gRPC:

```
1. gRPC handler enqueues (Tenant, Delete) with tenant_id + collection
2. Queue processor handles (Tenant, Delete):
   a. Scroll Qdrant for all points matching tenant_id → batch delete
   b. SQLite transaction:
      - DELETE FROM qdrant_chunks WHERE file_id IN
        (SELECT file_id FROM tracked_files WHERE tenant_id = ?)
      - DELETE FROM tracked_files WHERE tenant_id = ?
      - DELETE FROM watch_folders WHERE tenant_id = ?
3. Queue entry marked done
```

**Enqueue-only pattern:** gRPC handlers never perform direct database mutations.
All destructive operations are routed through the unified queue for consistency
and crash recovery.

#### Phase 4: Daemon Startup Automation

On daemon start (or restart), the daemon runs a 6-step startup sequence before entering
its normal event loop. This handles schema migrations, configuration changes, state
reconciliation, and crash recovery.

```
Step 1: Schema Integrity Check
   - Verify all core tables exist (schema_version, unified_queue, watch_folders,
     tracked_files, qdrant_chunks, search_events, resolution_events,
     sparse_vocabulary, corpus_statistics)
   - Run pending migrations if schema_version is behind
   - ABORT startup if schema check fails (cannot proceed without tables)

Step 2: Configuration Change Reconciliation
   - Compute config fingerprint = SHA256(sorted(allowed_extensions) + sorted(allowed_filenames)
     + sorted(exclude_directories) + sorted(exclude_patterns))
   - Compare fingerprint with stored value in schema_version metadata
   - If fingerprint differs (config changed since last run):
     a. For files in tracked_files that are NOW excluded (new exclusion rule):
        → queue file/delete for each
     b. For files in tracked_files whose extension is NO LONGER on allowlist:
        → queue file/delete for each
     c. Store new fingerprint
   - NOTE: Newly-allowed files are discovered during Step 5 filesystem walk

Step 3: Qdrant Collection Verification
   - Ensure 4 canonical collections exist: projects, libraries, rules, scratchpad
   - Create any missing collections with correct vector configuration
   - Verify named vector configuration (dense + sparse) matches expectations

Step 4: Watch Folder Path Validation
   - For each watch_folder entry WHERE enabled = 1:
     a. Validate path exists on filesystem
     b. If path invalid: set enabled = 0, deactivate, log warning
   - This catches projects that were moved or deleted while daemon was stopped

Step 5: Filesystem Recovery (tracked_files reconciliation)
   - For each watch_folder WHERE enabled = 1:
     a. Query tracked_files for all files with this watch_folder_id
     b. Walk filesystem to get current eligible files with mtime + hash
        (eligible = passes allowlist + exclusion + size gates)
     c. Compare (file_path is relative to watch_folder.path):
        - In tracked_files but not on disk → queue file/delete
        - On disk but not in tracked_files → queue file/ingest
        - In both but file_mtime or file_hash changed → queue file/update
        - In both and unchanged → skip (no action)
   - For first startup with empty tracked_files, the initial scan (Phase 1)
     handles population

Step 6: Crash Recovery
   - Reset stale in_progress queue items:
     WHERE status = 'in_progress' AND lease_expires_at < now()
     SET status = 'pending', leased_by = NULL, lease_expires_at = NULL,
         retry_count = retry_count + 1
   - Items exceeding max_retries are set to status = 'failed'
```

**Performance:** Steps 1-4 query SQLite only (milliseconds). Step 5 performs filesystem
walks but compares against SQLite (fast). Step 6 is a single SQL UPDATE.

**Scan distinction:** Initial scan (Phase 1) is for newly registered projects only.
Startup automation (Phase 4) runs on every daemon startup for all existing watched projects.
The file watcher (Phase 2) handles all changes during normal daemon operation.

### Daemon Watch Management

The daemon manages filesystem watches based on the `watch_folders` table.

#### Startup

```
1. Read all entries from watch_folders WHERE enabled = 1
2. For each folder:
   a. Validate path exists
   b. Set up recursive filesystem watch (notify crate)
   c. If folder not yet scanned (last_scan IS NULL):
      Queue folder for initial scan (folder/scan)
3. Start queue processor loop
4. Start watch_folders polling loop
```

#### Runtime: New Folder Registered

The daemon polls `watch_folders` table periodically (default: every 5 seconds) for changes.

```
1. Detect new entry or updated_at changed
2. If enabled = 1 and not already watching:
   a. Set up recursive filesystem watch
   b. Queue folder for initial scan
3. If enabled = 0 and currently watching:
   a. Remove filesystem watch
   b. Optionally trigger Phase 3 cleanup (if configured)
```

#### Runtime: Folder Unregistered

When a watch entry is deleted or disabled:

```
1. Remove filesystem watch for that path
2. If cleanup_on_disable configured:
   a. Delete all content from Qdrant for that tenant
   b. Update watch_folders.last_scan = NULL
```

#### Runtime: Pause/Resume

Global pause halts file event processing for maintenance or backups. CLI writes directly to SQLite; daemon detects changes via polling.

**Pause (`wqm watch pause`):**
```
1. CLI sets is_paused = 1 and pause_start_time on all enabled watches
2. Daemon polls DB every 5 seconds, detects change
3. In-memory AtomicBool pause flag set to true
4. FileWatcher buffers incoming events (up to 10K capacity, FIFO eviction)
5. Queue processor skips paused items
```

**Resume (`wqm watch resume`):**
```
1. CLI sets is_paused = 0 and clears pause_start_time on all paused watches
2. Daemon polls DB, detects change
3. In-memory pause flag set to false
4. FileWatcher drains buffered events into normal processing
5. Queue processor resumes normal operation
```

**gRPC alternative:** `PauseAllWatchers` / `ResumeAllWatchers` RPCs update DB and flag atomically.

**Persistence:** Pause state survives daemon restarts. On startup, daemon calls `poll_pause_state()` to restore the flag from DB.

**Diagnostic entries:** Each pause/resume writes a metadata entry to `unified_queue` for audit.

#### Watch Folders Table Reference

See [Watch Folders Table (Unified)](02-collection-architecture.md#watch-folders-table-unified) in the Collection Architecture section for the complete schema.

**Daemon polling query:**

```sql
SELECT * FROM watch_folders
WHERE updated_at > :last_poll_time OR enabled != :cached_enabled_state
```

**Item Types (MCP-relevant):**

| item_type | Used By             | payload_json                                  |
| --------- | ------------------- | --------------------------------------------- |
| `rules`   | MCP `rules` tool    | `{label, content, scope, project_id}`         |
| `library` | MCP `store` tool    | `{library_name, content, title, source, url}` |
| `project` | MCP `store` tool    | `{path, name}` (registers via gRPC, not queue) |
| `file`    | Daemon file watcher | `{file_path, ...}`                            |
| `folder`  | Daemon folder scan  | `{folder_path, patterns, ...}`                |

### Idempotency

All queue operations use SHA256-based idempotency keys:

```
idempotency_key = SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
```

### Queue Response

When operations are queued:

```json
{
  "success": true,
  "status": "queued",
  "message": "Operation queued for daemon processing.",
  "queue_id": "abc123"
}
```

---

