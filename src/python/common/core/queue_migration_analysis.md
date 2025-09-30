# SQLite Queue Schema Migration Analysis

**Document Version:** 1.0
**Date:** 2025-09-30
**Purpose:** Analyze existing SQLite schema and plan migration to enhanced queue system

---

## Executive Summary

This document analyzes the existing SQLite state management schema and provides a comprehensive mapping to the new enhanced queue system. The migration involves consolidating queue functionality from `processing_queue` and `file_processing` tables into the new `ingestion_queue`, `collection_metadata`, and `messages` tables.

---

## Current Schema Analysis

### 1. Existing `processing_queue` Table

**Current Structure:**
```sql
CREATE TABLE processing_queue (
    queue_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 2,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    scheduled_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    attempts INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,  -- JSON
    FOREIGN KEY (file_path) REFERENCES file_processing (file_path) ON DELETE CASCADE
)
```

**Indexes:**
- `idx_processing_queue_priority`: (priority DESC, scheduled_at ASC)
- `idx_processing_queue_file_path`: (file_path)
- `idx_processing_queue_scheduled_at`: (scheduled_at)

**Current Priority System:**
- Uses `ProcessingPriority` enum: LOW=1, NORMAL=2, HIGH=3, URGENT=4
- Default priority: 2 (NORMAL)

### 2. Existing `file_processing` Table

**Current Structure:**
```sql
CREATE TABLE file_processing (
    file_path TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 2,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    file_size INTEGER,
    file_hash TEXT,
    document_id TEXT,
    metadata TEXT,  -- JSON
    -- LSP-specific fields
    language_id TEXT,
    lsp_extracted BOOLEAN NOT NULL DEFAULT 0,
    symbols_count INTEGER DEFAULT 0,
    lsp_server_id INTEGER,
    last_lsp_analysis TIMESTAMP,
    lsp_metadata TEXT,  -- JSON
    FOREIGN KEY (lsp_server_id) REFERENCES lsp_servers (id) ON DELETE SET NULL
)
```

**Status Values:**
- PENDING, PROCESSING, COMPLETED, FAILED, SKIPPED, RETRYING, OCR_REQUIRED

### 3. Existing `error_log` Table

**Current Structure:**
```sql
CREATE TABLE error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    source TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON
)
```

**Indexes:**
- `idx_error_log_type`: (error_type)
- `idx_error_log_source`: (source)
- `idx_error_log_timestamp`: (timestamp)

---

## Schema Mapping: Old → New

### Mapping 1: `processing_queue` → `ingestion_queue`

| Old Field | New Field | Transformation | Notes |
|-----------|-----------|----------------|-------|
| `queue_id` | _(removed)_ | N/A | New PK is `file_absolute_path` |
| `file_path` | `file_absolute_path` | Direct copy | Becomes PRIMARY KEY |
| `collection` | `collection_name` | Direct copy | Renamed for consistency |
| `priority` | `priority` | Scale: 1-4 → 0-10 | Map: 1→2, 2→5, 3→7, 4→9 |
| `created_at` | `queued_timestamp` | Direct copy | Renamed for clarity |
| `scheduled_at` | _(removed)_ | N/A | Not needed in new schema |
| `attempts` | `retry_count` | Direct copy | Same semantics |
| `metadata` | _(see below)_ | Extract fields | Parse JSON for tenant_id, branch |
| _(new)_ | `tenant_id` | Extract from metadata | Default: 'default' |
| _(new)_ | `branch` | Extract from metadata | Default: 'main' |
| _(new)_ | `operation` | Infer from status | Default: 'ingest' |
| _(new)_ | `retry_from` | NULL | No historical retry chains |
| _(new)_ | `error_message_id` | Link if error exists | Join with error_log |

**Priority Scaling Function:**
```python
def scale_priority(old_priority: int) -> int:
    """Map old 1-4 scale to new 0-10 scale."""
    mapping = {1: 2, 2: 5, 3: 7, 4: 9}
    return mapping.get(old_priority, 5)  # Default to 5 if unknown
```

### Mapping 2: `file_processing` → Supporting Data

The `file_processing` table will **remain** but some data will inform queue operations:

| Field | Usage in Migration | Notes |
|-------|-------------------|-------|
| `error_message` | Create `messages` entry | If status=FAILED |
| `retry_count` | Copy to `ingestion_queue.retry_count` | If queued |
| `status` | Determine operation | PENDING→ingest, PROCESSING→skip |

### Mapping 3: `error_log` → `messages`

| Old Field | New Field | Transformation | Notes |
|-----------|-----------|----------------|-------|
| `id` | `id` | Direct copy | Preserve IDs |
| `error_type` | `error_type` | Direct copy | Same |
| `error_message` | `error_message` | Direct copy | Same |
| `metadata` | `error_details` | Direct copy | Renamed |
| `timestamp` | `occurred_timestamp` | Direct copy | Renamed |
| `source` | `file_path` | Parse source | Extract file path if present |
| _(new)_ | `collection_name` | Extract from source | Parse or lookup |
| _(new)_ | `retry_count` | 0 | Initialize to 0 |

---

## Data Transformation Requirements

### 1. Priority Rescaling

**Requirement:** Convert 4-level priority system to 10-level system.

**Implementation:**
```python
PRIORITY_MAPPING = {
    1: 2,   # LOW → 2 (slightly above minimum)
    2: 5,   # NORMAL → 5 (middle of range)
    3: 7,   # HIGH → 7 (above middle)
    4: 9,   # URGENT → 9 (near maximum)
}

def migrate_priority(old_priority: int) -> int:
    return PRIORITY_MAPPING.get(old_priority, 5)
```

### 2. Metadata Extraction

**Requirement:** Parse JSON metadata to extract tenant_id and branch.

**Implementation:**
```python
import json

def extract_metadata_fields(metadata_json: str) -> dict:
    """Extract tenant_id and branch from metadata JSON."""
    if not metadata_json:
        return {'tenant_id': 'default', 'branch': 'main'}

    try:
        metadata = json.loads(metadata_json)
        return {
            'tenant_id': metadata.get('tenant_id', 'default'),
            'branch': metadata.get('branch', 'main'),
        }
    except json.JSONDecodeError:
        return {'tenant_id': 'default', 'branch': 'main'}
```

### 3. Operation Type Inference

**Requirement:** Determine operation type from processing status.

**Implementation:**
```python
def infer_operation(status: str) -> str:
    """Infer operation type from processing status."""
    status_upper = status.upper()

    if status_upper in ('PENDING', 'RETRYING'):
        return 'ingest'
    elif status_upper == 'PROCESSING':
        return 'update'  # Assume ongoing processing is an update
    else:
        return 'ingest'  # Default
```

### 4. Error Message Linkage

**Requirement:** Link queue items to error messages.

**Implementation:**
```python
async def link_error_messages(conn, file_path: str, collection: str) -> Optional[int]:
    """Find and link most recent error for a file."""
    cursor = conn.execute("""
        SELECT id FROM error_log
        WHERE source LIKE ? OR metadata LIKE ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (f'%{file_path}%', f'%{file_path}%'))

    result = cursor.fetchone()
    return result[0] if result else None
```

---

## Migration Risks and Incompatibilities

### Risk 1: Data Loss - Queue IDs
**Severity:** LOW
**Description:** Old `queue_id` is not preserved in new schema
**Mitigation:** Store mapping in temporary table during migration
**Impact:** Historical queue ID references in logs become invalid

### Risk 2: Priority Interpretation
**Severity:** MEDIUM
**Description:** Priority rescaling may affect processing order temporarily
**Mitigation:** Clear queue before migration or migrate during low-activity period
**Impact:** Items may be processed out of expected order during transition

### Risk 3: Metadata Field Loss
**Severity:** LOW
**Description:** Metadata fields not mapped to tenant_id/branch are lost
**Mitigation:** Preserve full metadata in archived table
**Impact:** Custom metadata fields in old queue items not accessible

### Risk 4: Foreign Key Constraints
**Severity:** MEDIUM
**Description:** New FK constraints (retry_from, error_message_id) may fail on existing data
**Mitigation:** Set to NULL during migration, populate later if needed
**Impact:** Retry chain history is not preserved

### Risk 5: Concurrent Access During Migration
**Severity:** HIGH
**Description:** Daemon or MCP server accessing queue during migration causes corruption
**Mitigation:** **STOP ALL SERVICES** before migration. Use WAL checkpoint.
**Impact:** Data corruption if not handled properly

---

## Sample Data Extraction

### Sample 1: Processing Queue Item

**Current Data:**
```sql
SELECT * FROM processing_queue LIMIT 1;
-- Result:
-- queue_id: pq_1234567890_12345
-- file_path: /Users/chris/project/src/main.py
-- collection: my-project-code
-- priority: 2
-- created_at: 2025-09-30 10:30:00
-- scheduled_at: 2025-09-30 10:30:00
-- attempts: 0
-- metadata: {"user_triggered": true}
```

**After Migration:**
```sql
SELECT * FROM ingestion_queue WHERE file_absolute_path = '/Users/chris/project/src/main.py';
-- Expected Result:
-- file_absolute_path: /Users/chris/project/src/main.py
-- collection_name: my-project-code
-- tenant_id: default
-- branch: main
-- operation: ingest
-- priority: 5
-- queued_timestamp: 2025-09-30 10:30:00
-- retry_count: 0
-- retry_from: NULL
-- error_message_id: NULL
```

### Sample 2: File with Error

**Current Data:**
```sql
-- From processing_queue:
-- queue_id: pq_1234567891_67890
-- file_path: /Users/chris/project/src/broken.py
-- priority: 3
-- attempts: 2

-- From error_log:
-- id: 42
-- error_type: PARSE_ERROR
-- error_message: Failed to parse Python file
-- source: file_processor
-- metadata: {"file": "/Users/chris/project/src/broken.py"}
```

**After Migration:**
```sql
-- ingestion_queue:
-- file_absolute_path: /Users/chris/project/src/broken.py
-- collection_name: my-project-code
-- priority: 7
-- retry_count: 2
-- error_message_id: 42

-- messages (migrated from error_log):
-- id: 42
-- error_type: PARSE_ERROR
-- error_message: Failed to parse Python file
-- file_path: /Users/chris/project/src/broken.py
-- collection_name: my-project-code
-- retry_count: 2
```

---

## Collection Type Detection

Since the current schema doesn't have explicit collection types, we need to infer them:

### Detection Logic

```python
def detect_collection_type(collection_name: str, watch_folders: list) -> str:
    """Infer collection type from name and watch folder data."""

    # Check if collection is associated with a watch folder
    watched = any(wf.collection == collection_name for wf in watch_folders)

    if watched:
        # Determine if dynamic or cumulative based on patterns
        # Dynamic: watch folders with frequent changes (code, docs)
        # Cumulative: watch folders with append-only data (logs, archives)
        return 'watched-dynamic'  # Default for watched collections

    # Check naming patterns
    if 'project' in collection_name.lower():
        return 'project'

    # Default to non-watched
    return 'non-watched'
```

---

## Migration Validation Queries

### Validation 1: Count Preservation
```sql
-- Before migration
SELECT COUNT(*) as old_count FROM processing_queue;

-- After migration
SELECT COUNT(*) as new_count FROM ingestion_queue;

-- Should match
```

### Validation 2: Priority Distribution
```sql
-- Old priority distribution
SELECT priority, COUNT(*) FROM processing_queue GROUP BY priority;

-- New priority distribution (should reflect scaling)
SELECT priority, COUNT(*) FROM ingestion_queue GROUP BY priority;
```

### Validation 3: Error Linkage
```sql
-- Count items with errors before
SELECT COUNT(*) FROM file_processing WHERE error_message IS NOT NULL;

-- Count items with errors after
SELECT COUNT(*) FROM ingestion_queue WHERE error_message_id IS NOT NULL;
```

### Validation 4: Foreign Key Integrity
```sql
-- Check all FKs are valid
SELECT COUNT(*) FROM ingestion_queue
WHERE retry_from IS NOT NULL
  AND retry_from NOT IN (SELECT file_absolute_path FROM ingestion_queue);
-- Should be 0

SELECT COUNT(*) FROM ingestion_queue
WHERE error_message_id IS NOT NULL
  AND error_message_id NOT IN (SELECT id FROM messages);
-- Should be 0
```

---

## Rollback Strategy

### Phase 1: Backup
```bash
# Create full backup before migration
cp workspace_state.db workspace_state.db.backup.$(date +%Y%m%d_%H%M%S)
```

### Phase 2: Migration with Rollback Point
```sql
-- Create savepoint before migration
SAVEPOINT pre_migration;

-- ... perform migration ...

-- If validation fails:
ROLLBACK TO SAVEPOINT pre_migration;

-- If validation succeeds:
RELEASE SAVEPOINT pre_migration;
```

### Phase 3: Old Table Preservation
```sql
-- Rename old tables instead of dropping
ALTER TABLE processing_queue RENAME TO processing_queue_old;
ALTER TABLE error_log RENAME TO error_log_old;

-- Keep for 30 days, then:
-- DROP TABLE processing_queue_old;
-- DROP TABLE error_log_old;
```

---

## Next Steps

1. **Implement migration script** (Task 344.6)
2. **Configure WAL mode** (Task 344.7)
3. **Test migration with sample data** (Local testing)
4. **Perform migration in staging environment**
5. **Validate with comprehensive tests**
6. **Execute production migration with maintenance window**

---

## Appendix: Field Compatibility Matrix

| Old Field | New Field | Compatible | Transformation Required | Risk Level |
|-----------|-----------|------------|------------------------|------------|
| queue_id | - | ❌ | N/A - not migrated | LOW |
| file_path | file_absolute_path | ✅ | None | LOW |
| collection | collection_name | ✅ | None | LOW |
| priority (1-4) | priority (0-10) | ⚠️ | Scale mapping | MEDIUM |
| created_at | queued_timestamp | ✅ | None | LOW |
| scheduled_at | - | ❌ | N/A - not migrated | LOW |
| attempts | retry_count | ✅ | None | LOW |
| metadata | tenant_id, branch | ⚠️ | JSON parsing | MEDIUM |
| - | operation | ❌ | Inference required | MEDIUM |
| - | retry_from | ❌ | NULL (no history) | LOW |
| error_message | error_message_id | ⚠️ | Linkage required | MEDIUM |

**Legend:**
- ✅ Direct compatibility
- ⚠️ Transformation required
- ❌ No direct equivalent

---

**Document Status:** COMPLETE
**Review Required:** Yes
**Approved By:** Pending
