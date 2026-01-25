# ADR-002: Daemon-Only Write Policy

**Status:** Accepted
**Date:** 2026-01-25
**Deciders:** Architecture Team
**Supersedes:** ADR-001 Section 3 (Daemon-Only Writes), CONSOLIDATED_PRD_V2.md fallback behavior

## Context

The workspace-qdrant-mcp system has evolved its write architecture through several iterations:

1. **v0.1-v0.2**: MCP server could write directly to Qdrant
2. **v0.3**: Introduced daemon as preferred writer with direct Qdrant fallback
3. **Current state**: ADR-001 and CONSOLIDATED_PRD_V2.md still document direct Qdrant writes as fallback

This created ambiguity:
- ADR-001 shows SQLite queue fallback AND direct Qdrant write fallback
- CONSOLIDATED_PRD_V2.md (line 50-52) states "When the protocol does not work for a query, we fallback to direct access"
- First Principle 10 states "Daemon-Only Writes" but code had exceptions

The code audit (2026-01-25) identified this as a critical consistency issue requiring clarification.

## Decision

**The Rust daemon (memexd) is the ONLY component that writes to Qdrant. Period.**

### Write Path Architecture

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
│        │  SQLite State  │  ← All write requests go here             │
│        │  (ingestion    │                                           │
│        │   queue)       │                                           │
│        └───────┬────────┘                                           │
│                │                                                    │
│                ▼                                                    │
│        ┌────────────────┐                                           │
│        │  Rust Daemon   │  ← ONLY component that writes to Qdrant  │
│        │  (memexd)      │                                           │
│        └───────┬────────┘                                           │
│                │                                                    │
│                ▼                                                    │
│        ┌────────────────┐                                           │
│        │    Qdrant      │                                           │
│        └────────────────┘                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Rules

1. **MCP Server**: Writes ONLY to SQLite state/queue database
2. **CLI (wqm)**: Writes ONLY to SQLite state/queue database
3. **Daemon (memexd)**: Processes SQLite queue and writes to Qdrant
4. **Direct Qdrant writes**: PROHIBITED from MCP/CLI (no exceptions for writes)
5. **Direct Qdrant reads**: ALLOWED from MCP (search operations)

### Fallback Behavior

When daemon is unavailable:

| Operation | Behavior |
|-----------|----------|
| Write request | Enqueue to SQLite, return success with "queued" status |
| Search request | Query Qdrant directly (reads allowed) |
| Project search | Denied if daemon unavailable (requires active session) |
| Library search | Allowed (no daemon dependency for reads) |
| Memory query | Denied if daemon unavailable (requires consistency guarantee) |

**SQLite Queue Requirements:**
- All mutations MUST be transactional (ACID guarantees)
- Queue uses WAL mode for crash recovery
- Items marked `in_progress` revert to `pending` on daemon restart
- Idempotency keys prevent duplicate processing

### Collection Scope

This policy applies to ALL collections:

| Collection | Write Path | Notes |
|------------|-----------|-------|
| `projects` | Daemon only | Project code, docs, configs |
| `libraries` | Daemon only | External documentation |
| `memory` | Daemon only | LLM rules (previously had exception) |

**Removed Exception:** ADR-001 noted "Memory collection allows direct writes (meta-level data)." This exception is REMOVED. All writes, including memory rules, route through the daemon.

### Error Handling

When daemon unavailable and write requested:

```python
# MCP Server Response
{
    "success": true,
    "status": "queued",
    "message": "Content queued for processing. Daemon will ingest when available.",
    "queue_id": "uuid-of-queue-item"
}
```

When daemon unavailable and project search requested:

```python
# MCP Server Response
{
    "success": false,
    "error": "daemon_unavailable",
    "message": "Project search requires active daemon session. Start daemon with: wqm service start",
    "suggestion": "Use scope='library' for daemon-independent search"
}
```

## Rationale

### Why No Direct Qdrant Writes?

1. **Data Consistency**: Single writer eliminates race conditions
2. **Metadata Integrity**: Daemon computes all metadata (project_id, branch, symbols)
3. **Embedding Consistency**: Daemon uses consistent embedding model (Qdrant FastEmbed)
4. **Audit Trail**: All writes logged through single path
5. **Resource Control**: Daemon manages connection pooling and rate limiting

### Why Remove Memory Exception?

1. **Consistency**: Same rules for all collections simplifies reasoning
2. **Metadata**: Memory rules benefit from daemon metadata enrichment
3. **Audit**: Memory changes should be tracked like other writes
4. **Simplicity**: One code path for all writes

### Why SQLite Queue?

1. **Durability**: Survives daemon restarts and crashes
2. **ACID**: Transactional guarantees for queue operations
3. **Performance**: Low latency for enqueue operations
4. **Simplicity**: No additional infrastructure (Redis, RabbitMQ)
5. **Crash Recovery**: WAL mode enables recovery after failures

## Implementation

### Python MCP Server Changes

Remove from `server.py`:
- Direct `qdrant_client.upsert()` calls in fallback paths
- Local embedding generation in fallback paths
- Any code path that writes to Qdrant without daemon

Add to `server.py`:
- SQLite queue enqueue for all write operations
- Appropriate error messages when daemon unavailable
- Queue status in response metadata

### Rust Daemon Changes

Enable in `src/rust/daemon/core/src/lib.rs`:
- Uncomment `queue_processor` module
- Implement queue polling and processing
- Add crash recovery logic

### Configuration

```yaml
# Queue processor settings
queue:
  poll_interval_ms: 1000    # How often to check queue
  batch_size: 10            # Items per processing batch
  max_retries: 3            # Retry count before dead letter
  retry_backoff_ms: 1000    # Base backoff between retries
```

## Consequences

### Positive

- Single source of truth for writes
- Consistent metadata across all documents
- Simplified debugging (one write path)
- Better crash recovery
- Clear ownership boundaries

### Negative

- Write latency increased (queue + daemon processing)
- Daemon must be running for real-time ingestion
- More complex error handling in MCP

### Risks

- Queue backup if daemon offline for extended period
- User confusion about "queued" vs "ingested" status
- Need to monitor queue depth

## Migration

1. **Code Audit**: Grep for direct Qdrant writes, remove all
2. **Enable Queue Processor**: Uncomment in daemon `lib.rs`
3. **Update Error Messages**: Guide users to start daemon
4. **Documentation**: Update CLAUDE.md, README.md
5. **Testing**: Verify no writes bypass daemon

## Related Documents

- ADR-001: Canonical Collection Architecture (Section 3 superseded)
- FIRST-PRINCIPLES.md: Principle 10 (Daemon-Only Writes)
- CONSOLIDATED_PRD_V2.md: Lines 50-52 (to be updated)
- `code_audit.md`: 2026-01-25 findings

## Changelog

| Date | Change |
|------|--------|
| 2026-01-25 | Initial ADR created, supersedes ADR-001 Section 3 |
