-- Migration: Create unified_queue table
-- Version: 13
-- Date: 2026-01-25
-- Purpose: Add unified_queue for consolidated ingestion operations

-- LEGACY TABLES (deprecated, kept for Phase 1 compatibility):
-- content_ingestion_queue and ingestion_queue remain unchanged. New code should use unified_queue.
-- These will be removed in a later phase.

-- unified_queue replaces content_ingestion_queue and ingestion_queue
CREATE TABLE IF NOT EXISTS unified_queue (
    queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    item_type TEXT NOT NULL CHECK (item_type IN (
        'content', 'file', 'folder', 'project', 'library',
        'delete_tenant', 'delete_document', 'rename'
    )),
    op TEXT NOT NULL CHECK (op IN ('ingest', 'update', 'delete', 'scan')),
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 0 AND priority <= 10),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'done', 'failed'
    )),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    lease_until TEXT,
    worker_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL DEFAULT '{}',
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    last_error_at TEXT,
    branch TEXT DEFAULT 'main',
    metadata TEXT DEFAULT '{}'
);

-- Indexes for unified_queue - optimized for common access patterns
-- Fast priority-based dequeue (partial index for pending items only)
CREATE INDEX IF NOT EXISTS idx_unified_queue_dequeue
    ON unified_queue(status, priority DESC, created_at ASC)
    WHERE status = 'pending';

-- Unique index for idempotency (explicit index in addition to UNIQUE constraint)
CREATE UNIQUE INDEX IF NOT EXISTS idx_unified_queue_idempotency
    ON unified_queue(idempotency_key);

-- Lease expiry detection (partial index for in_progress items)
CREATE INDEX IF NOT EXISTS idx_unified_queue_lease_expiry
    ON unified_queue(lease_until)
    WHERE status = 'in_progress';

-- Per-project/tenant queries
CREATE INDEX IF NOT EXISTS idx_unified_queue_collection_tenant
    ON unified_queue(collection, tenant_id);

-- Type distribution analysis
CREATE INDEX IF NOT EXISTS idx_unified_queue_item_type
    ON unified_queue(item_type, status);

-- Failed item monitoring (partial index for failed items)
CREATE INDEX IF NOT EXISTS idx_unified_queue_failed
    ON unified_queue(status, last_error_at DESC)
    WHERE status = 'failed';

-- Worker tracking (partial index for in_progress items)
CREATE INDEX IF NOT EXISTS idx_unified_queue_worker
    ON unified_queue(worker_id, status)
    WHERE status = 'in_progress';

-- Schema version 13: unified queue table with item_type and op enums
INSERT OR IGNORE INTO schema_version (version) VALUES (13);
