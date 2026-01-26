-- Rollback script for migration 003 (development only).
-- WARNING: This will delete all unified_queue data.

DROP INDEX IF EXISTS idx_unified_queue_worker;
DROP INDEX IF EXISTS idx_unified_queue_failed;
DROP INDEX IF EXISTS idx_unified_queue_item_type;
DROP INDEX IF EXISTS idx_unified_queue_collection_tenant;
DROP INDEX IF EXISTS idx_unified_queue_lease_expiry;
DROP INDEX IF EXISTS idx_unified_queue_idempotency;
DROP INDEX IF EXISTS idx_unified_queue_dequeue;

DROP TABLE IF EXISTS unified_queue;

DELETE FROM schema_version WHERE version = 13;
