# Queue Performance Tuning Guide

Performance optimization guide for SQLite queue system targeting 1000+ documents/minute throughput.

## Performance Targets

- **Throughput**: 1000+ docs/min (16.67 ops/sec minimum)
- **Latency**: Mean <50ms, P95 <100ms, P99 <200ms
- **Memory**: Peak <256MB under load
- **Error Rate**: <1%

## SQLite Configuration Parameters

### Connection Pool Settings

```python
# queue_connection.py - ConnectionConfig
busy_timeout = 30.0          # Seconds to wait when database is locked
max_connections = 10         # Maximum pool size
min_connections = 2          # Minimum idle connections
connection_timeout = 30.0    # Seconds to wait for available connection
```

**Tuning**:
- Increase `max_connections` (10-20) for high concurrent load
- Increase `busy_timeout` (30-60s) if seeing lock contention
- Reduce `min_connections` (1-2) to save memory in low-load scenarios

### WAL Mode Parameters

```python
# queue_connection.py - ConnectionConfig
wal_autocheckpoint = 1000         # Pages before auto-checkpoint
wal_checkpoint_interval = 300     # Seconds between forced checkpoints
```

**Tuning**:
- Increase `wal_autocheckpoint` (2000-5000) for write-heavy workloads
- Decrease `wal_checkpoint_interval` (60-120s) if WAL grows too large
- Monitor WAL file size: should stay under 10-20MB

### Cache and Memory Settings

```python
# queue_connection.py - ConnectionConfig
cache_size = 10000           # Pages (~40MB with 4KB pages)
mmap_size = 268435456        # 256MB memory-mapped I/O
```

**Tuning**:
- Increase `cache_size` (20000-50000) if sufficient RAM available
- Increase `mmap_size` (512MB-1GB) for read-heavy workloads
- Monitor memory usage: cache_size × 4KB = RAM usage

### Synchronous Mode

```python
# queue_connection.py - ConnectionConfig
synchronous = "NORMAL"       # NORMAL, FULL, or OFF
```

**Options**:
- `NORMAL`: Good balance (recommended with WAL)
- `FULL`: Maximum durability, slower writes
- `OFF`: Fastest, risk data loss on crash (not recommended)

## Performance Optimization Strategies

### 1. Batch Operations

Use `enqueue_batch()` instead of individual `enqueue_file()` calls:

```python
# SLOW: Individual operations
for file_path in files:
    await client.enqueue_file(file_path, collection, priority)

# FAST: Batch operations
items = [{"file_path": f, "collection": c, "priority": p} for f in files]
await client.enqueue_batch(items)
```

**Impact**: 5-10x throughput improvement for bulk operations

### 2. Queue Depth Management

Configure queue depth limits to prevent memory exhaustion:

```python
# Limit queue size with overflow handling
await client.enqueue_batch(
    items,
    max_queue_depth=10000,
    overflow_strategy="replace_lowest"  # or "reject"
)
```

**Recommendation**: Set limit to 10-20x expected concurrent processing capacity

### 3. Priority-Based Processing

Use priority levels effectively to ensure critical items process first:

```python
# Priority scale: 0 (lowest) to 10 (highest)
await client.enqueue_file(file_path, collection, priority=9)  # Urgent
```

**Guidelines**:
- 0-2: Low priority background tasks
- 3-5: Normal priority
- 6-8: High priority user-initiated
- 9-10: Urgent/critical operations

### 4. Dequeue Batch Size

Tune dequeue batch size based on processing speed:

```python
# Adjust batch size to match processing capacity
items = await client.dequeue_batch(batch_size=50)
```

**Recommendation**:
- Fast processing (< 100ms/item): batch_size = 50-100
- Moderate (100-500ms/item): batch_size = 10-20
- Slow (> 500ms/item): batch_size = 1-5

### 5. Index Optimization

Ensure critical indexes are present (automatically created by schema):

```sql
-- Priority + timestamp index (critical for dequeue)
CREATE INDEX idx_priority_timestamp
ON ingestion_queue(priority DESC, queued_timestamp ASC);

-- Collection lookup
CREATE INDEX idx_collection
ON ingestion_queue(collection_name);
```

### 6. Checkpoint Strategy

Manual checkpoint control for write-heavy periods:

```python
# Periodic checkpoints during heavy writes
with pool.get_connection() as conn:
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
```

**Strategy**:
- `PASSIVE`: Non-blocking, use during normal operation
- `FULL`: Blocking, use during maintenance windows
- `TRUNCATE`: Removes WAL, use for database compaction

## Monitoring and Profiling

### Key Metrics to Monitor

1. **Queue Depth**: `SELECT COUNT(*) FROM ingestion_queue`
2. **WAL Size**: Check `<db>-wal` file size on disk
3. **Oldest Item**: `SELECT MIN(queued_timestamp) FROM ingestion_queue`
4. **Error Rate**: `SELECT COUNT(*) FROM messages WHERE created_timestamp > datetime('now', '-1 hour')`

### Performance Testing

Run performance test suite regularly:

```bash
# Full test suite
python -m workspace_qdrant_mcp.core.queue_performance_test

# Specific benchmark
python -m workspace_qdrant_mcp.core.queue_performance_test --benchmark throughput
```

### Query Performance Analysis

Check slow queries with EXPLAIN QUERY PLAN:

```python
import sqlite3
conn = sqlite3.connect("workspace_state.db")
cursor = conn.execute("""
    EXPLAIN QUERY PLAN
    SELECT * FROM ingestion_queue
    ORDER BY priority DESC, queued_timestamp ASC
    LIMIT 10
""")
print(cursor.fetchall())
```

Look for:
- "USING INDEX" - Good, index is used
- "SCAN TABLE" - Bad, full table scan
- Add indexes if seeing table scans on large tables

## Troubleshooting

### Problem: Database is Locked Errors

**Symptoms**: `sqlite3.OperationalError: database is locked`

**Solutions**:
1. Increase `busy_timeout` (30-60 seconds)
2. Verify WAL mode enabled: `PRAGMA journal_mode` should return "wal"
3. Check connection pool limits aren't exceeded
4. Reduce concurrent writers if possible

### Problem: Slow Dequeue Operations

**Symptoms**: Dequeue latency > 100ms

**Solutions**:
1. Verify priority index exists: `SELECT * FROM sqlite_master WHERE type='index'`
2. Run `ANALYZE` to update statistics: `ANALYZE ingestion_queue`
3. Check queue depth: reduce if > 100K items
4. Consider archiving old error messages

### Problem: High Memory Usage

**Symptoms**: Memory usage > 512MB

**Solutions**:
1. Reduce `cache_size` (5000-8000 pages)
2. Reduce `mmap_size` (128MB-256MB)
3. Reduce connection pool `max_connections` (5-10)
4. Run periodic `PRAGMA optimize`

### Problem: WAL File Growing Too Large

**Symptoms**: `<db>-wal` file > 50MB

**Solutions**:
1. Reduce `wal_autocheckpoint` (500-1000 pages)
2. Increase checkpoint frequency (60-120 seconds)
3. Run manual checkpoint: `PRAGMA wal_checkpoint(FULL)`
4. Check for long-running readers blocking checkpoints

### Problem: Poor Throughput < 10 ops/sec

**Symptoms**: Can't achieve target throughput

**Solutions**:
1. Use batch operations instead of individual
2. Verify indexes present with EXPLAIN QUERY PLAN
3. Check disk I/O isn't bottleneck (SSD recommended)
4. Increase connection pool size
5. Profile with performance test suite to identify bottleneck

## Production Deployment Checklist

- [ ] WAL mode enabled and verified
- [ ] Connection pool configured (2-10 connections)
- [ ] Indexes created (automatic with schema)
- [ ] Checkpoint interval configured (300s default)
- [ ] Queue depth monitoring in place
- [ ] Error rate monitoring in place
- [ ] Performance test suite run and passing
- [ ] Backup strategy for SQLite database file

## Performance Regression Testing

Include in CI/CD pipeline:

```bash
# Run performance tests
python -m workspace_qdrant_mcp.core.queue_performance_test

# Check exit code (0 = pass, 1 = fail)
if [ $? -ne 0 ]; then
    echo "Performance regression detected!"
    exit 1
fi
```

Target metrics should remain stable across releases:
- Throughput within 10% of baseline
- Latencies within 20% of baseline
- Memory usage within 15% of baseline
