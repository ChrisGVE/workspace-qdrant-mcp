# Collection Migration Guide

This guide explains how to migrate existing suffix-based collections to the new multi-tenant architecture using metadata-based project isolation.

## Overview

The migration system transforms collections from suffix-based naming (e.g., `project-docs`, `project-code`) to multi-tenant architecture where project isolation is handled through metadata rather than collection names. This approach provides:

- **Better Resource Utilization**: Fewer collections mean better memory and performance efficiency
- **Unified Search**: Search across all project data in a single collection
- **Simplified Management**: Easier backup, monitoring, and maintenance
- **Scalable Architecture**: Support for many projects without collection proliferation

## Migration Process

The migration follows these phases:

1. **Analysis**: Examine existing collections and identify patterns
2. **Planning**: Create migration strategy with conflict detection
3. **Backup**: Create safety backups for rollback capability
4. **Migration**: Transform data with metadata injection
5. **Validation**: Verify data integrity and completeness
6. **Cleanup**: Remove temporary artifacts and finalize

## Prerequisites

Before starting migration:

- Ensure Qdrant server is accessible and stable
- Verify sufficient disk space for backups (approximately 2x current data size)
- Stop or pause any active ingestion processes
- Create a full database backup using your preferred method
- Ensure network connectivity is stable for the migration duration

## Quick Start

### 1. Analyze Existing Collections

First, analyze your current collection structure:

```bash
# Analyze all collections
wqm migration analyze --output analysis.json

# Filter by pattern and minimum size
wqm migration analyze --pattern-filter suffix_based --min-points 100 --output analysis.json
```

This will show:
- Collection naming patterns detected
- Project associations
- Data volumes and estimated migration time
- Potential conflicts or issues

### 2. Create Migration Plan

Generate a migration plan based on the analysis:

```bash
# Create plan from analysis
wqm migration plan --analysis analysis.json --output migration_plan.json

# Customize batch settings
wqm migration plan --analysis analysis.json --batch-size 2000 --parallel-batches 4 --output plan.json
```

The plan includes:
- Migration order and dependencies
- Batch configuration for optimal performance
- Conflict detection and resolution strategies
- Time and storage estimates

### 3. Execute Migration

Run the migration with progress tracking:

```bash
# Dry run to preview changes
wqm migration execute --plan migration_plan.json --dry-run

# Execute with confirmation
wqm migration execute --plan migration_plan.json --confirm

# Execute with custom directories
wqm migration execute --plan migration_plan.json --backup-dir ./backups --report-dir ./reports
```

### 4. Monitor Progress

Check migration status:

```bash
# View recent migration status
wqm migration status

# Check specific migration
wqm migration status --execution-id abc123-def456

# List available backups
wqm migration list-backups
```

## Advanced Usage

### Programmatic API

For custom migration workflows, use the Python API:

```python
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.memory.migration_utils import CollectionMigrationManager

# Initialize client and manager
config = Config()
client = QdrantWorkspaceClient(config)
await client.initialize()

manager = CollectionMigrationManager(client, config)

# Analyze collections
collections = await manager.analyze_collections()

# Create migration plan
plan = await manager.create_migration_plan(collections)

# Customize plan
plan.batch_size = 5000
plan.parallel_batches = 6
plan.create_backups = True

# Execute migration
result = await manager.execute_migration(plan)

# Generate report
report_file = await manager.generate_migration_report(plan, result)
```

### Custom Filtering

Filter collections for selective migration:

```python
# Filter by pattern
suffix_collections = [
    col for col in collections 
    if col.pattern == CollectionPattern.SUFFIX_BASED
]

# Filter by project
project_collections = [
    col for col in collections 
    if col.project_name == "my-important-project"
]

# Filter by size
large_collections = [
    col for col in collections 
    if col.point_count > 10000
]

# Create plan for filtered collections
plan = await manager.create_migration_plan(large_collections)
```

### Batch Configuration

Optimize batch settings for your environment:

```python
# High-performance settings (more memory, faster network)
plan.batch_size = 10000
plan.parallel_batches = 8

# Conservative settings (limited resources)
plan.batch_size = 500
plan.parallel_batches = 2

# Balanced settings (default)
plan.batch_size = 2000
plan.parallel_batches = 4
```

## Migration Strategies

### Strategy 1: Progressive Migration

Migrate collections progressively by priority:

1. Start with small, high-priority collections
2. Validate results before proceeding
3. Gradually migrate larger collections
4. Keep production systems running on original collections until validation complete

### Strategy 2: Project-by-Project

Migrate entire projects at once:

1. Group collections by project
2. Migrate one project completely
3. Switch application to use new collections
4. Proceed to next project

### Strategy 3: Collection Type Migration

Migrate by collection type across all projects:

1. Migrate all `docs` collections first
2. Then migrate `code` collections
3. Finally migrate other types
4. Allows testing with specific data types

## Rollback Procedures

If migration fails or issues are discovered:

### Automatic Rollback

The migration system automatically attempts rollback on failure:

```bash
# Check if automatic rollback occurred
wqm migration status --execution-id <failed-execution-id>
```

### Manual Rollback

Restore from specific backup:

```bash
# List available backups
wqm migration list-backups

# Restore specific collection
wqm migration rollback --backup-file ./backups/collection_backup.json --confirm
```

### Programmatic Rollback

```python
# Restore from backup file
success = await manager.rollback_manager.restore_backup("backup_file.json")

# Restore multiple collections
for backup_file in backup_files:
    await manager.rollback_manager.restore_backup(backup_file)
```

## Validation

### Post-Migration Validation

After migration, validate data integrity:

```python
# Compare point counts
source_info = client.get_collection("original-collection")
target_info = client.get_collection("target-collection")

assert source_info.points_count == target_info.points_count

# Validate sample data
source_points = client.scroll("original-collection", limit=100)
target_points = client.scroll("target-collection", limit=100)

# Check metadata injection
for point in target_points[0]:
    assert 'project_id' in point.payload
    assert 'migrated_at' in point.payload
```

### Search Validation

Verify search functionality works correctly:

```python
# Test search on migrated collection
results = await client.search(
    collection_name="target-collection",
    query_vector=test_vector,
    filter={"project_id": "my-project"}
)

assert len(results) > 0
```

## Performance Tuning

### Batch Size Optimization

Adjust batch size based on:

- **Network latency**: Larger batches for high-latency connections
- **Memory availability**: Smaller batches for memory-constrained environments
- **Vector dimensions**: Smaller batches for high-dimensional vectors
- **Payload size**: Smaller batches for large payloads

```python
# Calculate optimal batch size
avg_vector_size = 384 * 4  # 384 dimensions * 4 bytes per float
avg_payload_size = 1024    # Estimated payload size in bytes
available_memory_mb = 1024 # Available memory in MB

optimal_batch = (available_memory_mb * 1024 * 1024) // (avg_vector_size + avg_payload_size)
plan.batch_size = min(optimal_batch, 10000)  # Cap at 10k
```

### Parallel Processing

Configure parallel processing:

```python
# Based on CPU cores
import os
plan.parallel_batches = min(os.cpu_count(), 6)

# Based on network bandwidth
# Higher bandwidth = more parallel batches
# Lower bandwidth = fewer parallel batches
```

### Memory Management

Monitor memory usage during migration:

```python
import psutil

# Check memory before migration
memory_before = psutil.virtual_memory()
if memory_before.percent > 80:
    plan.batch_size = plan.batch_size // 2
    plan.parallel_batches = max(1, plan.parallel_batches // 2)
```

## Troubleshooting

### Common Issues

#### 1. Connection Timeouts

**Symptoms**: Migration fails with connection timeout errors

**Solutions**:
- Reduce batch size: `plan.batch_size = 1000`
- Reduce parallel batches: `plan.parallel_batches = 2`
- Increase Qdrant timeout settings
- Check network stability

#### 2. Memory Issues

**Symptoms**: Out of memory errors during migration

**Solutions**:
- Reduce batch size significantly: `plan.batch_size = 500`
- Reduce parallel processing: `plan.parallel_batches = 1`
- Increase available memory
- Process collections sequentially

#### 3. Disk Space Issues

**Symptoms**: Backup creation fails due to insufficient disk space

**Solutions**:
- Disable backups: `plan.create_backups = False` (not recommended)
- Use external backup storage
- Clean up old backups: `wqm migration cleanup --days 7`
- Migrate smaller subsets at a time

#### 4. Collection Conflicts

**Symptoms**: Target collection already exists with different schema

**Solutions**:
- Use different target collection names
- Manually resolve schema conflicts
- Delete conflicting collections (with caution)

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger('workspace_qdrant_mcp.memory.migration_utils').setLevel(logging.DEBUG)

# Run migration with debug logging
result = await manager.execute_migration(plan)
```

### Error Recovery

If migration is interrupted:

1. Check migration status to see completed collections
2. Modify plan to exclude already-migrated collections
3. Resume migration with updated plan
4. Use backups if data corruption is detected

## Best Practices

### Before Migration

1. **Test on Staging**: Always test migration on a staging environment first
2. **Backup Everything**: Create full database backups using multiple methods
3. **Document State**: Record current collection structure and metadata
4. **Plan Downtime**: Schedule migration during low-usage periods
5. **Notify Users**: Inform users of potential service interruptions

### During Migration

1. **Monitor Progress**: Regularly check migration status and logs
2. **Watch Resources**: Monitor CPU, memory, and disk usage
3. **Network Stability**: Ensure stable network connection
4. **Parallel Migrations**: Avoid running multiple migrations simultaneously
5. **Emergency Stop**: Know how to safely stop migration if needed

### After Migration

1. **Validate Thoroughly**: Test all critical functionality
2. **Performance Check**: Verify search performance meets expectations
3. **Update Applications**: Modify applications to use new collection structure
4. **Monitor Metrics**: Watch performance metrics for any degradation
5. **Document Changes**: Update documentation and runbooks

### Production Considerations

1. **Gradual Rollout**: Migrate collections in phases
2. **Feature Flags**: Use feature flags to switch between old and new collections
3. **Monitoring**: Set up alerts for migration failures or data inconsistencies
4. **Rollback Plan**: Have a detailed rollback procedure ready
5. **Team Coordination**: Ensure all team members understand the migration process

## Maintenance

### Regular Cleanup

Clean up old migration artifacts:

```bash
# Remove backups older than 30 days
wqm migration cleanup --days 30 --confirm

# Custom cleanup with specific directories
wqm migration cleanup --backup-dir ./old_backups --report-dir ./old_reports --days 14
```

### Backup Management

Manage migration backups:

```bash
# List all backups with details
wqm migration list-backups --format json

# Archive old backups to external storage
# (custom script based on your backup strategy)
```

### Performance Monitoring

Monitor post-migration performance:

```python
# Check collection statistics
stats = await client.collection_manager.get_collection_stats()

# Monitor search performance
search_times = []
for _ in range(100):
    start = time.time()
    results = await client.search(collection_name, query_vector)
    search_times.append(time.time() - start)

avg_search_time = sum(search_times) / len(search_times)
```

## Migration Report Analysis

Understanding migration reports:

### Success Metrics

- **Collections Migrated**: Number of collections successfully migrated
- **Points Migrated**: Total number of points transferred
- **Success Rate**: Percentage of successful point migrations
- **Duration**: Total migration time

### Performance Metrics

- **Points per Second**: Migration throughput
- **Batch Success Rate**: Percentage of successful batches
- **Memory Usage**: Peak memory consumption during migration
- **Network Utilization**: Data transfer rates

### Issue Analysis

Review errors and warnings in reports:

- **Connection Errors**: Network or Qdrant connectivity issues
- **Memory Errors**: Resource constraint problems
- **Data Errors**: Inconsistent or corrupted data
- **Validation Errors**: Post-migration data integrity issues

## Migration Checklist

### Pre-Migration

- [ ] Full database backup created
- [ ] Staging environment tested
- [ ] Migration plan reviewed and approved
- [ ] Team notified of migration schedule
- [ ] Monitoring systems configured
- [ ] Rollback procedures documented
- [ ] Sufficient disk space verified
- [ ] Network stability confirmed

### During Migration

- [ ] Migration started with proper parameters
- [ ] Progress monitored regularly
- [ ] Resource usage within acceptable limits
- [ ] No critical errors in logs
- [ ] Backup creation successful
- [ ] Network connection stable

### Post-Migration

- [ ] Data integrity validated
- [ ] Search functionality verified
- [ ] Performance metrics acceptable
- [ ] Applications updated to use new structure
- [ ] Documentation updated
- [ ] Migration report generated and reviewed
- [ ] Cleanup tasks scheduled
- [ ] Team informed of completion

## Support and Resources

### Getting Help

- **Documentation**: This guide and API documentation
- **Logs**: Check migration logs for detailed error information
- **Community**: Project GitHub issues and discussions
- **Professional Support**: Contact maintainers for enterprise support

### Additional Resources

- [Multi-Tenant Architecture Guide](./multitenancy_architecture.md)
- [Backup and Recovery Guide](./BACKUP_RESTORE.md)
- [API Reference (gRPC)](./GRPC_API.md)

### Contributing

Report issues or contribute improvements:

1. Submit bug reports with migration logs
2. Suggest performance optimizations
3. Share successful migration strategies
4. Contribute to documentation improvements