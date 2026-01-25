# Collection Naming Migration Guide

This guide documents the migration path from underscore-prefixed collection names
(`_projects`, `_libraries`, `_memory`) to canonical collection names (`projects`,
`libraries`, `memory`) per ADR-001.

## Overview

### What Changed

| Old Name (Deprecated) | New Name (Canonical) | Purpose |
|----------------------|---------------------|---------|
| `_projects` | `projects` | All project code and documents |
| `_libraries` | `libraries` | Library documentation |
| `_memory` | `memory` | Agent memory and rules |
| `_agent_memory` | `memory` | Merged into unified memory |

### Why This Changed

1. **Consistency**: Canonical names without underscore prefix are cleaner
2. **Multi-tenancy**: Collections now use `tenant_id` metadata for project isolation
3. **Simplification**: Single unified collection per type instead of per-project collections

### Impact Assessment

- **New deployments**: No migration needed - will use canonical names automatically
- **Existing deployments with data**: Migration required to move data
- **MCP server**: Already updated to use canonical names
- **Daemon**: Already updated to use canonical names

## Migration Strategies

Choose based on your risk tolerance and downtime requirements:

| Strategy | Risk Level | Downtime | Complexity | Best For |
|----------|------------|----------|------------|----------|
| Collection Aliasing | Low | None | Low | Quick fix, testing |
| Data Migration | Medium | Brief | Medium | Complete migration |
| Parallel Operation | Low | None | High | Large deployments |

---

## Strategy 1: Collection Aliasing (Recommended for Quick Fix)

Use Qdrant collection aliases to map old names to new names. This provides
instant compatibility without data movement.

### Prerequisites

- Qdrant 1.7+ (alias support)
- Admin access to Qdrant

### Steps

1. **Create canonical collections** (if they don't exist):

```bash
# Using curl
curl -X PUT "http://localhost:6333/collections/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'

curl -X PUT "http://localhost:6333/collections/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'

curl -X PUT "http://localhost:6333/collections/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'
```

2. **Create aliases pointing old names to new collections**:

```bash
# Alias _projects -> projects
curl -X POST "http://localhost:6333/collections/aliases" \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {
        "create_alias": {
          "collection_name": "projects",
          "alias_name": "_projects"
        }
      }
    ]
  }'

# Alias _libraries -> libraries
curl -X POST "http://localhost:6333/collections/aliases" \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {
        "create_alias": {
          "collection_name": "libraries",
          "alias_name": "_libraries"
        }
      }
    ]
  }'

# Alias _memory -> memory
curl -X POST "http://localhost:6333/collections/aliases" \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {
        "create_alias": {
          "collection_name": "memory",
          "alias_name": "_memory"
        }
      },
      {
        "create_alias": {
          "collection_name": "memory",
          "alias_name": "_agent_memory"
        }
      }
    ]
  }'
```

3. **Verify aliases**:

```bash
curl "http://localhost:6333/aliases"
```

### Rollback

Remove aliases if needed:

```bash
curl -X POST "http://localhost:6333/collections/aliases" \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {"delete_alias": {"alias_name": "_projects"}},
      {"delete_alias": {"alias_name": "_libraries"}},
      {"delete_alias": {"alias_name": "_memory"}},
      {"delete_alias": {"alias_name": "_agent_memory"}}
    ]
  }'
```

### Limitations

- Data remains in old collections (not truly migrated)
- Old collection names become unusable for actual collections
- Requires alias support in Qdrant

---

## Strategy 2: Data Migration (Recommended for Complete Migration)

Move all data from old collections to new collections. This provides
a clean separation and allows old collections to be deleted.

### Prerequisites

- Python 3.10+ with qdrant-client
- Sufficient disk space for temporary data
- Brief downtime window (minutes to hours depending on data size)

### Migration Script

Save as `migrate_collections.py`:

```python
#!/usr/bin/env python3
"""
Collection naming migration script.
Migrates data from deprecated underscore-prefixed collections to canonical names.
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScrollRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Migration mapping
MIGRATION_MAP = {
    "_projects": "projects",
    "_libraries": "libraries",
    "_memory": "memory",
    "_agent_memory": "memory",  # Merged into unified memory
}

# Default vector configuration
DEFAULT_VECTOR_SIZE = 384
DEFAULT_DISTANCE = Distance.COSINE


def get_collection_config(client: QdrantClient, collection_name: str) -> Optional[dict]:
    """Get collection configuration if it exists."""
    try:
        info = client.get_collection(collection_name)
        return {
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance,
        }
    except Exception:
        return None


def create_collection_if_not_exists(
    client: QdrantClient,
    name: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    distance: Distance = DEFAULT_DISTANCE,
) -> bool:
    """Create collection if it doesn't exist. Returns True if created."""
    try:
        client.get_collection(name)
        logger.info(f"Collection '{name}' already exists")
        return False
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        logger.info(f"Created collection '{name}'")
        return True


def migrate_collection(
    client: QdrantClient,
    source: str,
    target: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict:
    """Migrate all points from source to target collection."""
    stats = {
        "source": source,
        "target": target,
        "points_migrated": 0,
        "batches_processed": 0,
        "errors": [],
    }

    # Check source exists
    source_config = get_collection_config(client, source)
    if not source_config:
        logger.warning(f"Source collection '{source}' does not exist, skipping")
        return stats

    # Create target if needed
    if not dry_run:
        create_collection_if_not_exists(
            client, target,
            vector_size=source_config["vector_size"],
            distance=source_config["distance"],
        )

    # Scroll through all points in source
    offset = None
    while True:
        result = client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        points, next_offset = result

        if not points:
            break

        if dry_run:
            logger.info(f"[DRY RUN] Would migrate {len(points)} points")
            stats["points_migrated"] += len(points)
        else:
            # Convert to PointStruct for upsert
            point_structs = [
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in points
            ]

            try:
                client.upsert(
                    collection_name=target,
                    points=point_structs,
                )
                stats["points_migrated"] += len(points)
                logger.info(f"Migrated {len(points)} points to '{target}'")
            except Exception as e:
                error_msg = f"Error migrating batch: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        stats["batches_processed"] += 1
        offset = next_offset

        if next_offset is None:
            break

    return stats


def run_migration(
    qdrant_url: str = "http://localhost:6333",
    api_key: Optional[str] = None,
    batch_size: int = 100,
    dry_run: bool = False,
    delete_source: bool = False,
) -> dict:
    """Run the full migration."""
    client = QdrantClient(url=qdrant_url, api_key=api_key)

    results = {
        "started_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "migrations": [],
        "total_points": 0,
        "total_errors": 0,
    }

    for source, target in MIGRATION_MAP.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Migrating: {source} -> {target}")
        logger.info('='*50)

        stats = migrate_collection(
            client, source, target,
            batch_size=batch_size,
            dry_run=dry_run,
        )

        results["migrations"].append(stats)
        results["total_points"] += stats["points_migrated"]
        results["total_errors"] += len(stats["errors"])

        # Optionally delete source collection
        if delete_source and not dry_run and stats["points_migrated"] > 0:
            if not stats["errors"]:
                try:
                    client.delete_collection(source)
                    logger.info(f"Deleted source collection '{source}'")
                    stats["source_deleted"] = True
                except Exception as e:
                    logger.error(f"Failed to delete '{source}': {e}")
                    stats["source_deleted"] = False
            else:
                logger.warning(f"Skipping deletion of '{source}' due to errors")

    results["completed_at"] = datetime.now().isoformat()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Migrate collections from deprecated to canonical names"
    )
    parser.add_argument(
        "--url", default="http://localhost:6333",
        help="Qdrant server URL"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Qdrant API key (if required)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of points to migrate per batch"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--delete-source", action="store_true",
        help="Delete source collections after successful migration"
    )

    args = parser.parse_args()

    logger.info("Starting collection naming migration")
    logger.info(f"Qdrant URL: {args.url}")
    logger.info(f"Dry run: {args.dry_run}")

    results = run_migration(
        qdrant_url=args.url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        delete_source=args.delete_source,
    )

    logger.info("\n" + "="*50)
    logger.info("MIGRATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total points migrated: {results['total_points']}")
    logger.info(f"Total errors: {results['total_errors']}")

    if results["total_errors"] > 0:
        logger.error("Migration completed with errors!")
        sys.exit(1)
    else:
        logger.info("Migration completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

### Usage

```bash
# Install dependencies
pip install qdrant-client

# Dry run first (recommended)
python migrate_collections.py --dry-run

# Actual migration
python migrate_collections.py

# Migration with source deletion
python migrate_collections.py --delete-source

# Custom Qdrant URL
python migrate_collections.py --url http://qdrant-server:6333 --api-key YOUR_KEY
```

### Rollback

If migration fails partway through:

1. **Stop the migration** (Ctrl+C)
2. **Check target collection** for migrated points
3. **Option A**: Resume migration (script is idempotent)
4. **Option B**: Delete target collection and start fresh:

```bash
curl -X DELETE "http://localhost:6333/collections/projects"
curl -X DELETE "http://localhost:6333/collections/libraries"
curl -X DELETE "http://localhost:6333/collections/memory"
```

### Estimated Migration Times

| Document Count | Batch Size | Estimated Time |
|----------------|------------|----------------|
| 10,000 | 100 | ~2 minutes |
| 100,000 | 100 | ~15 minutes |
| 1,000,000 | 100 | ~2 hours |

*Times vary based on document size, network latency, and Qdrant server performance.*

---

## Strategy 3: Parallel Operation (Zero-Downtime Migration)

Run both old and new collection names simultaneously, gradually migrating
traffic and data.

### Overview

1. Create canonical collections
2. Configure MCP server to write to both old and new
3. Background migrate historical data
4. Switch reads to canonical collections
5. Stop writes to old collections
6. Delete old collections

### Implementation Steps

#### Phase 1: Setup (No Downtime)

```bash
# Create canonical collections
curl -X PUT "http://localhost:6333/collections/projects" ...
curl -X PUT "http://localhost:6333/collections/libraries" ...
curl -X PUT "http://localhost:6333/collections/memory" ...
```

#### Phase 2: Dual-Write Mode

Configure MCP server environment:

```bash
export WQM_DUAL_WRITE=true
export WQM_WRITE_COLLECTIONS="_projects,projects"
```

This requires custom MCP server modification (not currently implemented).

#### Phase 3: Background Migration

Run migration script in background while service continues:

```bash
nohup python migrate_collections.py --batch-size 50 > migration.log 2>&1 &
```

#### Phase 4: Switch Reads

After migration completes and data is verified:

```bash
export WQM_READ_COLLECTIONS="projects,libraries,memory"
```

#### Phase 5: Stop Dual Writes

```bash
export WQM_DUAL_WRITE=false
export WQM_WRITE_COLLECTIONS="projects,libraries,memory"
```

#### Phase 6: Cleanup

```bash
# After confirming no issues (wait 24-48 hours)
curl -X DELETE "http://localhost:6333/collections/_projects"
curl -X DELETE "http://localhost:6333/collections/_libraries"
curl -X DELETE "http://localhost:6333/collections/_memory"
```

---

## Verification Checklist

After migration, verify:

- [ ] Canonical collections exist and are populated
- [ ] MCP server can read from canonical collections
- [ ] MCP server can write to canonical collections
- [ ] Search results return expected documents
- [ ] No errors in MCP server logs
- [ ] No errors in daemon logs
- [ ] Old collections deleted (if applicable)

### Verification Commands

```bash
# Check collections exist
curl "http://localhost:6333/collections"

# Check collection stats
curl "http://localhost:6333/collections/projects"
curl "http://localhost:6333/collections/libraries"
curl "http://localhost:6333/collections/memory"

# Test search via MCP
wqm search "test query" --collection projects

# Check for deprecated name warnings in logs
grep -i "deprecated" ~/.workspace-qdrant-mcp/logs/*.log
```

---

## Troubleshooting

### "Collection not found" Errors

**Cause**: Canonical collections don't exist yet.

**Solution**: Create collections or use aliasing strategy.

### "Already exists" Errors During Migration

**Cause**: Target collection already has data.

**Solution**: Migration script uses upsert (safe to re-run).

### Performance Degradation During Migration

**Cause**: Migration consuming Qdrant resources.

**Solution**: Reduce batch size, run during low-traffic periods.

### Points Missing After Migration

**Cause**: Network errors during migration.

**Solution**: Re-run migration (idempotent) or compare counts.

---

## FAQ

**Q: Do I need to migrate if I'm starting fresh?**

A: No. New deployments automatically use canonical names.

**Q: What happens to existing integrations using old names?**

A: They will receive deprecation warnings. Use aliasing for quick compatibility.

**Q: Can I run old and new MCP server versions together?**

A: Yes, if using aliasing or dual-write strategy.

**Q: How do I know if I have data in old collections?**

A: Check Qdrant dashboard or run:
```bash
curl "http://localhost:6333/collections/_projects" 2>/dev/null | jq '.result.points_count'
```

---

## Support

For migration assistance:

1. Check [GitHub Issues](https://github.com/your-org/workspace-qdrant-mcp/issues)
2. Review ADR-001 for naming rationale
3. Contact the development team

---

*Last updated: January 2026*
