# Qdrant Point Operations Cheatsheet

Reference for what Qdrant supports at the point level. Based on `qdrant-client` Rust crate docs.

## Single-Point Operations

| Operation | Method | What It Does |
|-----------|--------|-------------|
| **Upsert** | `upsert_points` | Insert or replace point (ID + vectors + payload). If ID exists, full overwrite. |
| **Delete points** | `delete_points` | Remove points by ID list or by filter. |
| **Set payload** | `set_payload` | Merge new fields into existing payload. Existing fields not in the update are preserved. |
| **Overwrite payload** | `overwrite_payload` | Replace entire payload. Fields not in the update are removed. |
| **Delete payload** | `delete_payload` | Remove specific payload keys. Other keys preserved. |
| **Clear payload** | `clear_payload` | Remove all payload fields. Vectors untouched. |
| **Update vectors** | `update_vectors` | Update specific named vectors on a point. Other named vectors on the same point are untouched. |
| **Delete vectors** | `delete_vectors` | Remove specific named vectors from a point. Point and other vectors remain. |

## Point ID

- **Immutable.** Cannot be changed after creation.
- To "change" an ID: delete old point + create new point with new ID.
- Types: unsigned integer (`u64`) or UUID string.
- ID uniqueness is per-collection.

## Payload Updates (No Vector Change)

```rust
// SetPayload: merge fields (existing fields preserved)
client.set_payload(
    SetPayloadPointsBuilder::new("collection", payload)
        .points_selector(vec![point_id])
        .wait(true)
).await?;

// OverwritePayload: replace entire payload
client.overwrite_payload(
    OverwritePayloadBuilder::new("collection", payload)
        .points_selector(vec![point_id])
        .wait(true)
).await?;

// DeletePayload: remove specific keys
client.delete_payload(
    DeletePayloadPointsBuilder::new("collection", vec!["key_to_remove"])
        .points_selector(vec![point_id])
        .wait(true)
).await?;
```

## Vector Updates (No Payload Change)

Named vectors can be updated independently. Updating "dense" does not touch "sparse" and vice versa.

```rust
// Update specific named vector(s) on existing points
client.update_vectors(
    UpdatePointVectorsBuilder::new(
        "collection",
        vec![PointVectors {
            id: Some(point_id.into()),
            vectors: Some(HashMap::from([
                ("dense".to_string(), vec![0.1, 0.2, 0.3, 0.4])
            ]).into()),
        }],
    ).wait(true)
).await?;

// Delete a named vector (point and other vectors remain)
client.delete_vectors(
    DeletePointVectorsBuilder::new("collection", vec!["sparse"])
        .points_selector(vec![point_id])
        .wait(true)
).await?;
```

## Batch Operations

Batch executes **multiple operations in a single request**. Operations run **sequentially** within the batch.

**Any combination** of operations is allowed:

```rust
pub enum Operation {
    Upsert(PointStructList),        // Insert/replace points (multiple per op)
    DeletePoints(DeletePoints),      // Delete by ID or filter (multiple per op)
    SetPayload(SetPayload),          // Merge payload fields (multiple per op)
    OverwritePayload(OverwritePayload), // Replace payload (multiple per op)
    DeletePayload(DeletePayload),    // Remove payload keys (multiple per op)
    ClearPayload(ClearPayload),      // Clear all payload (multiple per op)
    UpdateVectors(UpdateVectors),    // Update named vectors (multiple per op)
    DeleteVectors(DeleteVectors),    // Remove named vectors (multiple per op)
}
```

Each operation targets **multiple points**. Example: delete 10 points + upsert 50 new ones + set payload on 20 others, all in one call.

```rust
client.update_points_batch(
    UpdateBatchPointsBuilder::new("collection", vec![
        // Op 1: Delete old points
        PointsUpdateOperation {
            operation: Some(Operation::DeletePoints(DeletePoints {
                points: Some(PointsSelector { /* IDs or filter */ }),
            })),
        },
        // Op 2: Upsert new points
        PointsUpdateOperation {
            operation: Some(Operation::Upsert(PointStructList {
                points: vec![/* multiple PointStruct */],
                ..Default::default()
            })),
        },
        // Op 3: Update payload on other points
        PointsUpdateOperation {
            operation: Some(Operation::SetPayload(SetPayload {
                points_selector: Some(/* IDs or filter */),
                payload: HashMap::from([("key".into(), "value".into())]),
                ..Default::default()
            })),
        },
    ]).wait(true)
).await?;
```

## Point Selection

Operations that target existing points accept a `PointsSelector`:

- **By IDs**: List of specific point IDs
- **By filter**: Qdrant filter expression (match payload fields, ranges, geo, etc.)

This means you can do things like "set payload on all points where `tenant_id == X`" without knowing individual point IDs.

## Key Implications for Design

1. **Point ID is immutable** — choose the ID formula carefully; changing it requires delete + re-create.
2. **Payload is cheap to update** — adding metadata (instance tracking, commit hashes) doesn't require re-embedding.
3. **Named vectors are independently mutable** — can update dense without touching sparse, useful for BM25 vocabulary updates.
4. **Batch is fully flexible** — delete + upsert + payload update in one atomic-ish call. Operations within a batch are sequential.
5. **Filter-based operations** — can target points by payload field values, not just by ID. Enables bulk operations like "delete all points for tenant X".
