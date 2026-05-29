//! SQL execution for QueueWriteService commands.

use sqlx::SqlitePool;
use uuid::Uuid;
use wqm_common::hashing::generate_idempotency_key;
use wqm_common::queue_types::{ItemType, QueueOperation};
use wqm_common::timestamps;

use super::actor::WriteActor;
use super::commands::*;

impl WriteActor {
    pub(super) async fn exec_enqueue_item(
        &self,
        data: EnqueueItemData,
    ) -> WriteResult<EnqueueItemResult> {
        let item_type = ItemType::parse_str(&data.item_type)
            .ok_or_else(|| format!("invalid item_type: {}", data.item_type))?;
        let op = QueueOperation::parse_str(&data.op)
            .ok_or_else(|| format!("invalid op: {}", data.op))?;

        if data.tenant_id.is_empty() {
            return Err("tenant_id cannot be empty".into());
        }
        if data.collection.is_empty() {
            return Err("collection cannot be empty".into());
        }

        let idempotency_key = generate_idempotency_key(
            item_type,
            op,
            &data.tenant_id,
            &data.collection,
            &data.payload_json,
        )
        .map_err(|e| format!("idempotency key error: {}", e))?;

        let queue_id = Uuid::new_v4().to_string();
        let now = timestamps::now_utc();
        let branch = if data.branch.is_empty() {
            "main"
        } else {
            &data.branch
        };
        let metadata = data.metadata_json.as_deref().unwrap_or("{}");
        // Populate file_path for file items so the v36 composite partial
        // UNIQUE index can dedupe equivalent enqueues from the gRPC write
        // path (F-009).
        let file_path = extract_file_path(item_type, &data.payload_json);

        let result = sqlx::query(
            r#"INSERT OR IGNORE INTO unified_queue (
                queue_id, idempotency_key, item_type, op, tenant_id, collection,
                status, branch, payload_json, metadata, created_at, updated_at, retry_count,
                file_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7, ?8, ?9, ?10, ?10, 0, ?11)"#,
        )
        .bind(&queue_id)
        .bind(&idempotency_key)
        .bind(item_type.as_str())
        .bind(op.as_str())
        .bind(&data.tenant_id)
        .bind(&data.collection)
        .bind(branch)
        .bind(&data.payload_json)
        .bind(metadata)
        .bind(&now)
        .bind(&file_path)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        let is_new = result.rows_affected() > 0;
        let response_queue_id = if is_new {
            queue_id
        } else {
            sqlx::query_scalar::<_, String>(
                "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1",
            )
            .bind(&idempotency_key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?
            .unwrap_or_default()
        };

        Ok(EnqueueItemResult {
            queue_id: response_queue_id,
            idempotency_key,
            is_new,
        })
    }

    pub(super) async fn exec_retry_all(&self) -> WriteResult<RetryAllResult> {
        let now = timestamps::now_utc();
        let result = sqlx::query(
            r#"UPDATE unified_queue
            SET status = 'pending', retry_count = 0, error_message = NULL,
                last_error_at = NULL, lease_until = NULL, worker_id = NULL,
                qdrant_status = NULL, search_status = NULL, updated_at = ?1
            WHERE status = 'failed'"#,
        )
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(RetryAllResult {
            reset_count: result.rows_affected() as u32,
        })
    }

    pub(super) async fn exec_retry_item(
        &self,
        data: RetryItemData,
    ) -> WriteResult<RetryItemResult> {
        let prefix = format!("{}%", data.queue_id);
        let row = sqlx::query_as::<_, (String, String, i32)>(
            "SELECT queue_id, status, retry_count FROM unified_queue \
             WHERE queue_id = ?1 OR queue_id LIKE ?2 LIMIT 1",
        )
        .bind(&data.queue_id)
        .bind(&prefix)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        let Some((found_id, status, retry_count)) = row else {
            return Ok(RetryItemResult {
                found: false,
                resolved_id: String::new(),
                previous_status: String::new(),
                previous_retry_count: 0,
                reset: false,
            });
        };

        if status != "failed" {
            return Ok(RetryItemResult {
                found: true,
                resolved_id: found_id,
                previous_status: status,
                previous_retry_count: retry_count,
                reset: false,
            });
        }

        let now = timestamps::now_utc();
        sqlx::query(
            r#"UPDATE unified_queue
            SET status = 'pending', retry_count = 0, error_message = NULL,
                last_error_at = NULL, lease_until = NULL, worker_id = NULL,
                qdrant_status = NULL, search_status = NULL, updated_at = ?1
            WHERE queue_id = ?2"#,
        )
        .bind(&now)
        .bind(&found_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(RetryItemResult {
            found: true,
            resolved_id: found_id,
            previous_status: status,
            previous_retry_count: retry_count,
            reset: true,
        })
    }

    pub(super) async fn exec_clean_queue(&self, data: CleanQueueData) -> WriteResult<u32> {
        let statuses: Vec<&str> = if data.statuses.is_empty() {
            vec!["done", "failed"]
        } else {
            data.statuses.iter().map(|s| s.as_str()).collect()
        };

        for s in &statuses {
            if !["done", "failed"].contains(s) {
                return Err(format!(
                    "invalid status '{}': only 'done' and 'failed' are cleanable",
                    s
                ));
            }
        }

        if data.older_than_days <= 0 {
            return Err("older_than_days must be positive".into());
        }

        let placeholders: String = statuses
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            "DELETE FROM unified_queue \
             WHERE status IN ({}) AND updated_at < datetime('now', '-' || ?1 || ' days')",
            placeholders
        );

        let mut query = sqlx::query(&sql).bind(data.older_than_days);
        for s in &statuses {
            query = query.bind(*s);
        }

        let result = query
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

        Ok(result.rows_affected() as u32)
    }

    pub(super) async fn exec_cancel_items(
        &self,
        data: CancelItemsData,
    ) -> WriteResult<CancelItemsResult> {
        if data.tenant_id.is_empty() {
            return Err("tenant_id cannot be empty".into());
        }

        let statuses: Vec<&str> = if data.statuses.is_empty() {
            vec!["pending", "failed"]
        } else {
            data.statuses
                .iter()
                .map(|s| s.as_str())
                .filter(|s| *s != "in_progress")
                .collect()
        };

        if statuses.is_empty() {
            return Err(
                "no cancellable statuses specified (in_progress cannot be cancelled)".into(),
            );
        }

        let (resolved_tenant, project_path) = resolve_tenant(&self.pool, &data.tenant_id)
            .await
            .map_err(|e| format!("tenant resolution failed: {}", e))?;

        let placeholders: String = statuses
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");

        if data.dry_run {
            let count_sql = format!(
                "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = ?1 AND status IN ({})",
                placeholders
            );
            let mut query = sqlx::query_scalar::<_, i64>(&count_sql).bind(&resolved_tenant);
            for s in &statuses {
                query = query.bind(*s);
            }
            let count = query
                .fetch_one(&self.pool)
                .await
                .map_err(|e| format!("database error: {}", e))?;

            return Ok(CancelItemsResult {
                count: count as u32,
                tenant_id: resolved_tenant,
                project_path,
                is_dry_run: true,
            });
        }

        let delete_sql = format!(
            "DELETE FROM unified_queue WHERE tenant_id = ?1 AND status IN ({})",
            placeholders
        );
        let mut query = sqlx::query(&delete_sql).bind(&resolved_tenant);
        for s in &statuses {
            query = query.bind(*s);
        }
        let result = query
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

        Ok(CancelItemsResult {
            count: result.rows_affected() as u32,
            tenant_id: resolved_tenant,
            project_path,
            is_dry_run: false,
        })
    }

    pub(super) async fn exec_remove_item(
        &self,
        data: RemoveItemData,
    ) -> WriteResult<RemoveItemResult> {
        let prefix = format!("{}%", data.queue_id);
        let row = sqlx::query_as::<_, (String, String, String, String, String)>(
            "SELECT queue_id, item_type, op, collection, status \
             FROM unified_queue \
             WHERE queue_id = ?1 OR queue_id LIKE ?2 OR idempotency_key LIKE ?2 \
             LIMIT 1",
        )
        .bind(&data.queue_id)
        .bind(&prefix)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        let Some((found_id, item_type, op, collection, status)) = row else {
            return Ok(RemoveItemResult {
                found: false,
                resolved_id: String::new(),
                item_type: String::new(),
                op: String::new(),
                collection: String::new(),
                status: String::new(),
            });
        };

        if status == "in_progress" {
            tracing::warn!(
                queue_id = %found_id,
                "removing in_progress item — processor may error"
            );
        }

        sqlx::query("DELETE FROM unified_queue WHERE queue_id = ?1")
            .bind(&found_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

        Ok(RemoveItemResult {
            found: true,
            resolved_id: found_id,
            item_type,
            op,
            collection,
            status,
        })
    }

    pub(super) async fn exec_clean_queue_by_collection(
        &self,
        data: CleanQueueByCollectionData,
    ) -> WriteResult<u32> {
        if data.collections.is_empty() {
            return Err("at least one collection name is required".into());
        }

        let statuses: Vec<&str> = if data.statuses.is_empty() {
            vec!["pending", "failed"]
        } else {
            data.statuses
                .iter()
                .map(|s| s.as_str())
                .filter(|s| *s != "in_progress")
                .collect()
        };

        if statuses.is_empty() {
            return Err("no deletable statuses specified (in_progress cannot be deleted)".into());
        }

        let col_placeholders: String = data
            .collections
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");

        let status_placeholders: String = statuses
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1 + data.collections.len()))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            "DELETE FROM unified_queue WHERE collection IN ({}) AND status IN ({})",
            col_placeholders, status_placeholders
        );

        let mut query = sqlx::query(&sql);
        for c in &data.collections {
            query = query.bind(c);
        }
        for s in &statuses {
            query = query.bind(*s);
        }

        let result = query
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

        Ok(result.rows_affected() as u32)
    }
}

/// Extract `file_path` from a JSON payload, but only for file items.
///
/// Used by the gRPC write path to populate the dedup column so the
/// composite partial UNIQUE index on
/// `(tenant_id, branch, collection, item_type, op, file_path)` enforces
/// per-file uniqueness (F-009). Non-file items return `None`, which
/// excludes them from the partial index.
fn extract_file_path(item_type: ItemType, payload_json: &str) -> Option<String> {
    if item_type != ItemType::File {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(payload_json).ok()?;
    value
        .get("file_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Resolve a tenant hint to (tenant_id, project_path).
async fn resolve_tenant(pool: &SqlitePool, hint: &str) -> Result<(String, String), String> {
    let row = sqlx::query_as::<_, (String, String)>(
        "SELECT tenant_id, path FROM watch_folders WHERE tenant_id = ?1 LIMIT 1",
    )
    .bind(hint)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    if let Some((tid, path)) = row {
        return Ok((tid, path));
    }

    // Spec §16 §3.1: lookup uses syntactic-canonical form (no fs symlink follow)
    let canonical = wqm_common::paths::CanonicalPath::from_user_input(hint)
        .map(|cp| cp.into_string())
        .unwrap_or_else(|_| hint.to_string());

    let row = sqlx::query_as::<_, (String, String)>(
        "SELECT tenant_id, path FROM watch_folders WHERE path = ?1 LIMIT 1",
    )
    .bind(&canonical)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    if let Some((tid, path)) = row {
        return Ok((tid, path));
    }

    let pattern = format!("%{}%", hint);
    let rows = sqlx::query_as::<_, (String, String)>(
        "SELECT tenant_id, path FROM watch_folders WHERE path LIKE ?1",
    )
    .bind(&pattern)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    match rows.len() {
        0 => Err(format!("no project found matching '{}'", hint)),
        1 => Ok(rows.into_iter().next().unwrap()),
        n => Err(format!("ambiguous: '{}' matches {} projects", hint, n)),
    }
}
