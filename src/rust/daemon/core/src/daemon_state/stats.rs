//! Database statistics for watch folders.

use serde_json::Value as JsonValue;
use sqlx::Row;
use std::collections::HashMap;
use wqm_common::constants::COLLECTION_PROJECTS;

use super::{DaemonStateManager, DaemonStateResult};

impl DaemonStateManager {
    /// Get database statistics (watch folders only)
    pub async fn get_stats(&self) -> DaemonStateResult<HashMap<String, JsonValue>> {
        let mut stats = HashMap::new();

        // Watch folder counts by enabled/disabled status
        let watch_rows = sqlx::query(
            "SELECT CASE WHEN enabled = 1 THEN 'enabled' ELSE 'disabled' END as status, COUNT(*) as count FROM watch_folders GROUP BY enabled"
        )
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();

        let mut watch_stats = HashMap::new();
        for row in watch_rows {
            let status: String = row.try_get("status").unwrap_or_default();
            let count: i64 = row.try_get("count").unwrap_or(0);
            watch_stats.insert(status, JsonValue::Number(count.into()));
        }
        stats.insert(
            "watch_counts".to_string(),
            JsonValue::Object(watch_stats.into_iter().collect()),
        );

        // Watch folder counts by collection type
        let collection_rows = sqlx::query(
            "SELECT collection, COUNT(*) as count FROM watch_folders GROUP BY collection",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let mut collection_stats = HashMap::new();
        for row in collection_rows {
            let collection: String = row.try_get("collection").unwrap_or_default();
            let count: i64 = row.try_get("count").unwrap_or(0);
            collection_stats.insert(collection, JsonValue::Number(count.into()));
        }
        stats.insert(
            "collection_counts".to_string(),
            JsonValue::Object(collection_stats.into_iter().collect()),
        );

        // Active projects count
        let active_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folders WHERE is_active = 1 AND collection = ?1",
        )
        .bind(COLLECTION_PROJECTS)
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);
        stats.insert(
            "active_projects".to_string(),
            JsonValue::Number(active_count.into()),
        );

        Ok(stats)
    }
}
