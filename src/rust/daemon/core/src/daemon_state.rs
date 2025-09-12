//! Daemon State Management Module
//!
//! This module handles SQLite-based persistence for document processing daemon
//! operational state, metrics, and configuration.

use std::collections::HashMap;
use std::path::Path;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::{Row, SqlitePool, sqlite::SqlitePoolOptions};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Daemon state management errors
#[derive(thiserror::Error, Debug)]
pub enum DaemonStateError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("State error: {0}")]
    State(String),
}

/// Result type for daemon state operations
pub type DaemonStateResult<T> = Result<T, DaemonStateError>;

/// Daemon operational status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DaemonStatus {
    /// Daemon is starting up
    Starting,
    /// Daemon is running normally
    Running,
    /// Daemon is stopping
    Stopping,
    /// Daemon has stopped
    Stopped,
    /// Daemon encountered an error
    Error,
}

impl std::fmt::Display for DaemonStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DaemonStatus::Starting => write!(f, "starting"),
            DaemonStatus::Running => write!(f, "running"),
            DaemonStatus::Stopping => write!(f, "stopping"),
            DaemonStatus::Stopped => write!(f, "stopped"),
            DaemonStatus::Error => write!(f, "error"),
        }
    }
}

/// Processing metrics record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Number of documents processed
    pub documents_processed: i64,
    /// Number of chunks created
    pub chunks_created: i64,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: i64,
    /// Number of errors encountered
    pub error_count: i64,
    /// Last processing time
    pub last_processed_at: Option<DateTime<Utc>>,
}

/// Daemon state record in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStateRecord {
    /// Daemon instance ID
    pub id: Uuid,
    /// Daemon process ID
    pub pid: Option<u32>,
    /// Current status
    pub status: DaemonStatus,
    /// When the daemon was started
    pub started_at: DateTime<Utc>,
    /// When the daemon was last seen active
    pub last_active_at: DateTime<Utc>,
    /// Processing metrics
    pub metrics: ProcessingMetrics,
    /// Configuration (JSON object)
    pub configuration: HashMap<String, JsonValue>,
    /// Additional metadata (JSON object)
    pub metadata: HashMap<String, JsonValue>,
}

/// Daemon state manager for SQLite persistence
pub struct DaemonStateManager {
    pool: SqlitePool,
}

impl DaemonStateManager {
    /// Create a new daemon state manager
    pub async fn new<P: AsRef<Path>>(database_path: P) -> DaemonStateResult<Self> {
        info!("Initializing daemon state manager with database: {}", 
            database_path.as_ref().display());

        let database_url = format!("sqlite:{}", database_path.as_ref().display());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await?;

        Ok(Self { pool })
    }

    /// Initialize the database schema
    pub async fn initialize(&self) -> DaemonStateResult<()> {
        info!("Initializing daemon state database schema");

        // Create main daemon state table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS daemon_state (
                id TEXT PRIMARY KEY,
                pid INTEGER,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                last_active_at TIMESTAMP NOT NULL,
                documents_processed INTEGER DEFAULT 0,
                chunks_created INTEGER DEFAULT 0,
                total_processing_time_ms INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                last_processed_at TIMESTAMP,
                configuration TEXT NOT NULL DEFAULT '{}',
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create processing logs table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                daemon_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                document_path TEXT,
                processing_time_ms INTEGER,
                error_details TEXT,
                FOREIGN KEY (daemon_id) REFERENCES daemon_state(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create indexes for better performance
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_daemon_status ON daemon_state(status)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_daemon_last_active ON daemon_state(last_active_at)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_logs_daemon_timestamp ON processing_logs(daemon_id, timestamp)")
            .execute(&self.pool)
            .await?;

        info!("Daemon state database schema initialized successfully");
        Ok(())
    }

    /// Store daemon state record
    pub async fn store_daemon_state(&self, state: &DaemonStateRecord) -> DaemonStateResult<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO daemon_state (
                id, pid, status, started_at, last_active_at,
                documents_processed, chunks_created, total_processing_time_ms,
                error_count, last_processed_at, configuration, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
        )
        .bind(state.id.to_string())
        .bind(state.pid.map(|p| p as i64))
        .bind(state.status.to_string())
        .bind(state.started_at)
        .bind(state.last_active_at)
        .bind(state.metrics.documents_processed)
        .bind(state.metrics.chunks_created)
        .bind(state.metrics.total_processing_time_ms)
        .bind(state.metrics.error_count)
        .bind(state.metrics.last_processed_at)
        .bind(serde_json::to_string(&state.configuration)?)
        .bind(serde_json::to_string(&state.metadata)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get daemon state by ID
    pub async fn get_daemon_state(&self, id: &Uuid) -> DaemonStateResult<Option<DaemonStateRecord>> {
        let row = sqlx::query(
            r#"
            SELECT id, pid, status, started_at, last_active_at,
                   documents_processed, chunks_created, total_processing_time_ms,
                   error_count, last_processed_at, configuration, metadata
            FROM daemon_state WHERE id = ?1
            "#,
        )
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let status_str: String = row.try_get("status")?;
            let status = match status_str.as_str() {
                "starting" => DaemonStatus::Starting,
                "running" => DaemonStatus::Running,
                "stopping" => DaemonStatus::Stopping,
                "stopped" => DaemonStatus::Stopped,
                "error" => DaemonStatus::Error,
                _ => DaemonStatus::Error,
            };

            let metrics = ProcessingMetrics {
                documents_processed: row.try_get("documents_processed")?,
                chunks_created: row.try_get("chunks_created")?,
                total_processing_time_ms: row.try_get("total_processing_time_ms")?,
                error_count: row.try_get("error_count")?,
                last_processed_at: row.try_get("last_processed_at")?,
            };

            let configuration: String = row.try_get("configuration")?;
            let metadata: String = row.try_get("metadata")?;

            let record = DaemonStateRecord {
                id: Uuid::parse_str(&row.try_get::<String, _>("id")?)
                    .map_err(|e| DaemonStateError::State(format!("Invalid UUID: {}", e)))?,
                pid: row.try_get::<Option<i64>, _>("pid")?.map(|p| p as u32),
                status,
                started_at: row.try_get("started_at")?,
                last_active_at: row.try_get("last_active_at")?,
                metrics,
                configuration: serde_json::from_str(&configuration)?,
                metadata: serde_json::from_str(&metadata)?,
            };

            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    /// Update daemon status
    pub async fn update_daemon_status(&self, id: &Uuid, status: DaemonStatus) -> DaemonStateResult<()> {
        sqlx::query(
            r#"
            UPDATE daemon_state 
            SET status = ?1, last_active_at = ?2
            WHERE id = ?3
            "#,
        )
        .bind(status.to_string())
        .bind(Utc::now())
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update processing metrics
    pub async fn update_metrics(&self, id: &Uuid, metrics: &ProcessingMetrics) -> DaemonStateResult<()> {
        sqlx::query(
            r#"
            UPDATE daemon_state 
            SET documents_processed = ?1, chunks_created = ?2,
                total_processing_time_ms = ?3, error_count = ?4,
                last_processed_at = ?5, last_active_at = ?6
            WHERE id = ?7
            "#,
        )
        .bind(metrics.documents_processed)
        .bind(metrics.chunks_created)
        .bind(metrics.total_processing_time_ms)
        .bind(metrics.error_count)
        .bind(metrics.last_processed_at)
        .bind(Utc::now())
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Log processing event
    pub async fn log_processing_event(
        &self,
        daemon_id: &Uuid,
        level: &str,
        message: &str,
        document_path: Option<&str>,
        processing_time_ms: Option<i64>,
        error_details: Option<&str>,
    ) -> DaemonStateResult<()> {
        sqlx::query(
            r#"
            INSERT INTO processing_logs (
                daemon_id, timestamp, level, message, document_path,
                processing_time_ms, error_details
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
        )
        .bind(daemon_id.to_string())
        .bind(Utc::now())
        .bind(level)
        .bind(message)
        .bind(document_path)
        .bind(processing_time_ms)
        .bind(error_details)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get active daemon instances
    pub async fn get_active_daemons(&self) -> DaemonStateResult<Vec<DaemonStateRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, pid, status, started_at, last_active_at,
                   documents_processed, chunks_created, total_processing_time_ms,
                   error_count, last_processed_at, configuration, metadata
            FROM daemon_state
            WHERE status IN ('starting', 'running')
            ORDER BY last_active_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut records = Vec::new();
        for row in rows {
            let status_str: String = row.try_get("status")?;
            let status = match status_str.as_str() {
                "starting" => DaemonStatus::Starting,
                "running" => DaemonStatus::Running,
                "stopping" => DaemonStatus::Stopping,
                "stopped" => DaemonStatus::Stopped,
                "error" => DaemonStatus::Error,
                _ => DaemonStatus::Error,
            };

            let metrics = ProcessingMetrics {
                documents_processed: row.try_get("documents_processed")?,
                chunks_created: row.try_get("chunks_created")?,
                total_processing_time_ms: row.try_get("total_processing_time_ms")?,
                error_count: row.try_get("error_count")?,
                last_processed_at: row.try_get("last_processed_at")?,
            };

            let configuration: String = row.try_get("configuration")?;
            let metadata: String = row.try_get("metadata")?;

            let record = DaemonStateRecord {
                id: Uuid::parse_str(&row.try_get::<String, _>("id")?)
                    .map_err(|e| DaemonStateError::State(format!("Invalid UUID: {}", e)))?,
                pid: row.try_get::<Option<i64>, _>("pid")?.map(|p| p as u32),
                status,
                started_at: row.try_get("started_at")?,
                last_active_at: row.try_get("last_active_at")?,
                metrics,
                configuration: serde_json::from_str(&configuration)?,
                metadata: serde_json::from_str(&metadata)?,
            };

            records.push(record);
        }

        Ok(records)
    }

    /// Clean up old log records
    pub async fn cleanup_old_logs(&self, days_to_keep: u32) -> DaemonStateResult<u64> {
        let cutoff_time = Utc::now() - chrono::Duration::days(days_to_keep as i64);

        let result = sqlx::query("DELETE FROM processing_logs WHERE timestamp < ?1")
            .bind(cutoff_time)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    /// Close the database connection
    pub async fn close(&self) -> DaemonStateResult<()> {
        info!("Closing daemon state manager");
        self.pool.close().await;
        Ok(())
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> DaemonStateResult<HashMap<String, JsonValue>> {
        let mut stats = HashMap::new();

        // Count daemons by status
        let daemon_rows = sqlx::query("SELECT status, COUNT(*) as count FROM daemon_state GROUP BY status")
            .fetch_all(&self.pool)
            .await?;

        let mut daemon_stats = HashMap::new();
        for row in daemon_rows {
            let status: String = row.try_get("status")?;
            let count: i64 = row.try_get("count")?;
            daemon_stats.insert(status, JsonValue::Number(count.into()));
        }
        stats.insert("daemon_counts".to_string(), JsonValue::Object(daemon_stats.into_iter().collect()));

        // Total processing logs
        let log_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM processing_logs")
            .fetch_one(&self.pool)
            .await?;
        stats.insert("total_processing_logs".to_string(), JsonValue::Number(log_count.into()));

        // Total documents processed across all daemons
        let total_docs: Option<i64> = sqlx::query_scalar("SELECT SUM(documents_processed) FROM daemon_state")
            .fetch_one(&self.pool)
            .await?;
        stats.insert("total_documents_processed".to_string(), 
                    JsonValue::Number(total_docs.unwrap_or(0).into()));

        Ok(stats)
    }
}

impl Clone for DaemonStateManager {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_daemon_state_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("daemon_test.db");
        
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();
        
        assert!(db_path.exists());
    }

    #[tokio::test]
    async fn test_daemon_state_storage() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("daemon_storage_test.db");
        
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        let daemon_id = Uuid::new_v4();
        let state = DaemonStateRecord {
            id: daemon_id,
            pid: Some(12345),
            status: DaemonStatus::Running,
            started_at: Utc::now(),
            last_active_at: Utc::now(),
            metrics: ProcessingMetrics {
                documents_processed: 42,
                chunks_created: 84,
                total_processing_time_ms: 1000,
                error_count: 0,
                last_processed_at: Some(Utc::now()),
            },
            configuration: HashMap::new(),
            metadata: HashMap::new(),
        };

        manager.store_daemon_state(&state).await.unwrap();

        let retrieved = manager.get_daemon_state(&daemon_id).await.unwrap();
        assert!(retrieved.is_some());
        
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, daemon_id);
        assert_eq!(retrieved.status, DaemonStatus::Running);
        assert_eq!(retrieved.metrics.documents_processed, 42);
    }

    #[tokio::test]
    async fn test_processing_logs() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("daemon_logs_test.db");
        
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        let daemon_id = Uuid::new_v4();
        
        manager.log_processing_event(
            &daemon_id,
            "INFO",
            "Document processed successfully",
            Some("/path/to/document.pdf"),
            Some(500),
            None,
        ).await.unwrap();

        let stats = manager.get_stats().await.unwrap();
        let log_count = stats.get("total_processing_logs").unwrap();
        assert_eq!(log_count.as_i64().unwrap(), 1);
    }
}