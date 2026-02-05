//! Unified State Management Module
//!
//! This module handles SQLite-based persistence for daemon state including LSP server metadata,
//! health metrics, communication logs, configuration, and operational state.
//!
//! ## 3-Table SQLite Compliance Note
//!
//! **WARNING**: This module creates the following non-compliant SQLite tables:
//! - `lsp_servers` - Server metadata
//! - `lsp_health_metrics` - Health metrics history
//! - `lsp_communication_logs` - Communication audit logs
//! - `lsp_configurations` - Server configurations
//! - `lsp_project_server_states` - Per-project server state
//!
//! Per the 3-table SQLite compliance requirement, only `schema_version`, `unified_queue`,
//! and `watch_folders` tables are allowed. This module requires refactoring to:
//!
//! 1. **Server preferences**: Move to daemon config under `daemon.lsp.languages`
//! 2. **Project language state**: Store in `watch_folders.metadata` JSON column
//! 3. **Health metrics & logs**: Use structured logging (tracing) instead
//! 4. **Runtime state**: Keep in-memory instead of persisting
//!
//! Until this refactoring is complete, the LSP subsystem will create non-compliant tables.
//! This is tracked for future work.

use std::collections::HashMap;
use std::path::Path;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::{Row, SqlitePool, sqlite::SqlitePoolOptions};
use tracing::{debug, info};
use uuid::Uuid;

use crate::lsp::{
    Language, LspResult, ServerStatus,
};
use crate::lsp::lifecycle::{ServerMetadata, HealthMetrics};

/// LSP server state record in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStateRecord {
    /// Server ID
    pub id: Uuid,
    /// Server name
    pub name: String,
    /// Executable path
    pub executable_path: String,
    /// Supported languages (JSON array)
    pub languages: Vec<Language>,
    /// Server version
    pub version: Option<String>,
    /// Current status
    pub status: ServerStatus,
    /// When the server was first registered
    pub registered_at: DateTime<Utc>,
    /// When the server was last started
    pub last_started_at: Option<DateTime<Utc>>,
    /// When the server was last seen healthy
    pub last_healthy_at: Option<DateTime<Utc>>,
    /// Total uptime in seconds
    pub total_uptime_seconds: i64,
    /// Number of times restarted
    pub restart_count: i32,
    /// Configuration (JSON object)
    pub configuration: HashMap<String, JsonValue>,
    /// Additional metadata (JSON object)
    pub metadata: HashMap<String, JsonValue>,
}

/// Health metrics record in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetricsRecord {
    /// Record ID
    pub id: i64,
    /// Server ID
    pub server_id: Uuid,
    /// Timestamp of this health check
    pub timestamp: DateTime<Utc>,
    /// Server status at this time
    pub status: ServerStatus,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    /// Total requests processed
    pub requests_processed: u64,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<u64>,
    /// CPU usage percentage
    pub cpu_usage_percent: Option<f32>,
    /// Additional metrics (JSON object)
    pub additional_metrics: HashMap<String, JsonValue>,
}

/// Communication log record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationLogRecord {
    /// Record ID
    pub id: i64,
    /// Server ID
    pub server_id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Message direction (request/response/notification)
    pub direction: String,
    /// Message type (method name)
    pub message_type: String,
    /// Message content (truncated for storage)
    pub content: String,
    /// Response time if applicable
    pub response_time_ms: Option<u64>,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Configuration change record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationRecord {
    /// Record ID
    pub id: i64,
    /// Server ID (optional, can be global)
    pub server_id: Option<Uuid>,
    /// Configuration key
    pub key: String,
    /// Configuration value (JSON)
    pub value: JsonValue,
    /// When this configuration was set
    pub set_at: DateTime<Utc>,
    /// Who/what set this configuration
    pub set_by: String,
    /// Previous value (for rollback)
    pub previous_value: Option<JsonValue>,
}

/// Project-specific LSP server state for persistence and recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectServerState {
    /// Project identifier (tenant_id)
    pub project_id: String,
    /// Programming language
    pub language: Language,
    /// Project root path
    pub project_root: String,
    /// Number of times the server was restarted
    pub restart_count: u32,
    /// When the server was last started
    pub last_started_at: DateTime<Utc>,
    /// Server executable path (for re-spawning)
    pub executable_path: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, JsonValue>,
}

/// Key for identifying a project-language combination
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ProjectLanguageKey {
    pub project_id: String,
    pub language: Language,
}

/// Unified state manager for SQLite persistence
pub struct StateManager {
    pool: SqlitePool,
    database_path: std::path::PathBuf,
}

impl StateManager {
    /// Create a new state manager
    pub async fn new<P: AsRef<Path>>(database_path: P) -> LspResult<Self> {
        let database_path = database_path.as_ref().to_path_buf();
        
        info!("Initializing unified state manager with database: {}", 
              database_path.display());

        // Create parent directory if it doesn't exist
        if let Some(parent) = database_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Ensure the database file can be created by touching it
        if !database_path.exists() {
            tokio::fs::File::create(&database_path).await?;
        }

        // Create connection pool
        let database_url = format!("sqlite:{}", database_path.display());
        info!("Connecting to SQLite database: {}", database_url);
        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect(&database_url)
            .await?;

        let manager = Self {
            pool,
            database_path,
        };

        Ok(manager)
    }

    /// Initialize the database schema
    pub async fn initialize(&self) -> LspResult<()> {
        info!("Initializing unified state database schema");

        // Create servers table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS lsp_servers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                executable_path TEXT NOT NULL,
                languages TEXT NOT NULL,  -- JSON array
                version TEXT,
                status TEXT NOT NULL DEFAULT 'stopped',
                registered_at TEXT NOT NULL,
                last_started_at TEXT,
                last_healthy_at TEXT,
                total_uptime_seconds INTEGER NOT NULL DEFAULT 0,
                restart_count INTEGER NOT NULL DEFAULT 0,
                configuration TEXT NOT NULL DEFAULT '{}',  -- JSON object
                metadata TEXT NOT NULL DEFAULT '{}',       -- JSON object
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        "#)
        .execute(&self.pool)
        .await?;

        // Create health metrics table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS lsp_health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time_ms INTEGER NOT NULL,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                requests_processed INTEGER NOT NULL DEFAULT 0,
                avg_response_time_ms REAL NOT NULL DEFAULT 0.0,
                memory_usage_bytes INTEGER,
                cpu_usage_percent REAL,
                additional_metrics TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (server_id) REFERENCES lsp_servers(id) ON DELETE CASCADE
            )
        "#)
        .execute(&self.pool)
        .await?;

        // Create communication logs table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS lsp_communication_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                direction TEXT NOT NULL,  -- 'outgoing' or 'incoming'
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,    -- Truncated message content
                response_time_ms INTEGER,
                success BOOLEAN NOT NULL DEFAULT TRUE,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (server_id) REFERENCES lsp_servers(id) ON DELETE CASCADE
            )
        "#)
        .execute(&self.pool)
        .await?;

        // Create configuration table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS lsp_configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT,  -- NULL for global config
                key TEXT NOT NULL,
                value TEXT NOT NULL,  -- JSON value
                set_at TEXT NOT NULL,
                set_by TEXT NOT NULL,
                previous_value TEXT,   -- For rollback
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (server_id) REFERENCES lsp_servers(id) ON DELETE CASCADE
            )
        "#)
        .execute(&self.pool)
        .await?;

        // Create indexes for performance
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_servers_name ON lsp_servers(name)")
            .execute(&self.pool).await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_servers_status ON lsp_servers(status)")
            .execute(&self.pool).await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_health_server_timestamp ON lsp_health_metrics(server_id, timestamp)")
            .execute(&self.pool).await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_logs_server_timestamp ON lsp_communication_logs(server_id, timestamp)")
            .execute(&self.pool).await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_config_server_key ON lsp_configurations(server_id, key)")
            .execute(&self.pool).await?;

        // Create project server states table (Task 1.18)
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS lsp_project_server_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                language TEXT NOT NULL,
                project_root TEXT NOT NULL,
                restart_count INTEGER NOT NULL DEFAULT 0,
                last_started_at TEXT NOT NULL,
                executable_path TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, language)
            )
        "#)
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_project_server_project ON lsp_project_server_states(project_id)")
            .execute(&self.pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_project_server_language ON lsp_project_server_states(language)")
            .execute(&self.pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_project_server_last_started ON lsp_project_server_states(last_started_at)")
            .execute(&self.pool).await?;

        info!("Unified state database schema initialized successfully");
        Ok(())
    }

    /// Store server metadata
    pub async fn store_server_metadata(&self, metadata: &ServerMetadata) -> LspResult<()> {
        debug!("Storing server metadata for: {}", metadata.name);

        let languages_json = serde_json::to_string(&metadata.languages)?;
        let configuration_json = serde_json::to_string(&HashMap::<String, JsonValue>::new())?;
        let metadata_json = serde_json::to_string(&{
            let mut map = HashMap::new();
            map.insert("process_id".to_string(), 
                      metadata.process_id.map(|id| JsonValue::Number(id.into()))
                             .unwrap_or(JsonValue::Null));
            map.insert("working_directory".to_string(), 
                      JsonValue::String(metadata.working_directory.display().to_string()));
            map.insert("arguments".to_string(), 
                      JsonValue::Array(metadata.arguments.iter()
                                     .map(|s| JsonValue::String(s.clone()))
                                     .collect()));
            map
        })?;

        sqlx::query(r#"
            INSERT OR REPLACE INTO lsp_servers (
                id, name, executable_path, languages, version, status,
                registered_at, last_started_at, configuration, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        "#)
        .bind(metadata.id.to_string())
        .bind(&metadata.name)
        .bind(metadata.executable_path.to_string_lossy().as_ref())
        .bind(languages_json)
        .bind(&metadata.version)
        .bind("initializing")
        .bind(metadata.started_at.to_rfc3339())
        .bind(metadata.started_at.to_rfc3339())
        .bind(configuration_json)
        .bind(metadata_json)
        .execute(&self.pool)
        .await?;

        debug!("Server metadata stored successfully");
        Ok(())
    }

    /// Get server metadata by ID
    pub async fn get_server_metadata(&self, id: &Uuid) -> LspResult<Option<ServerStateRecord>> {
        debug!("Getting server metadata for ID: {}", id);

        let row = sqlx::query(r#"
            SELECT id, name, executable_path, languages, version, status,
                   registered_at, last_started_at, last_healthy_at, total_uptime_seconds,
                   restart_count, configuration, metadata
            FROM lsp_servers WHERE id = ?1
        "#)
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let languages: Vec<Language> = serde_json::from_str(row.get("languages"))?;
            let configuration: HashMap<String, JsonValue> = serde_json::from_str(row.get("configuration"))?;
            let metadata: HashMap<String, JsonValue> = serde_json::from_str(row.get("metadata"))?;

            let record = ServerStateRecord {
                id: Uuid::parse_str(row.get("id"))?,
                name: row.get("name"),
                executable_path: row.get("executable_path"),
                languages,
                version: row.get("version"),
                status: serde_json::from_str(&format!("\"{}\"", row.get::<String, _>("status")))?,
                registered_at: DateTime::parse_from_rfc3339(row.get("registered_at"))?.with_timezone(&Utc),
                last_started_at: row.get::<Option<String>, _>("last_started_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                last_healthy_at: row.get::<Option<String>, _>("last_healthy_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                total_uptime_seconds: row.get("total_uptime_seconds"),
                restart_count: row.get("restart_count"),
                configuration,
                metadata,
            };

            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    /// Update server status
    pub async fn update_server_status(&self, id: &Uuid, status: &ServerStatus) -> LspResult<()> {
        debug!("Updating server status for {}: {:?}", id, status);

        let status_str = serde_json::to_string(status)?.trim_matches('"').to_string();

        sqlx::query(r#"
            UPDATE lsp_servers 
            SET status = ?1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?2
        "#)
        .bind(status_str)
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update health metrics
    pub async fn update_health_metrics(&self, id: &Uuid, metrics: &HealthMetrics) -> LspResult<()> {
        debug!("Updating health metrics for server: {}", id);

        // Update server table with latest health info
        let status_str = serde_json::to_string(&metrics.status)?.trim_matches('"').to_string();
        
        sqlx::query(r#"
            UPDATE lsp_servers 
            SET status = ?1, last_healthy_at = ?2, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?3
        "#)
        .bind(status_str)
        .bind(metrics.last_healthy.to_rfc3339())
        .bind(id.to_string())
        .execute(&self.pool)
        .await?;

        // Insert health metrics record
        let additional_metrics_json = serde_json::to_string(&HashMap::<String, JsonValue>::new())?;
        
        sqlx::query(r#"
            INSERT INTO lsp_health_metrics (
                server_id, timestamp, status, response_time_ms, consecutive_failures,
                requests_processed, avg_response_time_ms, memory_usage_bytes,
                cpu_usage_percent, additional_metrics
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        "#)
        .bind(id.to_string())
        .bind(Utc::now().to_rfc3339())
        .bind(serde_json::to_string(&metrics.status)?.trim_matches('"'))
        .bind(metrics.response_time_ms as i64)
        .bind(metrics.consecutive_failures as i64)
        .bind(metrics.requests_processed as i64)
        .bind(metrics.avg_response_time_ms)
        .bind(metrics.memory_usage_bytes.map(|b| b as i64))
        .bind(metrics.cpu_usage_percent)
        .bind(additional_metrics_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get recent health metrics for a server
    pub async fn get_health_metrics_history(
        &self,
        id: &Uuid,
        limit: u32,
    ) -> LspResult<Vec<HealthMetricsRecord>> {
        debug!("Getting health metrics history for server: {} (limit: {})", id, limit);

        let rows = sqlx::query(r#"
            SELECT id, server_id, timestamp, status, response_time_ms, consecutive_failures,
                   requests_processed, avg_response_time_ms, memory_usage_bytes,
                   cpu_usage_percent, additional_metrics
            FROM lsp_health_metrics
            WHERE server_id = ?1
            ORDER BY timestamp DESC
            LIMIT ?2
        "#)
        .bind(id.to_string())
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut records = Vec::new();
        for row in rows {
            let additional_metrics: HashMap<String, JsonValue> = 
                serde_json::from_str(row.get("additional_metrics"))?;

            records.push(HealthMetricsRecord {
                id: row.get("id"),
                server_id: Uuid::parse_str(row.get("server_id"))?,
                timestamp: DateTime::parse_from_rfc3339(row.get("timestamp"))?.with_timezone(&Utc),
                status: serde_json::from_str(&format!("\"{}\"", row.get::<String, _>("status")))?,
                response_time_ms: row.get::<i64, _>("response_time_ms") as u64,
                consecutive_failures: row.get::<i64, _>("consecutive_failures") as u32,
                requests_processed: row.get::<i64, _>("requests_processed") as u64,
                avg_response_time_ms: row.get("avg_response_time_ms"),
                memory_usage_bytes: row.get::<Option<i64>, _>("memory_usage_bytes")
                    .map(|b| b as u64),
                cpu_usage_percent: row.get("cpu_usage_percent"),
                additional_metrics,
            });
        }

        Ok(records)
    }

    /// Log communication event
    pub async fn log_communication(
        &self,
        server_id: &Uuid,
        direction: &str,
        message_type: &str,
        content: &str,
        response_time_ms: Option<u64>,
        success: bool,
        error_message: Option<&str>,
    ) -> LspResult<()> {
        debug!("Logging communication for server {}: {} {}", server_id, direction, message_type);

        // Truncate content to reasonable size for storage
        let truncated_content = if content.len() > 1000 {
            format!("{}...[truncated from {} chars]", &content[..1000], content.len())
        } else {
            content.to_string()
        };

        sqlx::query(r#"
            INSERT INTO lsp_communication_logs (
                server_id, timestamp, direction, message_type, content,
                response_time_ms, success, error_message
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#)
        .bind(server_id.to_string())
        .bind(Utc::now().to_rfc3339())
        .bind(direction)
        .bind(message_type)
        .bind(truncated_content)
        .bind(response_time_ms.map(|t| t as i64))
        .bind(success)
        .bind(error_message)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Set configuration value
    pub async fn set_configuration(
        &self,
        server_id: Option<&Uuid>,
        key: &str,
        value: JsonValue,
        set_by: &str,
    ) -> LspResult<()> {
        debug!("Setting configuration: {} = {:?}", key, value);

        // Get previous value for rollback
        let previous_value = self.get_configuration(server_id, key).await?;

        sqlx::query(r#"
            INSERT INTO lsp_configurations (
                server_id, key, value, set_at, set_by, previous_value
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#)
        .bind(server_id.map(|id| id.to_string()))
        .bind(key)
        .bind(serde_json::to_string(&value)?)
        .bind(Utc::now().to_rfc3339())
        .bind(set_by)
        .bind(previous_value.map(|v| serde_json::to_string(&v).unwrap_or_default()))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get configuration value
    pub async fn get_configuration(
        &self,
        server_id: Option<&Uuid>,
        key: &str,
    ) -> LspResult<Option<JsonValue>> {
        debug!("Getting configuration: {}", key);

        let row = sqlx::query(r#"
            SELECT value FROM lsp_configurations 
            WHERE (server_id = ?1 OR (?1 IS NULL AND server_id IS NULL))
            AND key = ?2
            ORDER BY id DESC
            LIMIT 1
        "#)
        .bind(server_id.map(|id| id.to_string()))
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let value: JsonValue = serde_json::from_str(row.get("value"))?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Get all active servers
    pub async fn get_active_servers(&self) -> LspResult<Vec<ServerStateRecord>> {
        debug!("Getting all active servers");

        let rows = sqlx::query(r#"
            SELECT id, name, executable_path, languages, version, status,
                   registered_at, last_started_at, last_healthy_at, total_uptime_seconds,
                   restart_count, configuration, metadata
            FROM lsp_servers
            WHERE status IN ('running', 'initializing', 'degraded')
            ORDER BY name
        "#)
        .fetch_all(&self.pool)
        .await?;

        let mut records = Vec::new();
        for row in rows {
            let languages: Vec<Language> = serde_json::from_str(row.get("languages"))?;
            let configuration: HashMap<String, JsonValue> = serde_json::from_str(row.get("configuration"))?;
            let metadata: HashMap<String, JsonValue> = serde_json::from_str(row.get("metadata"))?;

            records.push(ServerStateRecord {
                id: Uuid::parse_str(row.get("id"))?,
                name: row.get("name"),
                executable_path: row.get("executable_path"),
                languages,
                version: row.get("version"),
                status: serde_json::from_str(&format!("\"{}\"", row.get::<String, _>("status")))?,
                registered_at: DateTime::parse_from_rfc3339(row.get("registered_at"))?.with_timezone(&Utc),
                last_started_at: row.get::<Option<String>, _>("last_started_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                last_healthy_at: row.get::<Option<String>, _>("last_healthy_at")
                    .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                total_uptime_seconds: row.get("total_uptime_seconds"),
                restart_count: row.get("restart_count"),
                configuration,
                metadata,
            });
        }

        Ok(records)
    }

    /// Clean up old records
    pub async fn cleanup_old_records(&self, days_to_keep: u32) -> LspResult<u64> {
        info!("Cleaning up records older than {} days", days_to_keep);

        let cutoff_date = Utc::now() - chrono::Duration::days(days_to_keep as i64);
        let cutoff_str = cutoff_date.to_rfc3339();

        // Clean up old health metrics
        let health_result = sqlx::query(r#"
            DELETE FROM lsp_health_metrics WHERE timestamp < ?1
        "#)
        .bind(&cutoff_str)
        .execute(&self.pool)
        .await?;

        // Clean up old communication logs
        let comm_result = sqlx::query(r#"
            DELETE FROM lsp_communication_logs WHERE timestamp < ?1
        "#)
        .bind(&cutoff_str)
        .execute(&self.pool)
        .await?;

        let total_deleted = health_result.rows_affected() + comm_result.rows_affected();
        info!("Cleaned up {} old records", total_deleted);

        Ok(total_deleted)
    }

    /// Store project server state for recovery
    pub async fn store_project_server_state(&self, state: &ProjectServerState) -> LspResult<()> {
        debug!(
            "Storing project server state: project_id={}, language={:?}",
            state.project_id, state.language
        );

        let language_str = serde_json::to_string(&state.language)?.trim_matches('"').to_string();
        let metadata_json = serde_json::to_string(&state.metadata)?;

        sqlx::query(r#"
            INSERT INTO lsp_project_server_states (
                project_id, language, project_root, restart_count, last_started_at,
                executable_path, metadata, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CURRENT_TIMESTAMP)
            ON CONFLICT(project_id, language) DO UPDATE SET
                project_root = excluded.project_root,
                restart_count = excluded.restart_count,
                last_started_at = excluded.last_started_at,
                executable_path = excluded.executable_path,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
        "#)
        .bind(&state.project_id)
        .bind(&language_str)
        .bind(&state.project_root)
        .bind(state.restart_count as i32)
        .bind(state.last_started_at.to_rfc3339())
        .bind(&state.executable_path)
        .bind(&metadata_json)
        .execute(&self.pool)
        .await?;

        debug!("Project server state stored successfully");
        Ok(())
    }

    /// Remove project server state (when server is stopped)
    pub async fn remove_project_server_state(
        &self,
        project_id: &str,
        language: &Language,
    ) -> LspResult<()> {
        debug!(
            "Removing project server state: project_id={}, language={:?}",
            project_id, language
        );

        let language_str = serde_json::to_string(language)?.trim_matches('"').to_string();

        sqlx::query(r#"
            DELETE FROM lsp_project_server_states
            WHERE project_id = ?1 AND language = ?2
        "#)
        .bind(project_id)
        .bind(&language_str)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Restore all project server states from SQLite
    pub async fn restore_project_server_states(
        &self,
    ) -> LspResult<HashMap<ProjectLanguageKey, ProjectServerState>> {
        debug!("Restoring project server states from SQLite");

        let rows = sqlx::query(r#"
            SELECT project_id, language, project_root, restart_count, last_started_at,
                   executable_path, metadata
            FROM lsp_project_server_states
            ORDER BY project_id, language
        "#)
        .fetch_all(&self.pool)
        .await?;

        let mut states = HashMap::new();

        for row in rows {
            let project_id: String = row.get("project_id");
            let language_str: String = row.get("language");
            let language: Language = serde_json::from_str(&format!("\"{}\"", language_str))?;
            let metadata: HashMap<String, JsonValue> = serde_json::from_str(row.get("metadata"))?;

            let state = ProjectServerState {
                project_id: project_id.clone(),
                language: language.clone(),
                project_root: row.get("project_root"),
                restart_count: row.get::<i32, _>("restart_count") as u32,
                last_started_at: DateTime::parse_from_rfc3339(row.get("last_started_at"))?
                    .with_timezone(&Utc),
                executable_path: row.get("executable_path"),
                metadata,
            };

            let key = ProjectLanguageKey {
                project_id,
                language,
            };

            states.insert(key, state);
        }

        info!("Restored {} project server states", states.len());
        Ok(states)
    }

    /// Clean up stale project server states (older than specified hours)
    pub async fn cleanup_stale_project_states(&self, max_age_hours: u32) -> LspResult<u64> {
        info!(
            "Cleaning up project server states older than {} hours",
            max_age_hours
        );

        let cutoff_date = Utc::now() - chrono::Duration::hours(max_age_hours as i64);
        let cutoff_str = cutoff_date.to_rfc3339();

        let result = sqlx::query(r#"
            DELETE FROM lsp_project_server_states
            WHERE last_started_at < ?1
        "#)
        .bind(&cutoff_str)
        .execute(&self.pool)
        .await?;

        let deleted = result.rows_affected();
        if deleted > 0 {
            info!("Cleaned up {} stale project server states", deleted);
        }

        Ok(deleted)
    }

    /// Get project server state count
    pub async fn get_project_server_state_count(&self) -> LspResult<i64> {
        let count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM lsp_project_server_states")
                .fetch_one(&self.pool)
                .await?;
        Ok(count)
    }

    /// Close the state manager
    pub async fn close(&self) -> LspResult<()> {
        info!("Closing unified state manager");
        self.pool.close().await;
        Ok(())
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> LspResult<HashMap<String, JsonValue>> {
        let mut stats = HashMap::new();

        // Count servers by status
        let server_rows = sqlx::query("SELECT status, COUNT(*) as count FROM lsp_servers GROUP BY status")
            .fetch_all(&self.pool)
            .await?;

        let mut server_counts = HashMap::new();
        for row in server_rows {
            let status: String = row.get("status");
            let count: i64 = row.get("count");
            server_counts.insert(status, JsonValue::Number(count.into()));
        }
        stats.insert("servers_by_status".to_string(), JsonValue::Object(server_counts.into_iter().collect()));

        // Get total health metrics records
        let health_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM lsp_health_metrics")
            .fetch_one(&self.pool)
            .await?;
        stats.insert("total_health_records".to_string(), JsonValue::Number(health_count.into()));

        // Get total communication log records
        let comm_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM lsp_communication_logs")
            .fetch_one(&self.pool)
            .await?;
        stats.insert("total_communication_records".to_string(), JsonValue::Number(comm_count.into()));

        // Get total project server states
        let project_state_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM lsp_project_server_states")
            .fetch_one(&self.pool)
            .await?;
        stats.insert("total_project_server_states".to_string(), JsonValue::Number(project_state_count.into()));

        // Database file size
        if let Ok(metadata) = tokio::fs::metadata(&self.database_path).await {
            stats.insert("database_size_bytes".to_string(), JsonValue::Number(metadata.len().into()));
        }

        Ok(stats)
    }
}

impl Clone for StateManager {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            database_path: self.database_path.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_state_manager_creation_and_init() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        let stats = manager.get_stats().await.unwrap();
        assert!(stats.contains_key("servers_by_status"));
    }

    #[tokio::test]
    async fn test_configuration_storage() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Set a configuration value
        let test_value = serde_json::json!({"timeout": 30});
        manager.set_configuration(None, "test_key", test_value.clone(), "test").await.unwrap();

        // Get it back
        let retrieved = manager.get_configuration(None, "test_key").await.unwrap();
        assert_eq!(retrieved, Some(test_value));

        // Get non-existent key
        let missing = manager.get_configuration(None, "missing_key").await.unwrap();
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn test_health_metrics_storage() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        let server_id = Uuid::new_v4();
        
        // Store server metadata first
        let metadata = ServerMetadata {
            id: server_id,
            name: "test-server".to_string(),
            executable_path: "/usr/bin/test".into(),
            languages: vec![Language::Python],
            version: Some("1.0.0".to_string()),
            started_at: Utc::now(),
            process_id: Some(12345),
            working_directory: "/tmp".into(),
            environment: HashMap::new(),
            arguments: vec![],
        };
        manager.store_server_metadata(&metadata).await.unwrap();

        // Store health metrics
        let health_metrics = HealthMetrics {
            status: ServerStatus::Running,
            last_healthy: Utc::now(),
            response_time_ms: 100,
            consecutive_failures: 0,
            requests_processed: 10,
            avg_response_time_ms: 95.5,
            memory_usage_bytes: Some(1024 * 1024),
            cpu_usage_percent: Some(5.5),
        };
        manager.update_health_metrics(&server_id, &health_metrics).await.unwrap();

        // Get metrics history
        let history = manager.get_health_metrics_history(&server_id, 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].status, ServerStatus::Running);
        assert_eq!(history[0].response_time_ms, 100);
    }

    #[tokio::test]
    async fn test_communication_logging() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        let server_id = Uuid::new_v4();
        
        // Store server metadata first
        let metadata = ServerMetadata {
            id: server_id,
            name: "test-server".to_string(),
            executable_path: "/usr/bin/test".into(),
            languages: vec![Language::Python],
            version: Some("1.0.0".to_string()),
            started_at: Utc::now(),
            process_id: Some(12345),
            working_directory: "/tmp".into(),
            environment: HashMap::new(),
            arguments: vec![],
        };
        manager.store_server_metadata(&metadata).await.unwrap();

        // Log communication
        manager.log_communication(
            &server_id,
            "outgoing",
            "initialize",
            r#"{"method": "initialize", "params": {}}"#,
            Some(150),
            true,
            None,
        ).await.unwrap();

        // Verify it's stored
        let stats = manager.get_stats().await.unwrap();
        let comm_count = stats.get("total_communication_records").unwrap().as_u64().unwrap();
        assert_eq!(comm_count, 1);
    }

    #[tokio::test]
    async fn test_project_server_state_storage() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Store a project server state
        let state = ProjectServerState {
            project_id: "test-project-123".to_string(),
            language: Language::Python,
            project_root: "/home/user/project".to_string(),
            restart_count: 2,
            last_started_at: Utc::now(),
            executable_path: Some("/usr/bin/pyright".to_string()),
            metadata: HashMap::new(),
        };
        manager.store_project_server_state(&state).await.unwrap();

        // Verify count
        let count = manager.get_project_server_state_count().await.unwrap();
        assert_eq!(count, 1);

        // Restore and verify
        let restored = manager.restore_project_server_states().await.unwrap();
        assert_eq!(restored.len(), 1);

        let key = ProjectLanguageKey {
            project_id: "test-project-123".to_string(),
            language: Language::Python,
        };
        let restored_state = restored.get(&key).unwrap();
        assert_eq!(restored_state.project_root, "/home/user/project");
        assert_eq!(restored_state.restart_count, 2);
    }

    #[tokio::test]
    async fn test_project_server_state_update() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Store initial state
        let state = ProjectServerState {
            project_id: "test-project".to_string(),
            language: Language::Rust,
            project_root: "/project".to_string(),
            restart_count: 0,
            last_started_at: Utc::now(),
            executable_path: Some("/usr/bin/rust-analyzer".to_string()),
            metadata: HashMap::new(),
        };
        manager.store_project_server_state(&state).await.unwrap();

        // Update with same project_id and language
        let updated_state = ProjectServerState {
            project_id: "test-project".to_string(),
            language: Language::Rust,
            project_root: "/project".to_string(),
            restart_count: 5,  // Updated
            last_started_at: Utc::now(),
            executable_path: Some("/usr/bin/rust-analyzer".to_string()),
            metadata: HashMap::new(),
        };
        manager.store_project_server_state(&updated_state).await.unwrap();

        // Verify count is still 1 (upsert worked)
        let count = manager.get_project_server_state_count().await.unwrap();
        assert_eq!(count, 1);

        // Verify updated value
        let restored = manager.restore_project_server_states().await.unwrap();
        let key = ProjectLanguageKey {
            project_id: "test-project".to_string(),
            language: Language::Rust,
        };
        assert_eq!(restored.get(&key).unwrap().restart_count, 5);
    }

    #[tokio::test]
    async fn test_project_server_state_removal() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Store state
        let state = ProjectServerState {
            project_id: "to-remove".to_string(),
            language: Language::TypeScript,
            project_root: "/ts-project".to_string(),
            restart_count: 0,
            last_started_at: Utc::now(),
            executable_path: None,
            metadata: HashMap::new(),
        };
        manager.store_project_server_state(&state).await.unwrap();
        assert_eq!(manager.get_project_server_state_count().await.unwrap(), 1);

        // Remove it
        manager
            .remove_project_server_state("to-remove", &Language::TypeScript)
            .await
            .unwrap();

        // Verify it's gone
        assert_eq!(manager.get_project_server_state_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_cleanup_stale_project_states() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Store a state from 2 days ago
        let old_state = ProjectServerState {
            project_id: "old-project".to_string(),
            language: Language::Go,
            project_root: "/old".to_string(),
            restart_count: 0,
            last_started_at: Utc::now() - chrono::Duration::hours(50),
            executable_path: None,
            metadata: HashMap::new(),
        };
        manager.store_project_server_state(&old_state).await.unwrap();

        // Store a recent state
        let new_state = ProjectServerState {
            project_id: "new-project".to_string(),
            language: Language::Python,
            project_root: "/new".to_string(),
            restart_count: 0,
            last_started_at: Utc::now(),
            executable_path: None,
            metadata: HashMap::new(),
        };
        manager.store_project_server_state(&new_state).await.unwrap();

        assert_eq!(manager.get_project_server_state_count().await.unwrap(), 2);

        // Cleanup states older than 24 hours
        let deleted = manager.cleanup_stale_project_states(24).await.unwrap();
        assert_eq!(deleted, 1);

        // Verify only new state remains
        assert_eq!(manager.get_project_server_state_count().await.unwrap(), 1);
        let restored = manager.restore_project_server_states().await.unwrap();
        assert!(restored.contains_key(&ProjectLanguageKey {
            project_id: "new-project".to_string(),
            language: Language::Python,
        }));
    }

    #[tokio::test]
    async fn test_multiple_languages_same_project() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = StateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Store multiple languages for same project
        for lang in [Language::Python, Language::Rust, Language::TypeScript] {
            let state = ProjectServerState {
                project_id: "multi-lang".to_string(),
                language: lang,
                project_root: "/multi".to_string(),
                restart_count: 0,
                last_started_at: Utc::now(),
                executable_path: None,
                metadata: HashMap::new(),
            };
            manager.store_project_server_state(&state).await.unwrap();
        }

        // Verify all 3 are stored
        assert_eq!(manager.get_project_server_state_count().await.unwrap(), 3);

        // Restore and verify
        let restored = manager.restore_project_server_states().await.unwrap();
        assert_eq!(restored.len(), 3);

        // Remove just one
        manager
            .remove_project_server_state("multi-lang", &Language::Rust)
            .await
            .unwrap();
        assert_eq!(manager.get_project_server_state_count().await.unwrap(), 2);
    }
}