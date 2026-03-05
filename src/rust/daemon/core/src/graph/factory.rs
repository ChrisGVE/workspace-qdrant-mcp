//! Graph backend factory for runtime backend selection.
//!
//! Provides a `GraphConfig` and `create_graph_store` factory that
//! instantiate the appropriate `GraphStore` implementation based on
//! configuration (SQLite CTE or LadybugDB).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::info;

use super::schema::{GraphDbError, GraphDbManager, GraphDbResult};
use super::shared::SharedGraphStore;
use super::sqlite_store::SqliteGraphStore;

/// Graph backend selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GraphBackend {
    /// SQLite with recursive CTEs (default, no extra dependencies).
    Sqlite,
    /// LadybugDB (Kuzu fork) — requires `ladybug` feature flag.
    Ladybug,
}

impl Default for GraphBackend {
    fn default() -> Self {
        Self::Sqlite
    }
}

/// Configuration for graph store creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Which backend to use.
    #[serde(default)]
    pub backend: GraphBackend,
    /// Directory for graph database files.
    /// Defaults to the same directory as `state.db`.
    #[serde(default)]
    pub db_dir: Option<PathBuf>,
    /// LadybugDB-specific: buffer pool size in bytes (default: 256MB).
    #[serde(default = "default_buffer_pool")]
    pub buffer_pool_size: u64,
    /// LadybugDB-specific: max worker threads (default: 2).
    #[serde(default = "default_max_threads")]
    pub max_threads: u64,
}

fn default_buffer_pool() -> u64 {
    256 * 1024 * 1024
}

fn default_max_threads() -> u64 {
    2
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            backend: GraphBackend::default(),
            db_dir: None,
            buffer_pool_size: default_buffer_pool(),
            max_threads: default_max_threads(),
        }
    }
}

/// Create a shared graph store based on configuration.
///
/// For SQLite backend: initializes `graph.db` via `GraphDbManager`,
/// creates `SqliteGraphStore`, and wraps in `SharedGraphStore`.
///
/// For LadybugDB: requires `ladybug` feature flag at compile time.
/// Returns an error if the feature is not enabled.
pub async fn create_sqlite_graph_store(
    db_dir: &Path,
) -> GraphDbResult<SharedGraphStore<SqliteGraphStore>> {
    let graph_db_path = db_dir.join(super::GRAPH_DB_FILENAME);
    info!(
        "Initializing SQLite graph store at: {}",
        graph_db_path.display()
    );

    let manager = GraphDbManager::new(&graph_db_path).await?;
    let store = SqliteGraphStore::new(manager.pool().clone());
    Ok(SharedGraphStore::new(store))
}

/// Create a LadybugDB-backed shared graph store.
///
/// Only available when compiled with the `ladybug` feature flag.
#[cfg(feature = "ladybug")]
pub async fn create_ladybug_graph_store(
    db_dir: &Path,
    config: &GraphConfig,
) -> GraphDbResult<SharedGraphStore<super::LadybugGraphStore>> {
    use super::ladybug_store::{LadybugConfig, LadybugGraphStore};

    let ladybug_dir = db_dir.join("ladybug");
    info!(
        "Initializing LadybugDB graph store at: {}",
        ladybug_dir.display()
    );

    let lb_config = LadybugConfig {
        db_path: ladybug_dir,
        buffer_pool_size: config.buffer_pool_size,
        max_num_threads: config.max_threads,
    };

    let store = LadybugGraphStore::new(lb_config)?;
    Ok(SharedGraphStore::new(store))
}

/// Validate that the requested backend is available at compile time.
///
/// Returns `Ok(())` if the backend is available, or an error describing
/// what feature flag needs to be enabled.
pub fn validate_backend(backend: &GraphBackend) -> GraphDbResult<()> {
    match backend {
        GraphBackend::Sqlite => Ok(()),
        GraphBackend::Ladybug => {
            #[cfg(feature = "ladybug")]
            {
                Ok(())
            }
            #[cfg(not(feature = "ladybug"))]
            {
                Err(GraphDbError::InvalidInput(
                    "LadybugDB backend requires the 'ladybug' feature flag. \
                     Rebuild with: cargo build --features ladybug"
                        .to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_config_defaults() {
        let config = GraphConfig::default();
        assert_eq!(config.backend, GraphBackend::Sqlite);
        assert!(config.db_dir.is_none());
        assert_eq!(config.buffer_pool_size, 256 * 1024 * 1024);
        assert_eq!(config.max_threads, 2);
    }

    #[test]
    fn test_graph_backend_serde() {
        let json = r#""sqlite""#;
        let backend: GraphBackend = serde_json::from_str(json).unwrap();
        assert_eq!(backend, GraphBackend::Sqlite);

        let json = r#""ladybug""#;
        let backend: GraphBackend = serde_json::from_str(json).unwrap();
        assert_eq!(backend, GraphBackend::Ladybug);
    }

    #[test]
    fn test_graph_config_serde() {
        let yaml = r#"
backend: sqlite
buffer_pool_size: 134217728
max_threads: 4
"#;
        let config: GraphConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.backend, GraphBackend::Sqlite);
        assert_eq!(config.buffer_pool_size, 128 * 1024 * 1024);
        assert_eq!(config.max_threads, 4);
    }

    #[test]
    fn test_validate_sqlite_backend() {
        assert!(validate_backend(&GraphBackend::Sqlite).is_ok());
    }

    #[test]
    fn test_validate_ladybug_backend() {
        let result = validate_backend(&GraphBackend::Ladybug);
        // Should be Ok if compiled with ladybug feature, Err otherwise
        #[cfg(feature = "ladybug")]
        assert!(result.is_ok());
        #[cfg(not(feature = "ladybug"))]
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_sqlite_graph_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = create_sqlite_graph_store(dir.path()).await;
        assert!(
            store.is_ok(),
            "Should create SQLite graph store: {:?}",
            store.err()
        );
    }
}
