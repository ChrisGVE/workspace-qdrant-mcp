/// LadybugDB backend configuration.
use std::path::PathBuf;

/// LadybugDB backend configuration.
#[derive(Debug, Clone)]
pub struct LadybugConfig {
    /// Path to the LadybugDB database directory.
    pub db_path: PathBuf,
    /// Buffer pool size in bytes (default: 256 MB, 0 = auto).
    pub buffer_pool_size: u64,
    /// Maximum number of threads for query processing (0 = auto).
    pub max_num_threads: u64,
}

impl Default for LadybugConfig {
    fn default() -> Self {
        let db_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".workspace-qdrant")
            .join("graph");
        Self {
            db_path,
            buffer_pool_size: 0, // auto-detect
            max_num_threads: 4,
        }
    }
}
