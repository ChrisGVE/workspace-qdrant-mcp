use workspace_qdrant_daemon::config::DaemonConfig;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use std::path::Path;

#[tokio::main]
async fn main() {
    println!("Testing daemon initialization with new config...");

    let config_path = Path::new("20250927-2143_working_config.yaml");

    // Load config
    println!("1. Loading configuration...");
    let config = match DaemonConfig::load(Some(config_path)) {
        Ok(config) => {
            println!("✅ Configuration loaded successfully!");
            config
        },
        Err(e) => {
            println!("❌ Configuration loading failed: {}", e);
            return;
        }
    };

    // Test database path
    let database_config = config.database();
    println!("Database path: {}", database_config.sqlite_path);

    // Now try to initialize the daemon
    println!("\n2. Initializing daemon...");
    match WorkspaceDaemon::new(config).await {
        Ok(_daemon) => {
            println!("✅ Daemon initialized successfully!");
        },
        Err(e) => {
            println!("❌ Daemon initialization failed: {}", e);
            println!("Error details: {:?}", e);
        }
    }
}