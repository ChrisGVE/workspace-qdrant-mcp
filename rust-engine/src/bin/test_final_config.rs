use workspace_qdrant_daemon::config::{DaemonConfig, get_config_string};
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use std::path::Path;
use std::error::Error;

#[tokio::main]
async fn main() {
    println!("Testing daemon initialization with final working config...");

    let config_path = Path::new("20250927-2145_final_working_config.yaml");

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

    // Test database path using lua-style configuration
    let database_path = get_config_string("database.sqlite_path", ":memory:");
    println!("Database path: {}", database_path);

    // Now try to initialize the daemon
    println!("\n2. Initializing daemon...");
    match WorkspaceDaemon::new(config).await {
        Ok(_daemon) => {
            println!("✅ Daemon initialized successfully!");
            println!("🎉 The configuration works!");
        },
        Err(e) => {
            println!("❌ Daemon initialization failed: {}", e);
            // Print the error chain to see the real issue
            let mut cause = e.source();
            while let Some(err) = cause {
                println!("  Caused by: {}", err);
                cause = err.source();
            }
        }
    }
}