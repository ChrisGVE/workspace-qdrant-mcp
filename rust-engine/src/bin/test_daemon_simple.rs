use workspace_qdrant_daemon::config::DaemonConfig;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use std::error::Error;

#[tokio::main]
async fn main() {
    println!("Testing daemon initialization with memory database...");

    // Create a configuration with a simple memory database
    let config = DaemonConfig::default();

    // Override the database path to use a memory database
    println!("Using default config with memory database");

    // Test what the database config looks like
    let database_config = config.database();
    println!("Original database path: {}", database_config.sqlite_path);

    // Create a simple test config that should work
    let test_config = DaemonConfig::default();
    println!("Test config database path: {}", test_config.database().sqlite_path);

    // Try the daemon init
    println!("\nInitializing daemon...");
    match WorkspaceDaemon::new(test_config).await {
        Ok(_daemon) => {
            println!("✅ Daemon initialized successfully!");
        },
        Err(e) => {
            println!("❌ Daemon initialization failed: {}", e);
            println!("Error details: {:?}", e);

            // Check the source of the error
            let mut source = e.source();
            while let Some(inner) = source {
                println!("Caused by: {}", inner);
                source = inner.source();
            }
        }
    }
}