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

    // Test what the database config looks like using lua-style config
    println!("Original database path: {}", workspace_qdrant_daemon::config::get_config_string("database.sqlite_path", ":memory:"));

    // Create a simple test config that should work
    let test_config = DaemonConfig::default();
    println!("Test config database path: {}", workspace_qdrant_daemon::config::get_config_string("database.sqlite_path", ":memory:"));

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