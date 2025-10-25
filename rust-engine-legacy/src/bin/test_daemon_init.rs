use workspace_qdrant_daemon::config::{DaemonConfig, get_config_string, get_config_bool, get_config_u64, get_config_u16};
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use std::path::Path;

#[tokio::main]
async fn main() {
    println!("Testing daemon initialization...");

    let config_path = Path::new("20250927-2141_test_config_simple.yaml");

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

    // Test lua-style configuration access
    println!("\n2. Testing lua-style configuration access...");

    println!("Testing server configuration...");
    let server_host = get_config_string("server.host", "127.0.0.1");
    let server_port = get_config_u16("server.port", 50051);
    println!("✅ server config worked: {}:{}", server_host, server_port);

    println!("Testing database configuration...");
    let database_path = get_config_string("database.sqlite_path", ":memory:");
    println!("✅ database config worked: {}", database_path);

    println!("Testing qdrant configuration...");
    let qdrant_url = get_config_string("qdrant.url", "http://localhost:6333");
    println!("✅ qdrant config worked: {}", qdrant_url);

    println!("Testing processing configuration...");
    let max_concurrent_tasks = get_config_u64("processing.max_concurrent_tasks", 4);
    println!("✅ processing config worked: {} concurrent tasks", max_concurrent_tasks);

    println!("Testing file_watcher configuration...");
    let file_watcher_enabled = get_config_bool("file_watcher.enabled", true);
    println!("✅ file_watcher config worked: enabled={}", file_watcher_enabled);

    println!("Testing auto_ingestion configuration...");
    let auto_ingestion_enabled = get_config_bool("auto_ingestion.enabled", true);
    println!("✅ auto_ingestion config worked: enabled={}", auto_ingestion_enabled);

    println!("Testing metrics configuration...");
    let metrics_enabled = get_config_bool("metrics.enabled", false);
    println!("✅ metrics config worked: enabled={}", metrics_enabled);

    // Now try to initialize the daemon
    println!("\n3. Initializing daemon...");
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