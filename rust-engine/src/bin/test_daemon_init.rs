use workspace_qdrant_daemon::config::DaemonConfig;
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

    // Test legacy compatibility methods
    println!("\n2. Testing legacy compatibility methods...");

    println!("Testing server()...");
    let server_config = config.server();
    println!("✅ server() worked: {}:{}", server_config.host, server_config.port);

    println!("Testing database()...");
    let database_config = config.database();
    println!("✅ database() worked: {}", database_config.sqlite_path);

    println!("Testing qdrant()...");
    let qdrant_config = config.qdrant();
    println!("✅ qdrant() worked: {}", qdrant_config.url);

    println!("Testing processing()...");
    let processing_config = config.processing();
    println!("✅ processing() worked: {} concurrent tasks", processing_config.max_concurrent_tasks);

    println!("Testing file_watcher()...");
    let file_watcher_config = config.file_watcher();
    println!("✅ file_watcher() worked: enabled={}", file_watcher_config.enabled);

    println!("Testing auto_ingestion()...");
    let auto_ingestion_config = config.auto_ingestion();
    println!("✅ auto_ingestion() worked: enabled={}", auto_ingestion_config.enabled);

    println!("Testing metrics()...");
    let metrics_config = config.metrics();
    println!("✅ metrics() worked: enabled={}", metrics_config.enabled);

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