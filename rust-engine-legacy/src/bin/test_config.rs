use workspace_qdrant_daemon::config::DaemonConfig;
use std::path::Path;

fn main() {
    let config_path = Path::new("20250927-2141_test_config_simple.yaml");

    println!("Testing configuration loading...");

    match DaemonConfig::load(Some(config_path)) {
        Ok(config) => {
            println!("✅ Configuration loaded successfully!");
            println!("Project name: {}", config.system.project_name);
            println!("gRPC server enabled: {}", config.grpc.server.enabled);
            println!("gRPC server port: {}", config.grpc.server.port);

            // Test the legacy compatibility methods
            println!("\nTesting legacy compatibility:");
            let server_config = config.server();
            println!("Legacy server host: {}", server_config.host);
            println!("Legacy server port: {}", server_config.port);
        },
        Err(e) => {
            println!("❌ Configuration loading failed: {}", e);
        }
    }
}