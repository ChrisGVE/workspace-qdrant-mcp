// Test that the new PRDv3 configuration can be loaded from YAML
use std::path::Path;

// Import our new config module
mod config;
use config::DaemonConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing PRDv3 configuration loading...");

    // Test 1: Create default configuration
    let default_config = DaemonConfig::default();
    println!("✓ Default configuration created");
    println!("  Project: {}", default_config.system.project_name);
    println!("  Version: {}", default_config.system.version);
    println!("  gRPC Host: {}", default_config.grpc.server.host);
    println!("  gRPC Port: {}", default_config.grpc.server.port);
    println!("  Qdrant URL: {}", default_config.external_services.qdrant.url);

    // Test 2: Validate configuration
    match default_config.validate() {
        Ok(()) => println!("✓ Configuration validation passed"),
        Err(e) => {
            println!("✗ Configuration validation failed: {}", e);
            return Err(e.into());
        }
    }

    // Test 3: Test compatibility methods
    let server_config = default_config.server();
    println!("✓ Legacy server config accessible: {}:{}", server_config.host, server_config.port);

    let database_config = default_config.database();
    println!("✓ Legacy database config accessible: {}", database_config.sqlite_path);

    let qdrant_config = default_config.qdrant();
    println!("✓ Legacy qdrant config accessible: {}", qdrant_config.url);

    // Test 4: Test environment variable loading
    match DaemonConfig::load(None) {
        Ok(env_config) => {
            println!("✓ Configuration loaded from environment/defaults");
            println!("  Host: {}", env_config.grpc.server.host);
        }
        Err(e) => {
            println!("✗ Environment configuration loading failed: {}", e);
            return Err(e.into());
        }
    }

    // Test 5: Create YAML and test serialization
    let yaml_content = serde_yaml::to_string(&default_config)?;
    println!("✓ Configuration serialized to YAML ({} bytes)", yaml_content.len());

    // Test 6: Deserialize from YAML
    let _parsed_config: DaemonConfig = serde_yaml::from_str(&yaml_content)?;
    println!("✓ Configuration deserialized from YAML");

    println!("\n🎉 All PRDv3 configuration tests passed!");
    Ok(())
}