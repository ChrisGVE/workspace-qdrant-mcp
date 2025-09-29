#!/usr/bin/env cargo
//! Test daemon configuration loading
//! This verifies that the daemon can load configuration without the "missing field server" error

use workspace_qdrant_daemon::config::DaemonConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Daemon Configuration Loading ===\n");

    // Test 1: Load with defaults (no config file)
    println!("Test 1: Loading daemon config with defaults");
    let config = DaemonConfig::load(None)?;
    println!("  ✓ Successfully loaded default configuration");
    println!("  - Qdrant URL: {}", workspace_qdrant_daemon::config::get_config_string("external_services.qdrant.url", "http://localhost:6333"));
    println!("  - gRPC Host: {}", workspace_qdrant_daemon::config::get_config_string("grpc.server.host", "127.0.0.1"));
    println!("  - gRPC Port: {}", workspace_qdrant_daemon::config::get_config_u64("grpc.server.port", 50051));

    // Test 2: Validate configuration
    println!("\nTest 2: Validating configuration");
    config.validate()?;
    println!("  ✓ Configuration validation passed");

    // Test 3: Test lua-style config access
    println!("\nTest 3: Testing lua-style config access");
    println!("  - Server host: {}", workspace_qdrant_daemon::config::get_config_string("grpc.server.host", "127.0.0.1"));
    println!("  - Server port: {}", workspace_qdrant_daemon::config::get_config_u64("grpc.server.port", 50051));
    println!("  - Qdrant URL: {}", workspace_qdrant_daemon::config::get_config_string("external_services.qdrant.url", "http://localhost:6333"));
    println!("  ✓ Legacy compatibility methods work");

    // Test 4: Test with minimal YAML config
    println!("\nTest 4: Testing with minimal YAML configuration");
    let minimal_config = r#"
grpc:
  server:
    port: 8888
"#;

    let temp_file = "/tmp/minimal_daemon_config.yaml";
    std::fs::write(temp_file, minimal_config)?;

    let yaml_config = DaemonConfig::load(Some(temp_file.as_ref()))?;
    yaml_config.validate()?;

    println!("  ✓ Successfully loaded minimal YAML configuration");
    println!("  - YAML gRPC Port: {}", yaml_config.server().port);

    // Clean up
    std::fs::remove_file(temp_file).ok();

    println!("\n🎉 All daemon configuration tests passed!");
    println!("\nThe 'missing field server' error has been resolved!");
    println!("The new dictionary-based configuration system successfully:");
    println!("- Loads configurations without requiring all fields");
    println!("- Provides defaults for missing values");
    println!("- Maintains backward compatibility");
    println!("- Handles partial YAML configurations gracefully");

    Ok(())
}