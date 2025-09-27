#!/usr/bin/env cargo
//! Test daemon configuration loading
//! This verifies that the daemon can load configuration without the "missing field server" error

use workspace_qdrant_daemon::config::{DaemonConfig, init_config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Daemon Configuration Loading ===\n");

    // Test 1: Load with defaults (no config file)
    println!("Test 1: Loading daemon config with defaults");
    let config = DaemonConfig::load(None)?;
    println!("  ✓ Successfully loaded default configuration");
    println!("  - Qdrant URL: {}", config.qdrant().url);
    println!("  - gRPC Host: {}", config.server().host);
    println!("  - gRPC Port: {}", config.server().port);

    // Test 2: Validate configuration
    println!("\nTest 2: Validating configuration");
    config.validate()?;
    println!("  ✓ Configuration validation passed");

    // Test 3: Test legacy compatibility methods
    println!("\nTest 3: Testing legacy compatibility methods");
    let server_config = config.server();
    let qdrant_config = config.qdrant();
    println!("  - Server host: {}", server_config.host);
    println!("  - Server port: {}", server_config.port);
    println!("  - Qdrant URL: {}", qdrant_config.url);
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