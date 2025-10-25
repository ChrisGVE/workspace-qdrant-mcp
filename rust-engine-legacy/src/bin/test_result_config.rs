//! Test program demonstrating the improved Result-based configuration API

use workspace_qdrant_daemon::{config, error::DaemonError};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Result-based Configuration API Test ===\n");

    // Initialize configuration with defaults
    config::init_config(None)?;

    // Test 1: Valid configuration access
    println!("1. Testing valid configuration access:");

    // Old pattern (bad):
    let project_name_old = config::get_config_string("system.project_name", "default");
    println!("   Old API: get_config_string() = '{}'", project_name_old);

    // New pattern (good):
    match config::try_get_config_string("system.project_name") {
        Ok(value) => println!("   New API: try_get_config_string() = Ok('{}')", value),
        Err(e) => println!("   New API: try_get_config_string() = Err({})", e),
    }

    println!();

    // Test 2: Missing configuration key
    println!("2. Testing missing configuration key:");

    // Old pattern (bad): Returns empty string, unclear if missing or actually empty
    let missing_old = config::get_config_string("non.existent.key", "");
    println!("   Old API: get_config_string('non.existent.key', '') = '{}'", missing_old);
    println!("   Problem: Empty string - is it missing or actually empty?");

    // New pattern (good): Clear error indication
    match config::try_get_config_string("non.existent.key") {
        Ok(value) => println!("   New API: try_get_config_string() = Ok('{}')", value),
        Err(DaemonError::ConfigKeyNotFound { path }) => {
            println!("   New API: try_get_config_string() = Err(ConfigKeyNotFound {{ path: '{}' }})", path);
            println!("   Benefit: Clear that the key doesn't exist!");
        },
        Err(e) => println!("   New API: try_get_config_string() = Err({})", e),
    }

    println!();

    // Test 3: Type mismatch
    println!("3. Testing type mismatch:");

    // Try to get a string value as a boolean
    match config::try_get_config_bool("system.project_name") {
        Ok(value) => println!("   New API: try_get_config_bool('system.project_name') = Ok({})", value),
        Err(DaemonError::ConfigTypeMismatch { path, expected_type, actual_type }) => {
            println!("   New API: try_get_config_bool() = Err(ConfigTypeMismatch {{");
            println!("     path: '{}',", path);
            println!("     expected_type: '{}',", expected_type);
            println!("     actual_type: '{}'", actual_type);
            println!("   }})");
            println!("   Benefit: Clear type mismatch error with details!");
        },
        Err(e) => println!("   New API: try_get_config_bool() = Err({})", e),
    }

    println!();

    // Test 4: Proper error handling pattern
    println!("4. Demonstrating proper error handling pattern:");

    // Example of idiomatic Rust error handling
    let port = match config::try_get_config_u16("grpc.server.port") {
        Ok(port) => port,
        Err(DaemonError::ConfigKeyNotFound { .. }) => {
            println!("   Port not configured, using default: 50051");
            50051
        },
        Err(DaemonError::ConfigTypeMismatch { path, expected_type, .. }) => {
            println!("   Invalid port configuration at '{}', expected {}, using default: 50051", path, expected_type);
            50051
        },
        Err(e) => {
            println!("   Configuration error: {}, using default: 50051", e);
            50051
        }
    };

    println!("   Final port value: {}", port);

    println!();
    println!("=== Summary ===");
    println!("✓ Result-based API provides clear error indication");
    println!("✓ Type mismatches are properly reported");
    println!("✓ Missing keys are distinguished from empty values");
    println!("✓ Idiomatic Rust error handling with match expressions");
    println!("✓ No more ambiguous empty string defaults!");

    Ok(())
}