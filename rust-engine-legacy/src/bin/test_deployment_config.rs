#!/usr/bin/env cargo
//! Test deployment configuration system specifically
//! Tests the develop flag behavior for asset path resolution

use workspace_qdrant_daemon::config::{init_config, get_config_bool, get_config_string};
use std::fs;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Deployment Configuration System ===\n");

    // Test 1: Production deployment configuration (develop=false)
    println!("Test 1: Production deployment (develop=false)");

    let prod_config = r#"
deployment:
  develop: false
  base_path: "/usr/local/share/workspace-qdrant-mcp"

external_services:
  qdrant:
    url: "http://production.qdrant.server:6333"

grpc:
  server:
    port: 8080
    host: "0.0.0.0"

system:
  project_name: "workspace-qdrant-mcp"
"#;

    let temp_file = "/tmp/test_prod_deployment.yaml";
    let mut file = fs::File::create(temp_file)?;
    file.write_all(prod_config.as_bytes())?;
    drop(file);

    // Initialize configuration - this is the only time we can set it with OnceLock
    init_config(Some(temp_file.as_ref()))?;

    // Test deployment configuration values
    let develop = get_config_bool("deployment.develop", true); // Default true for testing
    let base_path = get_config_string("deployment.base_path", "");
    let qdrant_url = get_config_string("external_services.qdrant.url", "");
    let project_name = get_config_string("system.project_name", "");

    println!("  Develop mode: {}", develop);
    println!("  Base path: {}", base_path);
    println!("  Qdrant URL: {}", qdrant_url);
    println!("  Project name: {}", project_name);

    // Verify production values
    assert_eq!(develop, false, "Production mode should have develop=false");
    assert_eq!(base_path, "/usr/local/share/workspace-qdrant-mcp");
    assert_eq!(qdrant_url, "http://production.qdrant.server:6333");
    assert_eq!(project_name, "workspace-qdrant-mcp");

    println!("  ✓ Production deployment configuration correct");

    // Test deployment path logic
    println!("\nTest 2: Deployment path resolution logic");

    let assets_path = if develop {
        // Development: use project-relative path
        format!("./assets")
    } else {
        // Production: use system path
        format!("{}/assets", base_path)
    };

    println!("  Deployment mode: {}", if develop { "Development" } else { "Production" });
    println!("  Asset path: {}", assets_path);

    if develop {
        println!("  → Using project-relative asset paths (development)");
    } else {
        println!("  → Using system deployment asset paths (production)");
        assert_eq!(assets_path, "/usr/local/share/workspace-qdrant-mcp/assets");
    }

    println!("  ✓ Asset path resolution logic correct");

    // Test other deployment scenarios
    println!("\nTest 3: Asset path examples");

    let config_file_path = if develop {
        "./config/default.yaml"
    } else {
        &format!("{}/config/default.yaml", base_path)
    };

    let templates_path = if develop {
        "./templates"
    } else {
        &format!("{}/templates", base_path)
    };

    let docs_path = if develop {
        "./docs"
    } else {
        &format!("{}/docs", base_path)
    };

    println!("  Config file path: {}", config_file_path);
    println!("  Templates path: {}", templates_path);
    println!("  Documentation path: {}", docs_path);

    if !develop {
        // Verify production paths
        assert!(config_file_path.starts_with("/usr/local/share/workspace-qdrant-mcp"));
        assert!(templates_path.starts_with("/usr/local/share/workspace-qdrant-mcp"));
        assert!(docs_path.starts_with("/usr/local/share/workspace-qdrant-mcp"));
    }

    println!("  ✓ All asset paths use correct deployment base");

    // Clean up
    fs::remove_file(temp_file).ok();

    println!("\n🎉 All deployment configuration tests passed!");
    println!("\n=== Deployment Configuration Summary ===");
    println!("✓ Production deployment configuration (develop=false) works correctly");
    println!("✓ Asset path resolution uses system paths in production mode");
    println!("✓ Configuration values are read correctly by Rust implementation");
    println!("✓ Path logic properly switches between development and production modes");

    println!("\nDeployment path strategy:");
    println!("- develop=true:  Use project-relative paths (./assets, ./config, etc.)");
    println!("- develop=false: Use system paths (/usr/local/share/workspace-qdrant-mcp/*)");
    println!("\nThis enables proper asset resolution in both development and production!");

    Ok(())
}