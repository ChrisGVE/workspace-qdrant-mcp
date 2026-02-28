//! Tests for LSP server detection and discovery.

use crate::lsp::{Language, LspServerDetector};

#[test]
fn test_lsp_server_detector() {
    let detector = LspServerDetector::new();

    // Test server knowledge
    assert!(detector.is_known_server("rust-analyzer"));
    assert!(detector.is_known_server("ruff-lsp"));
    assert!(detector.is_known_server("typescript-language-server"));
    assert!(!detector.is_known_server("unknown-server"));

    // Test language server mapping
    let python_servers = detector.get_servers_for_language(&Language::Python);
    assert!(!python_servers.is_empty());
    assert!(python_servers.contains(&"ruff-lsp"));

    let rust_servers = detector.get_servers_for_language(&Language::Rust);
    assert!(!rust_servers.is_empty());
    assert!(rust_servers.contains(&"rust-analyzer"));

    // Note: Server template details are private implementation
}

#[tokio::test]
async fn test_lsp_server_detection() {
    let detector = LspServerDetector::new();

    // This will test what's actually available on the system
    let servers = detector.detect_servers().await.unwrap();

    // We can't assume any specific servers are installed
    // but we can test the detection mechanism
    for server in &servers {
        assert!(!server.name.is_empty());
        assert!(server.path.exists());
        assert!(!server.languages.is_empty());
    }

    println!("Detected {} LSP servers:", servers.len());
    for server in servers {
        println!(
            "  - {} at {} (languages: {:?})",
            server.name,
            server.path.display(),
            server.languages
        );
    }
}

/// Test server detection reports available servers
#[tokio::test]
async fn test_server_detection_reports_installed() {
    let detector = LspServerDetector::new();
    let servers = detector.detect_servers().await.unwrap();

    println!("Detected {} LSP servers on this system:", servers.len());
    for server in &servers {
        println!(
            "  - {} ({}): {:?}",
            server.name,
            server.path.display(),
            server.languages
        );
    }

    // Verify detection structure is valid
    for server in &servers {
        assert!(!server.name.is_empty(), "Server name should not be empty");
        assert!(server.path.exists(), "Server path should exist");
        assert!(
            !server.languages.is_empty(),
            "Server should support at least one language"
        );
    }
}
