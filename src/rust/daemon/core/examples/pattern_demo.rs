//! Pattern Manager demonstration example
//!
//! This example shows how to use the PatternManager to work with embedded
//! YAML patterns for project detection and file filtering.

use workspace_qdrant_core::patterns::PatternManager;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Pattern Manager Demonstration");
    println!("================================\n");

    // Create PatternManager with embedded patterns
    let manager = PatternManager::new()?;
    println!("✅ PatternManager initialized successfully\n");

    // Show pattern statistics
    let stats = manager.pattern_stats();
    println!("📊 Pattern Statistics:");
    println!("  • Ecosystems: {}", stats.ecosystems_count);
    println!("  • Exclude patterns: {}", stats.exclude_patterns_count);
    println!("  • Include patterns: {}", stats.include_patterns_count);
    println!("  • Languages: {}\n", stats.languages_count);

    // Test file exclusion
    println!("🚫 File Exclusion Tests:");
    let test_files = [
        "node_modules/package.json",
        "target/debug/main",
        ".git/config",
        "src/main.rs",
        "README.md",
        ".DS_Store",
    ];

    for file in &test_files {
        let excluded = manager.should_exclude(file);
        let status = if excluded { "❌ EXCLUDED" } else { "✅ allowed" };
        println!("  {} {}", status, file);
    }

    // Test file inclusion
    println!("\n✅ File Inclusion Tests:");
    for file in &test_files {
        let included = manager.should_include(file);
        let status = if included { "✅ INCLUDED" } else { "❌ ignored" };
        println!("  {} {}", status, file);
    }

    // Test language detection
    println!("\n🔍 Language Detection:");
    let extensions = ["rs", "py", "js", "java", "go", "unknown"];
    for ext in &extensions {
        if let Some(lang_info) = manager.language_info(ext) {
            println!("  .{} → {} ({})", ext, lang_info.lsp_id, lang_info.category);
        } else {
            println!("  .{} → ❓ Unknown", ext);
        }
    }

    // Test ecosystem detection
    println!("\n🌍 Ecosystem Detection:");
    let test_projects = [
        ("Rust project", vec!["Cargo.toml", "src/main.rs", "Cargo.lock"]),
        ("Node.js project", vec!["package.json", "node_modules/", "src/app.js"]),
        ("Python project", vec!["requirements.txt", "setup.py", "src/__init__.py"]),
        ("Mixed project", vec!["README.md", "LICENSE"]),
    ];

    for (project_name, files) in &test_projects {
        let files_str: Vec<String> = files.iter().map(|s| s.to_string()).collect();
        if let Some((ecosystem, confidence)) = manager.detect_ecosystem(&files_str) {
            println!("  {} → {} ({:.1}% confidence)", project_name, ecosystem, confidence * 100.0);
        } else {
            println!("  {} → ❓ No ecosystem detected", project_name);
        }
    }

    // Show some supported extensions
    println!("\n📄 Some Supported Extensions:");
    let extensions = manager.supported_extensions();
    let sample_extensions: Vec<String> = extensions.into_iter().take(10).collect();
    for ext in &sample_extensions {
        if let Some(lang_info) = manager.language_info(ext) {
            println!("  .{} → {}", ext, lang_info.lsp_id);
        }
    }
    let total_extensions = manager.supported_extensions().len();
    println!("  ... and {} more\n", total_extensions.saturating_sub(10));

    println!("🎉 Pattern Manager demonstration completed successfully!");
    Ok(())
}