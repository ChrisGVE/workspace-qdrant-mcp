//! Tests for metadata enrichment and collection type parsing.

use std::collections::HashMap;
use std::fs;

use tempfile::tempdir;

use super::collection_type::CollectionType;
use super::enrichment::enrich_metadata;

#[test]
fn test_parse_project_collection() {
    let ctype = CollectionType::from_name("_0f72d776622e");
    match ctype {
        CollectionType::Project { project_id } => {
            assert_eq!(project_id, "0f72d776622e");
        }
        _ => panic!("Expected Project collection type"),
    }
}

#[test]
fn test_parse_library_collection() {
    let ctype = CollectionType::from_name("_fastapi");
    match ctype {
        CollectionType::Library { library_name } => {
            assert_eq!(library_name, "fastapi");
        }
        _ => panic!("Expected Library collection type"),
    }

    let ctype = CollectionType::from_name("_pandas-stubs");
    match ctype {
        CollectionType::Library { library_name } => {
            assert_eq!(library_name, "pandas-stubs");
        }
        _ => panic!("Expected Library collection type"),
    }
}

#[test]
fn test_parse_user_collection() {
    let ctype = CollectionType::from_name("myapp-notes");
    match ctype {
        CollectionType::User {
            basename,
            collection_type,
        } => {
            assert_eq!(basename, "myapp");
            assert_eq!(collection_type, "notes");
        }
        _ => panic!("Expected User collection type"),
    }

    let ctype = CollectionType::from_name("workspace-qdrant-docs");
    match ctype {
        CollectionType::User {
            basename,
            collection_type,
        } => {
            assert_eq!(basename, "workspace-qdrant");
            assert_eq!(collection_type, "docs");
        }
        _ => panic!("Expected User collection type"),
    }
}

#[test]
fn test_parse_rules_collection() {
    let ctype = CollectionType::from_name("rules");
    assert!(matches!(ctype, CollectionType::Rules));

    // Legacy "memory" name also maps to Rules
    let ctype = CollectionType::from_name("memory");
    assert!(matches!(ctype, CollectionType::Rules));
}

#[test]
fn test_parse_edge_cases() {
    // Single underscore (too short for project ID)
    let ctype = CollectionType::from_name("_");
    assert!(matches!(ctype, CollectionType::Library { .. }));

    // Underscore with non-hex characters (library)
    let ctype = CollectionType::from_name("_numpy123");
    assert!(matches!(ctype, CollectionType::Library { .. }));

    // No dash (user collection with no type)
    let ctype = CollectionType::from_name("mycollection");
    match ctype {
        CollectionType::User {
            basename,
            collection_type,
        } => {
            assert_eq!(basename, "mycollection");
            assert_eq!(collection_type, "");
        }
        _ => panic!("Expected User collection type"),
    }
}

#[test]
fn test_enrich_project_collection_metadata() {
    let temp_dir = tempdir().unwrap();
    let test_file = temp_dir.path().join("main.py");
    fs::write(&test_file, "print('hello')").unwrap();

    let metadata = enrich_metadata("_0f72d776622e", &test_file, None, None, None);

    assert_eq!(
        metadata.get("project_id"),
        Some(&"0f72d776622e".to_string())
    );
    assert!(metadata.contains_key("branch"));
    assert_eq!(metadata.get("branch"), Some(&"main".to_string())); // No git repo
    assert_eq!(metadata.get("file_type"), Some(&"code".to_string()));
    assert_eq!(metadata.get("extension"), Some(&"py".to_string()));
    assert_eq!(metadata.get("is_test"), Some(&"false".to_string()));
}

#[test]
fn test_enrich_user_collection_auto_detect() {
    let temp_dir = tempdir().unwrap();
    let test_file = temp_dir.path().join("notes.md");
    fs::write(&test_file, "# Notes").unwrap();

    // Simulate MCP source (auto-detect)
    let metadata = enrich_metadata("myapp-notes", &test_file, None, Some("McpServer"), None);

    assert!(metadata.contains_key("project_id"));
    assert!(!metadata.contains_key("branch")); // USER collections don't get branch
}

#[test]
fn test_enrich_user_collection_cli_with_project() {
    let temp_dir = tempdir().unwrap();
    let test_file = temp_dir.path().join("notes.md");
    fs::write(&test_file, "# Notes").unwrap();

    // Simulate CLI with explicit project_id
    let mut base = HashMap::new();
    base.insert("project_id".to_string(), "abc123".to_string());

    let metadata = enrich_metadata(
        "myapp-notes",
        &test_file,
        Some(base),
        Some("CliCommand"),
        None,
    );

    assert_eq!(metadata.get("project_id"), Some(&"abc123".to_string()));
    assert!(!metadata.contains_key("branch"));
}

#[test]
fn test_enrich_user_collection_cli_without_project() {
    let temp_dir = tempdir().unwrap();
    let test_file = temp_dir.path().join("notes.md");
    fs::write(&test_file, "# Notes").unwrap();

    // Simulate CLI without project_id
    let metadata = enrich_metadata("myapp-notes", &test_file, None, Some("CliCommand"), None);

    assert!(!metadata.contains_key("project_id"));
    assert!(!metadata.contains_key("branch"));
}

#[test]
fn test_enrich_library_collection_metadata() {
    let temp_dir = tempdir().unwrap();
    let test_file = temp_dir.path().join("fastapi_doc.md");
    fs::write(&test_file, "# FastAPI").unwrap();

    let metadata = enrich_metadata("_fastapi", &test_file, None, None, None);

    assert_eq!(
        metadata.get("library_name"),
        Some(&"fastapi".to_string())
    );
    assert!(!metadata.contains_key("project_id"));
    assert!(!metadata.contains_key("branch"));
    assert!(!metadata.contains_key("file_type"));
}

#[test]
fn test_enrich_rules_collection_metadata() {
    let temp_dir = tempdir().unwrap();
    let test_file = temp_dir.path().join("note.txt");
    fs::write(&test_file, "Note").unwrap();

    let metadata = enrich_metadata("rules", &test_file, None, None, None);

    assert_eq!(metadata.get("scope"), Some(&"global".to_string()));
    assert!(!metadata.contains_key("project_id"));
    assert!(!metadata.contains_key("branch"));
}

#[test]
fn test_file_type_classification() {
    let temp_dir = tempdir().unwrap();

    // Test code file
    let py_file = temp_dir.path().join("main.py");
    fs::write(&py_file, "").unwrap();
    let metadata = enrich_metadata("_abc123def456", &py_file, None, None, None);
    assert_eq!(metadata.get("file_type"), Some(&"code".to_string()));
    assert_eq!(metadata.get("extension"), Some(&"py".to_string()));
    assert_eq!(metadata.get("is_test"), Some(&"false".to_string()));

    // Test file is still classified as "code" (test detection is separate)
    let test_file = temp_dir.path().join("test_main.py");
    fs::write(&test_file, "").unwrap();
    let metadata = enrich_metadata("_abc123def456", &test_file, None, None, None);
    assert_eq!(metadata.get("file_type"), Some(&"code".to_string()));
    assert_eq!(metadata.get("is_test"), Some(&"true".to_string()));

    // Test text file (was "docs", now "text" for lightweight markup)
    let md_file = temp_dir.path().join("README.md");
    fs::write(&md_file, "").unwrap();
    let metadata = enrich_metadata("_abc123def456", &md_file, None, None, None);
    assert_eq!(metadata.get("file_type"), Some(&"text".to_string()));
    assert_eq!(metadata.get("extension"), Some(&"md".to_string()));
    assert_eq!(metadata.get("is_test"), Some(&"false".to_string()));
}

#[test]
fn test_base_metadata_preservation() {
    let temp_dir = tempdir().unwrap();
    let test_file = temp_dir.path().join("test.txt");
    fs::write(&test_file, "test").unwrap();

    let mut base = HashMap::new();
    base.insert("custom_field".to_string(), "custom_value".to_string());

    let metadata = enrich_metadata("rules", &test_file, Some(base), None, None);

    // Base metadata should be preserved
    assert_eq!(
        metadata.get("custom_field"),
        Some(&"custom_value".to_string())
    );
    // Rules-specific metadata should be added
    assert_eq!(metadata.get("scope"), Some(&"global".to_string()));
}
