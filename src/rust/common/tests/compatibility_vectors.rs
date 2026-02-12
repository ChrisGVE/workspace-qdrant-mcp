//! Cross-language compatibility test vectors
//!
//! These tests generate known outputs that the TypeScript native bridge
//! tests validate against. If either side changes, both test suites
//! will detect the divergence.

use wqm_common::constants::*;
use wqm_common::hashing::{compute_content_hash, generate_idempotency_key};
use wqm_common::nlp::tokenize;
use wqm_common::project_id::ProjectIdCalculator;
use wqm_common::queue_types::{ItemType, QueueOperation, QueueStatus};

#[test]
fn test_constants_match_expected() {
    assert_eq!(COLLECTION_PROJECTS, "projects");
    assert_eq!(COLLECTION_LIBRARIES, "libraries");
    assert_eq!(COLLECTION_MEMORY, "memory");
    assert_eq!(DEFAULT_QDRANT_URL, "http://localhost:6333");
    assert_eq!(DEFAULT_GRPC_PORT, 50051);
    assert_eq!(DEFAULT_BRANCH, "main");
}

#[test]
fn test_git_url_normalization_vectors() {
    // These vectors must match TypeScript native-bridge.test.ts
    assert_eq!(
        ProjectIdCalculator::normalize_git_url("https://github.com/user/repo.git"),
        "github.com/user/repo"
    );
    assert_eq!(
        ProjectIdCalculator::normalize_git_url("git@github.com:user/repo.git"),
        "github.com/user/repo"
    );
    assert_eq!(
        ProjectIdCalculator::normalize_git_url("http://github.com/user/repo"),
        "github.com/user/repo"
    );
    assert_eq!(
        ProjectIdCalculator::normalize_git_url("https://GitHub.COM/User/Repo.git"),
        "github.com/user/repo"
    );

    // SSH and HTTPS must normalize to same string
    assert_eq!(
        ProjectIdCalculator::normalize_git_url("git@github.com:user/repo.git"),
        ProjectIdCalculator::normalize_git_url("https://github.com/user/repo.git"),
    );
}

#[test]
fn test_project_id_vectors() {
    let calc = ProjectIdCalculator::new();

    // Remote project: 12-char hex
    let id = calc.calculate(
        std::path::Path::new("/home/user/project"),
        Some("https://github.com/user/repo.git"),
        None,
    );
    assert_eq!(id.len(), 12);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(!id.starts_with("local_"));

    // Local project: "local_" + 12-char hex
    let id = calc.calculate(std::path::Path::new("/home/user/project"), None, None);
    assert!(id.starts_with("local_"));
    assert_eq!(id.len(), 18); // "local_" (6) + 12

    // SSH and HTTPS same repo => same ID
    let id1 = calc.calculate(
        std::path::Path::new("/path1"),
        Some("https://github.com/user/repo.git"),
        None,
    );
    let id2 = calc.calculate(
        std::path::Path::new("/path2"),
        Some("git@github.com:user/repo.git"),
        None,
    );
    assert_eq!(id1, id2);

    // Disambiguation produces different IDs
    let id1 = calc.calculate(
        std::path::Path::new("/home/user/work/project"),
        Some("https://github.com/user/repo.git"),
        Some("work/project"),
    );
    let id2 = calc.calculate(
        std::path::Path::new("/home/user/personal/project"),
        Some("https://github.com/user/repo.git"),
        Some("personal/project"),
    );
    assert_ne!(id1, id2);
}

#[test]
fn test_content_hash_vector() {
    // Known SHA256 of "hello world" — must match TypeScript test
    assert_eq!(
        compute_content_hash("hello world"),
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );
}

#[test]
fn test_idempotency_key_vectors() {
    // Must match TypeScript native-bridge.test.ts
    let key = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Ingest,
        "proj_abc123",
        "my-project-code",
        "{}",
    )
    .unwrap();
    assert_eq!(key.len(), 32);
    assert!(key.chars().all(|c| c.is_ascii_hexdigit()));

    // Deterministic
    let key2 = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Ingest,
        "proj_abc123",
        "my-project-code",
        "{}",
    )
    .unwrap();
    assert_eq!(key, key2);

    // Different payload => different key
    let key3 = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Ingest,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/different"}"#,
    )
    .unwrap();
    assert_ne!(key, key3);

    // Invalid op for type => error
    assert!(generate_idempotency_key(
        ItemType::DeleteTenant,
        QueueOperation::Ingest,
        "proj",
        "projects",
        "{}",
    )
    .is_err());
}

#[test]
fn test_tokenization_vectors() {
    // Must match TypeScript native-bridge.test.ts
    let tokens = tokenize("Hello World, this is a test!");
    assert!(tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
    assert!(tokens.contains(&"test".to_string()));
    // Stopwords removed
    assert!(!tokens.contains(&"this".to_string()));
    assert!(!tokens.contains(&"is".to_string()));
    assert!(!tokens.contains(&"a".to_string()));

    // Code tokens
    let tokens = tokenize("fn process_file(path: &str) -> Result<()>");
    assert!(tokens.contains(&"fn".to_string()));
    assert!(tokens.contains(&"process_file".to_string()));
    assert!(tokens.contains(&"path".to_string()));
    assert!(tokens.contains(&"result".to_string()));

    // Empty input => empty output
    assert!(tokenize("").is_empty());

    // All stopwords => empty output
    assert!(tokenize("the and or but").is_empty());
}

#[test]
fn test_queue_type_validation_vectors() {
    // Valid item types
    assert!(ItemType::from_str("file").is_some());
    assert!(ItemType::from_str("content").is_some());
    assert!(ItemType::from_str("folder").is_some());
    assert!(ItemType::from_str("project").is_some());
    assert!(ItemType::from_str("library").is_some());
    assert!(ItemType::from_str("delete_tenant").is_some());
    assert!(ItemType::from_str("delete_document").is_some());
    assert!(ItemType::from_str("rename").is_some());
    assert!(ItemType::from_str("invalid").is_none());
    assert!(ItemType::from_str("").is_none());

    // Valid operations
    assert!(QueueOperation::from_str("ingest").is_some());
    assert!(QueueOperation::from_str("update").is_some());
    assert!(QueueOperation::from_str("delete").is_some());
    assert!(QueueOperation::from_str("scan").is_some());
    assert!(QueueOperation::from_str("invalid").is_none());

    // Valid statuses
    assert!(QueueStatus::from_str("pending").is_some());
    assert!(QueueStatus::from_str("in_progress").is_some());
    assert!(QueueStatus::from_str("done").is_some());
    assert!(QueueStatus::from_str("failed").is_some());
    assert!(QueueStatus::from_str("invalid").is_none());

    // Operation+type validation
    assert!(QueueOperation::Ingest.is_valid_for(ItemType::File));
    assert!(QueueOperation::Update.is_valid_for(ItemType::File));
    assert!(QueueOperation::Delete.is_valid_for(ItemType::File));
    assert!(!QueueOperation::Scan.is_valid_for(ItemType::File));
    assert!(QueueOperation::Scan.is_valid_for(ItemType::Folder));
    assert!(QueueOperation::Delete.is_valid_for(ItemType::DeleteTenant));
    assert!(!QueueOperation::Ingest.is_valid_for(ItemType::DeleteTenant));
    assert!(QueueOperation::Update.is_valid_for(ItemType::Rename));
    assert!(!QueueOperation::Ingest.is_valid_for(ItemType::Rename));
    assert!(!QueueOperation::Update.is_valid_for(ItemType::Folder));
}
