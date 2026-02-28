//! Tests for ProcessingErrorFeedback and ErrorFeedbackManager (Task 461.13).

use super::super::*;

#[test]
fn test_processing_error_type_as_str() {
    assert_eq!(ProcessingErrorType::FileNotFound.as_str(), "file_not_found");
    assert_eq!(ProcessingErrorType::ParsingError.as_str(), "parsing_error");
    assert_eq!(ProcessingErrorType::QdrantError.as_str(), "qdrant_error");
    assert_eq!(
        ProcessingErrorType::EmbeddingError.as_str(),
        "embedding_error"
    );
    assert_eq!(ProcessingErrorType::Unknown.as_str(), "unknown");
}

#[test]
fn test_processing_error_type_from_str() {
    assert_eq!(
        ProcessingErrorType::from_str("file_not_found"),
        ProcessingErrorType::FileNotFound
    );
    assert_eq!(
        ProcessingErrorType::from_str("parsing_error"),
        ProcessingErrorType::ParsingError
    );
    assert_eq!(
        ProcessingErrorType::from_str("qdrant_error"),
        ProcessingErrorType::QdrantError
    );
    assert_eq!(
        ProcessingErrorType::from_str("embedding_error"),
        ProcessingErrorType::EmbeddingError
    );
    assert_eq!(
        ProcessingErrorType::from_str("other"),
        ProcessingErrorType::Unknown
    );
}

#[test]
fn test_processing_error_type_should_skip_permanently() {
    assert!(ProcessingErrorType::FileNotFound.should_skip_permanently());
    assert!(!ProcessingErrorType::ParsingError.should_skip_permanently());
    assert!(!ProcessingErrorType::QdrantError.should_skip_permanently());
    assert!(!ProcessingErrorType::EmbeddingError.should_skip_permanently());
    assert!(!ProcessingErrorType::Unknown.should_skip_permanently());
}

#[test]
fn test_processing_error_feedback_new() {
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/path/to/file.txt",
        ProcessingErrorType::ParsingError,
        "Failed to parse file",
    );

    assert_eq!(feedback.watch_id, "watch-1");
    assert_eq!(feedback.file_path, "/path/to/file.txt");
    assert_eq!(feedback.error_type, ProcessingErrorType::ParsingError);
    assert_eq!(feedback.error_message, "Failed to parse file");
    assert!(feedback.queue_item_id.is_none());
    assert!(feedback.context.is_empty());
}

#[test]
fn test_processing_error_feedback_with_context() {
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/path/to/file.txt",
        ProcessingErrorType::EmbeddingError,
        "Embedding failed",
    )
    .with_queue_item_id("queue-123")
    .with_context("chunk_index", "5")
    .with_context("model", "all-MiniLM-L6-v2");

    assert_eq!(feedback.queue_item_id, Some("queue-123".to_string()));
    assert_eq!(feedback.context.get("chunk_index"), Some(&"5".to_string()));
    assert_eq!(
        feedback.context.get("model"),
        Some(&"all-MiniLM-L6-v2".to_string())
    );
}

#[tokio::test]
async fn test_error_feedback_manager_record_and_query() {
    let manager = ErrorFeedbackManager::new();

    // Record an error
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/path/to/file.txt",
        ProcessingErrorType::ParsingError,
        "Parse error",
    );
    manager.record_error(feedback).await;

    // Query recent errors
    let errors = manager.get_recent_errors("watch-1").await;
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].file_path, "/path/to/file.txt");
}

#[tokio::test]
async fn test_error_feedback_manager_permanent_skip() {
    let manager = ErrorFeedbackManager::new();

    // Record FileNotFound - should add to permanent skip
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/missing/file.txt",
        ProcessingErrorType::FileNotFound,
        "File not found",
    );
    manager.record_error(feedback).await;

    // Check if file is skipped
    assert!(manager.should_skip_file("watch-1", "/missing/file.txt").await);
    assert!(
        !manager
            .should_skip_file("watch-1", "/other/file.txt")
            .await
    );
    assert!(
        !manager
            .should_skip_file("watch-2", "/missing/file.txt")
            .await
    );
}

#[tokio::test]
async fn test_error_feedback_manager_error_counts() {
    let manager = ErrorFeedbackManager::new();

    // Record multiple errors of different types
    manager
        .record_error(ProcessingErrorFeedback::new(
            "watch-1",
            "file1.txt",
            ProcessingErrorType::ParsingError,
            "error",
        ))
        .await;
    manager
        .record_error(ProcessingErrorFeedback::new(
            "watch-1",
            "file2.txt",
            ProcessingErrorType::ParsingError,
            "error",
        ))
        .await;
    manager
        .record_error(ProcessingErrorFeedback::new(
            "watch-1",
            "file3.txt",
            ProcessingErrorType::QdrantError,
            "error",
        ))
        .await;

    let counts = manager.get_error_counts("watch-1").await;
    assert_eq!(counts.get(&ProcessingErrorType::ParsingError), Some(&2));
    assert_eq!(counts.get(&ProcessingErrorType::QdrantError), Some(&1));
    assert_eq!(counts.get(&ProcessingErrorType::FileNotFound), None);
}

#[tokio::test]
async fn test_error_feedback_manager_remove_skip() {
    let manager = ErrorFeedbackManager::new();

    // Add to skip list
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/missing/file.txt",
        ProcessingErrorType::FileNotFound,
        "Not found",
    );
    manager.record_error(feedback).await;
    assert!(manager.should_skip_file("watch-1", "/missing/file.txt").await);

    // Remove from skip list
    let removed = manager
        .remove_skip("watch-1", "/missing/file.txt")
        .await;
    assert!(removed);
    assert!(
        !manager
            .should_skip_file("watch-1", "/missing/file.txt")
            .await
    );
}

#[tokio::test]
async fn test_error_feedback_manager_clear_skips() {
    let manager = ErrorFeedbackManager::new();

    // Add multiple files to skip list
    for i in 0..5 {
        let feedback = ProcessingErrorFeedback::new(
            "watch-1",
            format!("/missing/file{}.txt", i),
            ProcessingErrorType::FileNotFound,
            "Not found",
        );
        manager.record_error(feedback).await;
    }

    let skipped = manager.get_skipped_files("watch-1").await;
    assert_eq!(skipped.len(), 5);

    // Clear all skips
    manager.clear_skips("watch-1").await;
    let skipped = manager.get_skipped_files("watch-1").await;
    assert!(skipped.is_empty());
}

#[tokio::test]
async fn test_error_feedback_manager_summary() {
    let manager = ErrorFeedbackManager::new();

    // Add errors for multiple watches
    manager
        .record_error(ProcessingErrorFeedback::new(
            "watch-1",
            "file1.txt",
            ProcessingErrorType::ParsingError,
            "error",
        ))
        .await;
    manager
        .record_error(ProcessingErrorFeedback::new(
            "watch-1",
            "file2.txt",
            ProcessingErrorType::FileNotFound,
            "error",
        ))
        .await;
    manager
        .record_error(ProcessingErrorFeedback::new(
            "watch-2",
            "file3.txt",
            ProcessingErrorType::QdrantError,
            "error",
        ))
        .await;

    let summary = manager.get_processing_error_summary().await;
    assert_eq!(summary.len(), 2);

    let watch1_summary = summary.iter().find(|s| s.watch_id == "watch-1");
    assert!(watch1_summary.is_some());
    let watch1 = watch1_summary.unwrap();
    assert_eq!(watch1.recent_error_count, 2);
    assert_eq!(watch1.skipped_file_count, 1); // FileNotFound adds to skip
}

#[tokio::test]
async fn test_error_feedback_manager_max_recent() {
    let manager = ErrorFeedbackManager::new().with_max_recent(3);

    // Add more errors than max
    for i in 0..5 {
        manager
            .record_error(ProcessingErrorFeedback::new(
                "watch-1",
                format!("file{}.txt", i),
                ProcessingErrorType::ParsingError,
                format!("error {}", i),
            ))
            .await;
    }

    let errors = manager.get_recent_errors("watch-1").await;
    assert_eq!(errors.len(), 3); // Should be capped at max
    // Should have the most recent 3 (indices 2, 3, 4)
    assert!(errors.iter().any(|e| e.file_path == "file2.txt"));
    assert!(errors.iter().any(|e| e.file_path == "file3.txt"));
    assert!(errors.iter().any(|e| e.file_path == "file4.txt"));
}
