//! FTS5 Edge Case Tests — Line Endings, Long Lines, Whitespace, and Sparse Content
//!
//! Covers:
//! - CRLF line endings
//! - Mixed line endings (CRLF + LF)
//! - Trailing newline vs no trailing newline
//! - Very long lines (>10KB)
//! - Whitespace-only files
//! - Sparse content (many blank lines)

use tempfile::TempDir;

use workspace_qdrant_core::fts_batch_processor::{
    FileChange, FtsBatchConfig, FtsBatchProcessor,
};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, SearchOptions};

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

// ---------------------------------------------------------------------------
// CRLF line endings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_crlf_content_indexable_and_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Content with CRLF line endings (Windows-style)
    let content = "fn main() {\r\n    println!(\"hello\");\r\n}\r\n";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/main.rs", None, None, None)
        .await
        .unwrap();

    // Search should find content despite \r being part of lines
    let results = search_exact(&db, "println", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    // The matched content may include \r but should still match
    let results = search_exact(&db, "hello", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_crlf_diff_update_works() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Original with CRLF
    let original = "fn main() {\r\n    old_code();\r\n}\r\n";

    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: original.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/main.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    // Update: change content, still CRLF
    let updated = "fn main() {\r\n    new_code();\r\n}\r\n";

    processor.add_change(FileChange {
        file_id: 1,
        old_content: original.to_string(),
        new_content: updated.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/main.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    // Old content gone, new content present
    let old = search_exact(&db, "old_code", &SearchOptions::default())
        .await
        .unwrap();
    assert!(old.matches.is_empty(), "old_code should be gone after update");

    let new = search_exact(&db, "new_code", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(new.matches.len(), 1, "new_code should be found");
}

#[tokio::test]
async fn test_mixed_line_endings_crlf_and_lf() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Mixed: some lines CRLF, some LF
    let content = "line_one\r\nline_two\nline_three\r\nline_four\n";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/mixed.rs", None, None, None)
        .await
        .unwrap();

    // All lines should be searchable
    for term in &["line_one", "line_two", "line_three", "line_four"] {
        let results = search_exact(&db, term, &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(
            results.matches.len(),
            1,
            "{} should be found in mixed line endings file",
            term
        );
    }
}

// ---------------------------------------------------------------------------
// Trailing newline behavior
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_trailing_newline_vs_no_trailing() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Without trailing newline: split('\n') gives ["line1", "line2"]
    let no_trailing = "trailing_test_a\ntrailing_test_b";
    processor
        .full_rewrite(1, no_trailing, "proj1", Some("main"), "src/no_trail.rs", None, None, None)
        .await
        .unwrap();

    // With trailing newline: split('\n') gives ["line1", "line2", ""]
    let with_trailing = "trailing_test_c\ntrailing_test_d\n";
    processor
        .full_rewrite(2, with_trailing, "proj1", Some("main"), "src/with_trail.rs", None, None, None)
        .await
        .unwrap();

    // Both files should have their content searchable
    for term in &[
        "trailing_test_a",
        "trailing_test_b",
        "trailing_test_c",
        "trailing_test_d",
    ] {
        let results = search_exact(&db, term, &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1, "{} should be found", term);
    }
}

// ---------------------------------------------------------------------------
// Very long lines
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_very_long_line_indexed_and_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Create a line that's ~20KB long
    let long_prefix = "let data = \"";
    let long_middle = "x".repeat(20_000);
    let long_suffix = "\";";
    let long_line = format!("{}{}{}", long_prefix, long_middle, long_suffix);
    let content = format!("fn main() {{\n    {}\n}}", long_line);

    processor
        .full_rewrite(1, &content, "proj1", Some("main"), "src/long.rs", None, None, None)
        .await
        .unwrap();

    // Should find content at the start of the long line
    let results = search_exact(&db, "let data", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Should find fn main
    let results = search_exact(&db, "fn main", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_multiple_long_lines_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // 10 lines each ~5KB
    let mut lines = Vec::new();
    for i in 0..10 {
        let padding = "a".repeat(5_000);
        lines.push(format!("long_marker_{} {}", i, padding));
    }
    let content = lines.join("\n");

    processor
        .full_rewrite(1, &content, "proj1", Some("main"), "src/multilong.rs", None, None, None)
        .await
        .unwrap();

    // Each marker should be findable (use paren-free unique markers)
    for i in 0..10 {
        let results = search_exact(
            &db,
            &format!("long_marker_{} a", i),
            &SearchOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            results.matches.len(),
            1,
            "long_marker_{} should have exactly 1 match",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Whitespace-only files
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_whitespace_only_file() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // File with only spaces, tabs, and newlines
    let content = "   \n\t\t\n  \n   \t   \n";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/whitespace.rs", None, None, None)
        .await
        .unwrap();

    // Search for non-whitespace should find nothing
    let results = search_exact(&db, "function", &SearchOptions::default())
        .await
        .unwrap();
    assert!(results.matches.is_empty());
}

#[tokio::test]
async fn test_single_space_file() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor
        .full_rewrite(1, " ", "proj1", Some("main"), "src/space.rs", None, None, None)
        .await
        .unwrap();

    // Should not crash, file should be indexed
    let results = search_exact(&db, "anything", &SearchOptions::default())
        .await
        .unwrap();
    assert!(results.matches.is_empty());
}

// ---------------------------------------------------------------------------
// Sparse content (many blank lines)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sparse_content_with_many_blank_lines() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // File with content separated by many blank lines
    let mut lines = Vec::new();
    lines.push("fn sparse_start()".to_string());
    for _ in 0..100 {
        lines.push(String::new());
    }
    lines.push("fn sparse_middle()".to_string());
    for _ in 0..100 {
        lines.push(String::new());
    }
    lines.push("fn sparse_end()".to_string());
    let content = lines.join("\n");

    processor
        .full_rewrite(1, &content, "proj1", Some("main"), "src/sparse.rs", None, None, None)
        .await
        .unwrap();

    // All three functions should be searchable
    let start = search_exact(&db, "sparse_start", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(start.matches.len(), 1);
    assert_eq!(start.matches[0].line_number, 1);

    let middle = search_exact(&db, "sparse_middle", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(middle.matches.len(), 1);
    assert_eq!(middle.matches[0].line_number, 102);

    let end = search_exact(&db, "sparse_end", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(end.matches.len(), 1);
    assert_eq!(end.matches[0].line_number, 203);
}
