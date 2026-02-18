//! FTS5 Edge Case Tests (Task 62)
//!
//! Tests for edge cases and robustness in the FTS5 search pipeline:
//! - CRLF line endings
//! - Very long lines (>10KB)
//! - Whitespace-only files
//! - Trailing newlines vs no trailing newlines
//! - Sparse content (many blank lines)
//! - Special characters in content and search patterns
//! - File ID reuse / overwrite behavior
//! - Mixed encodings (valid UTF-8 with multi-byte sequences)
//! - Concurrent update + search for same file
//! - Diff with CRLF → LF normalization

use std::sync::Arc;
use tempfile::TempDir;

use workspace_qdrant_core::fts_batch_processor::{
    FileChange, FtsBatchConfig, FtsBatchProcessor,
};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, search_regex, SearchOptions};

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

// ---------------------------------------------------------------------------
// 1. CRLF line endings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_crlf_content_indexable_and_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Content with CRLF line endings (Windows-style)
    let content = "fn main() {\r\n    println!(\"hello\");\r\n}\r\n";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/main.rs")
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
        .full_rewrite(1, content, "proj1", Some("main"), "src/mixed.rs")
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
// 2. Very long lines
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
        .full_rewrite(1, &content, "proj1", Some("main"), "src/long.rs")
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
        .full_rewrite(1, &content, "proj1", Some("main"), "src/multilong.rs")
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
// 3. Whitespace-only files
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_whitespace_only_file() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // File with only spaces, tabs, and newlines
    let content = "   \n\t\t\n  \n   \t   \n";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/whitespace.rs")
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
        .full_rewrite(1, " ", "proj1", Some("main"), "src/space.rs")
        .await
        .unwrap();

    // Should not crash, file should be indexed
    let results = search_exact(&db, "anything", &SearchOptions::default())
        .await
        .unwrap();
    assert!(results.matches.is_empty());
}

// ---------------------------------------------------------------------------
// 4. Trailing newline behavior
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_trailing_newline_vs_no_trailing() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Without trailing newline: split('\n') gives ["line1", "line2"]
    let no_trailing = "trailing_test_a\ntrailing_test_b";
    processor
        .full_rewrite(1, no_trailing, "proj1", Some("main"), "src/no_trail.rs")
        .await
        .unwrap();

    // With trailing newline: split('\n') gives ["line1", "line2", ""]
    let with_trailing = "trailing_test_c\ntrailing_test_d\n";
    processor
        .full_rewrite(2, with_trailing, "proj1", Some("main"), "src/with_trail.rs")
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
// 5. Sparse content (many blank lines)
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
        .full_rewrite(1, &content, "proj1", Some("main"), "src/sparse.rs")
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

// ---------------------------------------------------------------------------
// 6. Special characters in content and search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_special_characters_in_content() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Content with various special characters
    let content = r#"let re = Regex::new(r"(\d+)\.(\d+)");
let path = "C:\Users\test";
let sql = "SELECT * FROM users WHERE name = 'O''Brien'";
let html = "<div class=\"container\">&amp;</div>";
let special = "tabs	and	tabs";
"#;

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/special.rs")
        .await
        .unwrap();

    // Search for various patterns
    let results = search_exact(&db, "Regex::new", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    let results = search_exact(&db, "SELECT * FROM", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    let results = search_exact(&db, "&amp;", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_quotes_and_sql_injection_safe() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Content that might cause SQL injection if not parameterized
    let content = "let x = \"'; DROP TABLE code_lines; --\";\nlet y = normal_code();";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/injection.rs")
        .await
        .unwrap();

    // Should store the content without SQL injection
    let results = search_exact(&db, "normal_code", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1, "File should be indexed without corruption");

    // Search for the injection string should find it as content
    let results = search_exact(&db, "DROP TABLE", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1, "SQL-like content should be searchable");
}

#[tokio::test]
async fn test_regex_special_chars_in_exact_search() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "fn compute(x: Vec<i32>) -> Result<(), Box<dyn Error>> {}";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/generics.rs")
        .await
        .unwrap();

    // Exact search with regex-special characters should work
    let results = search_exact(&db, "Vec<i32>", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    let results = search_exact(&db, "Box<dyn Error>", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

// ---------------------------------------------------------------------------
// 7. File ID reuse / overwrite
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_file_id_reuse_after_delete() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Create file with ID 1
    processor
        .full_rewrite(1, "fn original_v1() {}", "proj1", Some("main"), "src/a.rs")
        .await
        .unwrap();

    let r1 = search_exact(&db, "original_v1", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(r1.matches.len(), 1);

    // Delete file ID 1
    processor.delete_file(1).await.unwrap();

    // Reuse file ID 1 for a different file
    processor
        .full_rewrite(1, "fn replacement_v2() {}", "proj1", Some("main"), "src/b.rs")
        .await
        .unwrap();

    // Old content should be gone
    let old = search_exact(&db, "original_v1", &SearchOptions::default())
        .await
        .unwrap();
    assert!(old.matches.is_empty(), "Old content should be deleted");

    // New content should be present
    let new = search_exact(&db, "replacement_v2", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(new.matches.len(), 1, "New content should be found");
    assert_eq!(new.matches[0].file_path, "src/b.rs");
}

#[tokio::test]
async fn test_overwrite_same_file_id_without_delete() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Create file with ID 1
    processor
        .full_rewrite(1, "fn version_one() {}", "proj1", Some("main"), "src/file.rs")
        .await
        .unwrap();

    // Overwrite with full_rewrite (same ID, different content)
    processor
        .full_rewrite(
            1,
            "fn version_two() {}",
            "proj1",
            Some("main"),
            "src/file.rs",
        )
        .await
        .unwrap();

    // Only v2 should be found
    let v1 = search_exact(&db, "version_one", &SearchOptions::default())
        .await
        .unwrap();
    assert!(v1.matches.is_empty(), "version_one should be gone after overwrite");

    let v2 = search_exact(&db, "version_two", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(v2.matches.len(), 1, "version_two should be found");
}

// ---------------------------------------------------------------------------
// 8. Multi-byte UTF-8 sequences
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_emoji_content() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "// 🚀 Launch sequence\nfn rocket_launch() {\n    println!(\"🎯 Target acquired\");\n}";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/emoji.rs")
        .await
        .unwrap();

    // Search for emoji in context (trigram needs ≥3 chars; emoji is 1 char / 4 bytes)
    let results = search_exact(&db, "🚀 Launch", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Search for regular text
    let results = search_exact(&db, "rocket_launch", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_box_drawing_characters() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Box drawing characters (previously caused panics from byte-offset slicing)
    let content = "// ┌──────────┐\n// │ box_draw │\n// └──────────┘";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/boxdraw.rs")
        .await
        .unwrap();

    let results = search_exact(&db, "box_draw", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_cjk_mixed_with_ascii() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "// 変数の定義\nlet name = \"太郎\";\n// 関数の呼び出し\ncall_function(name);";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/cjk.rs")
        .await
        .unwrap();

    // Search for CJK text
    let results = search_exact(&db, "変数の定義", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Search for ASCII mixed with CJK
    let results = search_exact(&db, "call_function", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

// ---------------------------------------------------------------------------
// 9. Concurrent update + search on same file
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_concurrent_update_and_search_same_file() {
    let (_tmp, db) = setup_db().await;
    let db = Arc::new(db);

    // Initial content
    {
        let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
        processor
            .full_rewrite(
                1,
                "fn stable_function() {}",
                "proj1",
                Some("main"),
                "src/concurrent.rs",
            )
            .await
            .unwrap();
    }

    // Spawn search tasks while updating
    let mut handles = Vec::new();

    // 5 concurrent searches
    for _ in 0..5 {
        let db_clone = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            // The function should be found at some point during the test
            let results = search_exact(
                &db_clone,
                "stable_function",
                &SearchOptions::default(),
            )
            .await;
            assert!(results.is_ok(), "Search should not error during concurrent access");
        }));
    }

    // Concurrent update
    {
        let db_clone = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let processor = FtsBatchProcessor::new(&db_clone, FtsBatchConfig::default());
            let result = processor
                .full_rewrite(
                    1,
                    "fn updated_function() {}",
                    "proj1",
                    Some("main"),
                    "src/concurrent.rs",
                )
                .await;
            assert!(result.is_ok(), "Update should not error during concurrent access");
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

// ---------------------------------------------------------------------------
// 10. Diff edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_diff_completely_different_content() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let v1 = "fn alpha() {\n    one();\n    two();\n    three();\n}";
    let v2 = "struct Beta {\n    x: i32,\n    y: f64,\n}";

    // Ingest v1
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: v1.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // Replace with completely different v2
    processor.add_change(FileChange {
        file_id: 1,
        old_content: v1.to_string(),
        new_content: v2.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // v1 content fully gone
    for term in &["alpha", "one()", "two()", "three()"] {
        let results = search_exact(&db, term, &SearchOptions::default())
            .await
            .unwrap();
        assert!(results.matches.is_empty(), "{} from v1 should be gone", term);
    }

    // v2 content present
    let results = search_exact(&db, "struct Beta", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_diff_content_to_empty() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let v1 = "fn nonempty() {\n    code();\n}";

    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: v1.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // "Update" to empty content
    processor.add_change(FileChange {
        file_id: 1,
        old_content: v1.to_string(),
        new_content: String::new(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    let results = search_exact(&db, "nonempty", &SearchOptions::default())
        .await
        .unwrap();
    assert!(results.matches.is_empty(), "Content should be gone after emptying file");
}

#[tokio::test]
async fn test_diff_empty_to_content() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Start with empty file
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: String::new(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // Update empty to content
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: "fn appeared() {}".to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    let results = search_exact(&db, "appeared", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1, "Content should appear after empty→content update");
}

// ---------------------------------------------------------------------------
// 11. Regex edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_regex_with_special_quantifiers() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "let aaa = 1;\nlet aaab = 2;\nlet bbb = 3;\nlet abc = 4;";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/regex.rs")
        .await
        .unwrap();

    // Regex: a+ followed by b
    let results = search_regex(&db, "a+b", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 2, "Should match aaab and abc");

    // Regex: word boundary pattern (Rust regex doesn't support backreferences)
    // Matches: aaa, bbb, abc (all are 3 lowercase letters)
    let results = search_regex(&db, "let [a-z]{3} =", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(
        results.matches.len(),
        3,
        "Should find 'let aaa =', 'let bbb =', and 'let abc ='"
    );
}

#[tokio::test]
async fn test_case_insensitive_search() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "fn MyFunction() {}\nfn myfunction() {}\nfn MYFUNCTION() {}";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/case.rs")
        .await
        .unwrap();

    // Case-sensitive (default): only one match
    let results = search_exact(
        &db,
        "MyFunction",
        &SearchOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1, "Case-sensitive should find exactly one");

    // Case-insensitive: all three
    let results = search_exact(
        &db,
        "myfunction",
        &SearchOptions {
            case_insensitive: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        results.matches.len(),
        3,
        "Case-insensitive should find all three variants"
    );
}

// ---------------------------------------------------------------------------
// 12. Batch with heterogeneous edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_batch_with_mixed_edge_cases() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let long_line = "x".repeat(15_000);
    let files: Vec<(i64, &str, &str)> = vec![
        (1, "", "src/empty.rs"),                         // empty
        (2, "   \n\n\t\n", "src/whitespace.rs"),         // whitespace only
        (3, "fn normal_code() {}", "src/normal.rs"),     // normal
        (4, "🚀 emoji_file", "src/emoji.rs"),            // emoji
        (5, "line\r\nwith\r\ncrlf", "src/crlf.rs"),      // CRLF
        (6, &long_line, "src/longline.rs"),              // long single line
    ];

    for (id, content, path) in &files {
        processor.add_change(FileChange {
            file_id: *id,
            old_content: String::new(),
            new_content: content.to_string(),
            tenant_id: "batch-edge".to_string(),
            branch: Some("main".to_string()),
            file_path: path.to_string(),
        });
    }

    // Should not crash on any edge case
    let stats = processor.flush(6).await.unwrap();
    assert_eq!(stats.files_processed, 6);

    // Normal file should be searchable
    let results = search_exact(
        &db,
        "normal_code",
        &SearchOptions {
            tenant_id: Some("batch-edge".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Emoji file should be searchable
    let results = search_exact(
        &db,
        "emoji_file",
        &SearchOptions {
            tenant_id: Some("batch-edge".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
}
