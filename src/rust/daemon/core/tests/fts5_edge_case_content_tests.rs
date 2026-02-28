//! FTS5 Edge Case Tests — Special Characters, File ID Reuse, and Multi-byte UTF-8
//!
//! Covers:
//! - Special characters in content and search patterns
//! - SQL injection safety
//! - Regex-special characters in exact search
//! - File ID reuse / overwrite behavior
//! - Emoji content
//! - Box-drawing characters
//! - CJK mixed with ASCII

use tempfile::TempDir;

use workspace_qdrant_core::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, SearchOptions};

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

// ---------------------------------------------------------------------------
// Special characters in content and search
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
        .full_rewrite(1, content, "proj1", Some("main"), "src/special.rs", None, None, None)
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
        .full_rewrite(1, content, "proj1", Some("main"), "src/injection.rs", None, None, None)
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
        .full_rewrite(1, content, "proj1", Some("main"), "src/generics.rs", None, None, None)
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
// File ID reuse / overwrite
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_file_id_reuse_after_delete() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Create file with ID 1
    processor
        .full_rewrite(1, "fn original_v1() {}", "proj1", Some("main"), "src/a.rs", None, None, None)
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
        .full_rewrite(1, "fn replacement_v2() {}", "proj1", Some("main"), "src/b.rs", None, None, None)
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
        .full_rewrite(1, "fn version_one() {}", "proj1", Some("main"), "src/file.rs", None, None, None)
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
            None,
            None,
            None,
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
// Multi-byte UTF-8 sequences
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_emoji_content() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "// \u{1F680} Launch sequence\nfn rocket_launch() {\n    println!(\"\u{1F3AF} Target acquired\");\n}";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/emoji.rs", None, None, None)
        .await
        .unwrap();

    // Search for emoji in context (trigram needs >=3 chars; emoji is 1 char / 4 bytes)
    let results = search_exact(&db, "\u{1F680} Launch", &SearchOptions::default())
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
    let content = "// \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}\n// \u{2502} box_draw \u{2502}\n// \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/boxdraw.rs", None, None, None)
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

    let content = "// \u{5909}\u{6570}\u{306E}\u{5B9A}\u{7FA9}\nlet name = \"\u{592A}\u{90CE}\";\n// \u{95A2}\u{6570}\u{306E}\u{547C}\u{3073}\u{51FA}\u{3057}\ncall_function(name);";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/cjk.rs", None, None, None)
        .await
        .unwrap();

    // Search for CJK text
    let results = search_exact(&db, "\u{5909}\u{6570}\u{306E}\u{5B9A}\u{7FA9}", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Search for ASCII mixed with CJK
    let results = search_exact(&db, "call_function", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}
