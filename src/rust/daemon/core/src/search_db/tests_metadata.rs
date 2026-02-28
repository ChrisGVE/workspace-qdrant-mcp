//! Tests for file_metadata table and project/branch/path scoped FTS5 searches.

#![cfg(test)]

use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_file_metadata_table_exists() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_metadata')",
    )
    .fetch_one(manager.pool())
    .await
    .unwrap();

    assert!(exists, "file_metadata table should exist after migration v4");
    manager.close().await;
}

#[tokio::test]
async fn test_file_metadata_indexes_exist() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    for idx_name in &["idx_file_metadata_tenant", "idx_file_metadata_tenant_branch"] {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name=?1)",
        )
        .bind(*idx_name)
        .fetch_one(manager.pool())
        .await
        .unwrap();

        assert!(exists, "Index {} should exist", idx_name);
    }

    manager.close().await;
}

#[tokio::test]
async fn test_file_metadata_upsert() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("project-abc")
        .bind("main")
        .bind("/src/lib.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .execute(manager.pool())
        .await
        .unwrap();

    let row = sqlx::query("SELECT tenant_id, branch, file_path FROM file_metadata WHERE file_id = 1")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(row.get::<String, _>("tenant_id"), "project-abc");
    assert_eq!(row.get::<String, _>("branch"), "main");
    assert_eq!(row.get::<String, _>("file_path"), "/src/lib.rs");

    // Upsert (update branch)
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("project-abc")
        .bind("feature/new")
        .bind("/src/lib.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .execute(manager.pool())
        .await
        .unwrap();

    let row2 = sqlx::query("SELECT branch FROM file_metadata WHERE file_id = 1")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(row2.get::<String, _>("branch"), "feature/new");

    manager.close().await;
}

#[tokio::test]
async fn test_file_metadata_null_branch() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("project-xyz")
        .bind(None::<String>)
        .bind("/readme.md")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .execute(manager.pool())
        .await
        .unwrap();

    let row = sqlx::query("SELECT branch FROM file_metadata WHERE file_id = 1")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert!(row.get::<Option<String>, _>("branch").is_none());

    manager.close().await;
}

#[tokio::test]
async fn test_file_metadata_delete() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj")
        .bind("main")
        .bind("/a.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .execute(manager.pool())
        .await
        .unwrap();

    sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_SQL)
        .bind(1_i64)
        .execute(manager.pool())
        .await
        .unwrap();

    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE file_id = 1")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(count, 0);

    manager.close().await;
}

#[tokio::test]
async fn test_file_metadata_delete_by_tenant() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    for i in 1..=3 {
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(i as i64)
            .bind("proj-a")
            .bind("main")
            .bind(format!("/file{}.rs", i))
            .bind(None::<&str>)
            .bind(None::<&str>)
            .bind(None::<&str>)
            .execute(manager.pool())
            .await
            .unwrap();
    }
    // Different tenant
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(10_i64)
        .bind("proj-b")
        .bind("main")
        .bind("/other.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .execute(manager.pool())
        .await
        .unwrap();

    sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_BY_TENANT_SQL)
        .bind("proj-a")
        .execute(manager.pool())
        .await
        .unwrap();

    let count_a: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE tenant_id = 'proj-a'")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(count_a, 0, "All proj-a rows should be deleted");

    let count_b: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE tenant_id = 'proj-b'")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(count_b, 1, "proj-b should be untouched");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_search_by_project() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert code_lines + file_metadata for two projects
    // Project A - file 1
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn alpha() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64).bind("proj-a").bind("main").bind("/src/alpha.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    // Project B - file 2
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn alpha_beta() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64).bind("proj-b").bind("main").bind("/src/beta.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    manager.rebuild_fts().await.unwrap();

    // Search "alpha" scoped to proj-a
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_SQL)
        .bind("alpha")
        .bind("proj-a")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1, "Should find only proj-a's match");
    assert_eq!(rows[0].get::<i64, _>("file_id"), 1);
    assert_eq!(rows[0].get::<String, _>("tenant_id"), "proj-a");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_search_by_project_branch() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Same project, different branches
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn feature_code() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64).bind("proj-a").bind("main").bind("/src/main.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn feature_code_v2() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64).bind("proj-a").bind("feature/v2").bind("/src/main.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    manager.rebuild_fts().await.unwrap();

    // Search "feature_code" on branch "feature/v2"
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_BRANCH_SQL)
        .bind("feature_code")
        .bind("proj-a")
        .bind("feature/v2")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<String, _>("branch"), "feature/v2");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_search_by_path_prefix() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn handler() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64).bind("proj").bind("main").bind("/src/api/handler.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn handler_test() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64).bind("proj").bind("main").bind("/tests/api_test.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    manager.rebuild_fts().await.unwrap();

    // Search "handler" under /src/
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PATH_PREFIX_SQL)
        .bind("handler")
        .bind("/src/%")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1, "Should find only the /src/ file");
    assert_eq!(rows[0].get::<String, _>("file_path"), "/src/api/handler.rs");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_search_by_project_path() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // proj-a, /src/
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn widget() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64).bind("proj-a").bind("main").bind("/src/widget.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    // proj-b, /src/
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn widget_v2() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64).bind("proj-b").bind("main").bind("/src/widget.rs")
        .bind(None::<&str>).bind(None::<&str>).bind(None::<&str>)
        .execute(manager.pool()).await.unwrap();

    manager.rebuild_fts().await.unwrap();

    // Search "widget" scoped to proj-a + /src/
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_PATH_SQL)
        .bind("widget")
        .bind("proj-a")
        .bind("/src/%")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<String, _>("tenant_id"), "proj-a");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_scoped_search_performance_1000_files() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert 1000 files across 10 projects, each with 10 code lines
    let mut tx = manager.pool().begin().await.unwrap();
    for file_idx in 0..1000_i64 {
        let tenant = format!("proj-{}", file_idx % 10);
        let file_path = format!("/src/module{}/file{}.rs", file_idx / 100, file_idx);

        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(file_idx)
            .bind(&tenant)
            .bind("main")
            .bind(&file_path)
            .bind(None::<&str>)
            .bind(None::<&str>)
            .bind(None::<&str>)
            .execute(&mut *tx)
            .await
            .unwrap();

        for line_idx in 0..10 {
            let seq = crate::code_lines_schema::initial_seq(line_idx);
            let content = format!("fn process_item_{}() {{ /* file {} line {} */ }}", file_idx, file_idx, line_idx);
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
                .bind(file_idx)
                .bind(seq)
                .bind(&content)
                .execute(&mut *tx)
                .await
                .unwrap();
        }
    }
    tx.commit().await.unwrap();
    manager.rebuild_fts().await.unwrap();

    // Scoped search: "process_item" within proj-0 (100 files)
    let start = std::time::Instant::now();
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_SQL)
        .bind("process_item")
        .bind("proj-0")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    let elapsed = start.elapsed();

    // Should return ~1000 lines (100 files * 10 lines each)
    assert_eq!(rows.len(), 1000, "Should find 1000 lines for proj-0");
    assert!(elapsed.as_millis() < 5000, "Scoped search should complete in <5s, took {}ms", elapsed.as_millis());

    // Verify all results are from proj-0
    for row in &rows {
        assert_eq!(row.get::<String, _>("tenant_id"), "proj-0");
    }

    manager.close().await;
}
