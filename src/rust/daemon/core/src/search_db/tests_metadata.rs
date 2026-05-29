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

    assert!(
        exists,
        "file_metadata table should exist after migration v4"
    );
    manager.close().await;
}

#[tokio::test]
async fn test_file_metadata_indexes_exist() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    for idx_name in &[
        "idx_file_metadata_tenant",
        "idx_file_metadata_tenant_branch",
    ] {
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
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

    let row =
        sqlx::query("SELECT tenant_id, branch, file_path FROM file_metadata WHERE file_id = 1")
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
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
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
async fn test_file_metadata_churn_count() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // First index: reindex_count seeds to 1, first_indexed_at is stamped.
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj-churn")
        .bind("main")
        .bind("/gen/output.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>)
        .bind(0_i64)
        .execute(manager.pool())
        .await
        .unwrap();

    let row =
        sqlx::query("SELECT reindex_count, first_indexed_at FROM file_metadata WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
    assert_eq!(row.get::<i64, _>("reindex_count"), 1);
    let first_at = row.get::<Option<String>, _>("first_indexed_at");
    assert!(
        first_at.is_some(),
        "first_indexed_at should be stamped on insert"
    );

    // Re-index the same file_id: count increments, first_indexed_at preserved.
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj-churn")
        .bind("main")
        .bind("/gen/output.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>)
        .bind(0_i64)
        .execute(manager.pool())
        .await
        .unwrap();

    let row2 =
        sqlx::query("SELECT reindex_count, first_indexed_at FROM file_metadata WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
    assert_eq!(row2.get::<i64, _>("reindex_count"), 2);
    assert_eq!(
        row2.get::<Option<String>, _>("first_indexed_at"),
        first_at,
        "first_indexed_at must be preserved across re-index"
    );

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
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
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
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
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
            .bind(None::<i64>) // size_bytes (search.db v7)
            .bind(0_i64) // fts5_skipped (search.db v8)
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
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

    sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_BY_TENANT_SQL)
        .bind("proj-a")
        .execute(manager.pool())
        .await
        .unwrap();

    let count_a: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE tenant_id = 'proj-a'")
            .fetch_one(manager.pool())
            .await
            .unwrap();
    assert_eq!(count_a, 0, "All proj-a rows should be deleted");

    let count_b: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE tenant_id = 'proj-b'")
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
    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn alpha() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj-a")
        .bind("main")
        .bind("/src/alpha.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

    // Project B - file 2
    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn alpha_beta() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64)
        .bind("proj-b")
        .bind("main")
        .bind("/src/beta.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

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
    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn feature_code() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj-a")
        .bind("main")
        .bind("/src/main.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn feature_code_v2() {}')")
        .execute(manager.pool()).await.unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64)
        .bind("proj-a")
        .bind("feature/v2")
        .bind("/src/main.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

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

    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn handler() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj")
        .bind("main")
        .bind("/src/api/handler.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn handler_test() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64)
        .bind("proj")
        .bind("main")
        .bind("/tests/api_test.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

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
    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn widget() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj-a")
        .bind("main")
        .bind("/src/widget.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

    // proj-b, /src/
    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn widget_v2() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(2_i64)
        .bind("proj-b")
        .bind("main")
        .bind("/src/widget.rs")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<i64>) // size_bytes (search.db v7)
        .bind(0_i64) // fts5_skipped (search.db v8)
        .execute(manager.pool())
        .await
        .unwrap();

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
            .bind(None::<i64>) // size_bytes (search.db v7)
            .bind(0_i64) // fts5_skipped (search.db v8)
            .execute(&mut *tx)
            .await
            .unwrap();

        for line_idx in 0..10 {
            let seq = crate::code_lines_schema::initial_seq(line_idx);
            let content = format!(
                "fn process_item_{}() {{ /* file {} line {} */ }}",
                file_idx, file_idx, line_idx
            );
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
    assert!(
        elapsed.as_millis() < 5000,
        "Scoped search should complete in <5s, took {}ms",
        elapsed.as_millis()
    );

    // Verify all results are from proj-0
    for row in &rows {
        assert_eq!(row.get::<String, _>("tenant_id"), "proj-0");
    }

    manager.close().await;
}

/// Regression test for the Prometheus exporter's aggregation query
/// (`file_metadata_stats_by_tenant_branch`). The exporter reduces 10k+ rows
/// in `file_metadata` to a handful of (tenant_id, branch) gauge series — if
/// the GROUP BY ever forgets a column or the CASE expression misclassifies
/// `fts5_skipped`, dashboards lie silently. Test asserts the math on a
/// minimal hand-built fixture.
#[tokio::test]
async fn test_file_metadata_stats_by_tenant_branch() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Fixture: 2 tenants × 2 branches with mixed sizes + skipped flags.
    //   proj-a / main      : 2 files, 100+200 = 300 bytes, 0 skipped
    //   proj-a / main      : +1 skipped file at 50000 bytes → 3 files, 50300 bytes, 1 skipped
    //   proj-a / feature/x : 1 file, 75 bytes, 0 skipped
    //   proj-b / main      : 1 file with NULL size_bytes, 0 skipped
    let fixture: &[(i64, &str, Option<&str>, &str, Option<i64>, i64)] = &[
        (1, "proj-a", Some("main"), "/a.rs", Some(100), 0),
        (2, "proj-a", Some("main"), "/b.rs", Some(200), 0),
        (3, "proj-a", Some("main"), "/big.csv", Some(50_000), 1),
        (4, "proj-a", Some("feature/x"), "/c.rs", Some(75), 0),
        (5, "proj-b", Some("main"), "/d.rs", None, 0),
    ];
    for (file_id, tenant, branch, path, size, skipped) in fixture {
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(*file_id)
            .bind(*tenant)
            .bind(*branch)
            .bind(*path)
            .bind(None::<&str>) // base_point
            .bind(None::<&str>) // relative_path
            .bind(None::<&str>) // file_hash
            .bind(*size)
            .bind(*skipped)
            .execute(manager.pool())
            .await
            .unwrap();
    }

    let mut stats = manager
        .file_metadata_stats_by_tenant_branch()
        .await
        .unwrap();
    // Stable order for assertion regardless of how SQLite returns rows.
    stats.sort_by(|a, b| {
        a.tenant_id
            .cmp(&b.tenant_id)
            .then_with(|| a.branch.cmp(&b.branch))
    });

    assert_eq!(stats.len(), 3, "expected 3 (tenant, branch) groups");

    // proj-a / feature/x
    assert_eq!(stats[0].tenant_id, "proj-a");
    assert_eq!(stats[0].branch, "feature/x");
    assert_eq!(stats[0].file_count, 1);
    assert_eq!(stats[0].total_bytes, 75);
    assert_eq!(stats[0].skipped_count, 0);

    // proj-a / main — includes the big.csv with fts5_skipped=1
    assert_eq!(stats[1].tenant_id, "proj-a");
    assert_eq!(stats[1].branch, "main");
    assert_eq!(stats[1].file_count, 3);
    assert_eq!(stats[1].total_bytes, 50_300);
    assert_eq!(
        stats[1].skipped_count, 1,
        "fts5_skipped=1 should be counted in the skipped bucket; missing here means \
         the CASE expression dropped it or the GROUP BY scope is wrong",
    );

    // proj-b / main — NULL size_bytes must sum as 0, not propagate NULL
    assert_eq!(stats[2].tenant_id, "proj-b");
    assert_eq!(stats[2].branch, "main");
    assert_eq!(stats[2].file_count, 1);
    assert_eq!(
        stats[2].total_bytes, 0,
        "COALESCE(SUM(NULL), 0) must yield 0, not NULL",
    );
    assert_eq!(stats[2].skipped_count, 0);

    manager.close().await;
}

#[tokio::test]
async fn test_file_metadata_stats_normalizes_null_branch_to_none_literal() {
    // The exporter pushes `(tenant_id, branch)` as Prometheus labels and
    // empty label values are awkward to filter on. NULL branch must become
    // the literal string "(none)" so the gauge stays queryable.
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(1_i64)
        .bind("proj")
        .bind(None::<&str>) // branch = NULL
        .bind("/orphan.md")
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(None::<&str>)
        .bind(Some(42_i64))
        .bind(0_i64)
        .execute(manager.pool())
        .await
        .unwrap();

    let stats = manager
        .file_metadata_stats_by_tenant_branch()
        .await
        .unwrap();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].branch, "(none)");
    assert_eq!(stats[0].file_count, 1);
    assert_eq!(stats[0].total_bytes, 42);

    manager.close().await;
}
