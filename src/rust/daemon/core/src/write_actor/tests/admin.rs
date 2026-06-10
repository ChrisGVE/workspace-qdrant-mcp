//! Tests for admin-related WriteActor commands:
//! RenameTenantAdmin, RebalanceIdf, UpsertRuleMirror, DeleteRuleMirror.

use crate::write_actor::commands::*;

use super::common::setup_test_db;

// ── RenameTenantAdmin tests ──────────────────────────────────────────

#[tokio::test]
async fn rename_tenant_updates_all_tables() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();

    // Insert data across tables referencing old tenant
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w-rename', '/tmp/rename', 'projects', 'old-tenant', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, payload_json, created_at, updated_at) \
         VALUES ('q-rename', 'idem-rename', 'file', 'add', 'old-tenant', 'projects', 'pending', '{}', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, file_path, tenant_id, created_at, updated_at) \
         VALUES ('w-rename', '/tmp/rename/foo.rs', 'old-tenant', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .rename_tenant_admin(RenameTenantAdminData {
            old_tenant_id: "old-tenant".into(),
            new_tenant_id: "new-tenant".into(),
        })
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.total_rows_updated >= 3);

    // Verify old tenant is gone from all tables
    let old_watch = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE tenant_id = 'old-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_watch, 0);

    let old_queue = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'old-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_queue, 0);

    let old_tracked = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM tracked_files WHERE tenant_id = 'old-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_tracked, 0);

    // Verify new tenant exists
    let new_watch = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE tenant_id = 'new-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(new_watch, 1);
}

#[tokio::test]
async fn rename_tenant_rejects_empty_ids() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .rename_tenant_admin(RenameTenantAdminData {
            old_tenant_id: "".into(),
            new_tenant_id: "new".into(),
        })
        .await;

    assert!(result.is_err());
}

// ── RebalanceIdf tests ───────────────────────────────────────────────

#[tokio::test]
async fn rebalance_idf_updates_corpus_statistics() {
    let (pool, handle) = setup_test_db().await;

    // Insert initial stats row
    sqlx::query(
        "INSERT INTO corpus_statistics (collection, last_corrected_n) VALUES ('projects', 0)",
    )
    .execute(&pool)
    .await
    .unwrap();

    let result = handle
        .rebalance_idf(RebalanceIdfData {
            collection: "projects".into(),
            last_corrected_n: 42,
        })
        .await
        .unwrap();

    assert!(result.success);

    let n = sqlx::query_scalar::<_, i64>(
        "SELECT last_corrected_n FROM corpus_statistics WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(n, 42);
}

// ── ReembedTenant tests ──────────────────────────────────────────────

async fn insert_watch_folder(pool: &sqlx::SqlitePool, tenant: &str, path: &str) {
    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, enabled, created_at, updated_at) \
         VALUES (?1, ?2, 'projects', ?3, 1, ?4, ?4)",
    )
    .bind(format!("w-{tenant}"))
    .bind(path)
    .bind(tenant)
    .bind(&now)
    .execute(pool)
    .await
    .unwrap();
}

async fn fetch_scan_payload(pool: &sqlx::SqlitePool, tenant: &str) -> String {
    sqlx::query_scalar::<_, String>(
        "SELECT payload_json FROM unified_queue \
         WHERE tenant_id = ?1 AND item_type = 'folder' AND op = 'scan'",
    )
    .bind(tenant)
    .fetch_one(pool)
    .await
    .unwrap()
}

#[tokio::test]
async fn reembed_tenant_default_is_repair_scan_without_uplift() {
    let (pool, handle) = setup_test_db().await;
    insert_watch_folder(&pool, "t-repair", "/tmp/does-not-exist-repair").await;

    let result = handle
        .reembed_tenant(ReembedTenantData {
            tenant_id: "t-repair".into(),
            force: false,
        })
        .await
        .unwrap();

    assert_eq!(result.files_enqueued, 1);
    assert!(result.message.contains("repair"), "{}", result.message);
    let payload = fetch_scan_payload(&pool, "t-repair").await;
    // The non-forced payload must stay byte-compatible with the historical
    // encoding (idempotency keys hash the payload verbatim).
    assert!(
        !payload.contains("uplift"),
        "non-forced scan payload must omit uplift: {payload}"
    );
}

#[tokio::test]
async fn reembed_tenant_force_enqueues_uplift_scans() {
    let (pool, handle) = setup_test_db().await;
    insert_watch_folder(&pool, "t-force", "/tmp/does-not-exist-force").await;

    let result = handle
        .reembed_tenant(ReembedTenantData {
            tenant_id: "t-force".into(),
            force: true,
        })
        .await
        .unwrap();

    assert_eq!(result.files_enqueued, 1);
    assert!(result.message.contains("forced"), "{}", result.message);
    let payload = fetch_scan_payload(&pool, "t-force").await;
    assert!(
        payload.contains(r#""uplift":true"#),
        "forced scan payload must carry uplift:true: {payload}"
    );
    // The flag must round-trip through FolderPayload so the folder strategy
    // sees it at dequeue time.
    let parsed: crate::unified_queue_schema::FolderPayload =
        serde_json::from_str(&payload).unwrap();
    assert!(parsed.uplift);
}

// ── UpsertRuleMirror / DeleteRuleMirror tests ────────────────────────

#[tokio::test]
async fn upsert_and_delete_rule_mirror() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();

    // Upsert a new rule
    handle
        .upsert_rule_mirror(UpsertRuleMirrorData {
            rule_id: "rule-1".into(),
            rule_text: "Always greet politely".into(),
            scope: "global".into(),
            tenant_id: "t1".into(),
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .await
        .unwrap();

    let count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM rules_mirror WHERE rule_id = 'rule-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);

    // Upsert same rule with updated text (ON CONFLICT update)
    handle
        .upsert_rule_mirror(UpsertRuleMirrorData {
            rule_id: "rule-1".into(),
            rule_text: "Updated rule text".into(),
            scope: "global".into(),
            tenant_id: "t1".into(),
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .await
        .unwrap();

    let text = sqlx::query_scalar::<_, String>(
        "SELECT rule_text FROM rules_mirror WHERE rule_id = 'rule-1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(text, "Updated rule text");

    // Delete
    handle
        .delete_rule_mirror(DeleteRuleMirrorData {
            rule_id: "rule-1".into(),
        })
        .await
        .unwrap();

    let count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM rules_mirror WHERE rule_id = 'rule-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 0);
}
