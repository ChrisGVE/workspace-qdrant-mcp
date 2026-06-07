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

    // tracked_files has no tenant_id column — its tenancy is reached through
    // watch_folders, so a rename must leave the row itself untouched.
    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, relative_path, created_at, updated_at) \
         VALUES ('w-rename', 'foo.rs', ?1, ?1)",
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
    // watch_folders + unified_queue rows; tracked_files needs no update.
    assert!(result.total_rows_updated >= 2);

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

    // tracked_files rows follow their watch_folder's tenant: after the rename
    // the row must be reachable under the NEW tenant via the join.
    let tracked_under_new = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = 'new-tenant'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(tracked_under_new, 1);

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
