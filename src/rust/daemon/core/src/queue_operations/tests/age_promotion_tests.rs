//! Age-based promotion tests for the fairness dequeue.
//!
//! The dequeue ORDER BY promotes items that have been pending a long time so a
//! tenant whose queued items have low op-weight (e.g. folder/scan) cannot be
//! starved indefinitely by other tenants with larger, higher-weight queues. An
//! item past the warning threshold gets +1 and past the critical threshold +2,
//! applied BEFORE the project-active ranking — so an aged, inactive, low-weight
//! item can outrank a fresh, active, high-weight one.

use super::*;

/// Insert a pending queue row with an explicit op and created_at so the test
/// can control item age (the age CASE compares against `strftime('%s','now')`).
async fn insert_pending(
    pool: &SqlitePool,
    queue_id: &str,
    tenant_id: &str,
    op: &str,
    created_at: &str,
) {
    sqlx::query(
        r#"INSERT INTO unified_queue
           (queue_id, item_type, op, tenant_id, collection, status,
            branch, idempotency_key, payload_json, created_at, updated_at)
           VALUES (?1, 'file', ?2, ?3, 'projects', 'pending',
            'main', ?4, ?5, ?6, ?6)"#,
    )
    .bind(queue_id)
    .bind(op)
    .bind(tenant_id)
    .bind(format!("key-{}", queue_id))
    .bind(format!(r#"{{"file_path":"/test/{}.rs"}}"#, queue_id))
    .bind(created_at)
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, tenant_id: &str, is_active: i64) {
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES (?1, ?2, 'projects', ?3, ?4,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    )
    .bind(watch_id)
    .bind(format!("/test/{}", watch_id))
    .bind(tenant_id)
    .bind(is_active)
    .execute(pool)
    .await
    .unwrap();
}

/// An item aged past the critical threshold outranks a fresh item even though
/// the fresh one belongs to an active project and has a higher op-weight — this
/// is the whole point of age promotion (anti-starvation).
#[tokio::test]
async fn test_aged_low_weight_item_outranks_fresh_active_item() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_age_promote.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Active project (fresh, high-weight add) vs inactive project (ancient,
    // low-weight scan). Ancient timestamp => age >> critical threshold => +2.
    insert_watch_folder(&pool, "w-active", "tenant-active", 1).await;
    insert_watch_folder(&pool, "w-inactive", "tenant-inactive", 0).await;
    insert_pending(
        &pool,
        "fresh-add",
        "tenant-active",
        "add",
        "2099-01-01T00:00:00.000Z",
    )
    .await;
    insert_pending(
        &pool,
        "aged-scan",
        "tenant-inactive",
        "scan",
        "2000-01-01T00:00:00.000Z",
    )
    .await;

    // DESC pass with default thresholds (warning=300, critical=900).
    let items = manager
        .dequeue_unified(2, "worker-1", Some(300), None, None, Some(true), None, None)
        .await
        .unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0].queue_id, "aged-scan",
        "an item past the critical age threshold must outrank a fresh active add"
    );
    assert_eq!(items[1].queue_id, "fresh-add");
}

/// When neither item has crossed a threshold, age promotion stays neutral and
/// the normal active-vs-inactive ranking decides: the fresh active add wins.
#[tokio::test]
async fn test_no_promotion_below_threshold() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_age_noprom.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    insert_watch_folder(&pool, "w-active", "tenant-active", 1).await;
    insert_watch_folder(&pool, "w-inactive", "tenant-inactive", 0).await;
    // Both fresh (created "now"): neither crosses the age thresholds, so the
    // age CASE is 0 for both and the active project ranks first.
    let now = wqm_common::timestamps::now_utc();
    insert_pending(&pool, "fresh-add", "tenant-active", "add", &now).await;
    insert_pending(&pool, "fresh-scan", "tenant-inactive", "scan", &now).await;

    let items = manager
        .dequeue_unified(2, "worker-1", Some(300), None, None, Some(true), None, None)
        .await
        .unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0].queue_id, "fresh-add",
        "with no item aged past threshold, the active project add ranks first"
    );
    assert_eq!(items[1].queue_id, "fresh-scan");
}

/// Regression guard: an aged item must still be promoted on the ASC
/// anti-starvation pass (priority_descending = false), NOT buried. The age CASE
/// is always DESC, so aging dominates regardless of the fairness flip — burying
/// aged items on the very pass meant to rescue stragglers would defeat the
/// feature.
#[tokio::test]
async fn test_aged_item_promoted_on_asc_pass_too() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_age_asc.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    insert_watch_folder(&pool, "w-active", "tenant-active", 1).await;
    insert_watch_folder(&pool, "w-inactive", "tenant-inactive", 0).await;
    insert_pending(
        &pool,
        "fresh-add",
        "tenant-active",
        "add",
        "2099-01-01T00:00:00.000Z",
    )
    .await;
    insert_pending(
        &pool,
        "aged-scan",
        "tenant-inactive",
        "scan",
        "2000-01-01T00:00:00.000Z",
    )
    .await;

    // ASC anti-starvation pass (priority_descending = false).
    let items = manager
        .dequeue_unified(
            2,
            "worker-1",
            Some(300),
            None,
            None,
            Some(false),
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0].queue_id, "aged-scan",
        "the aged item must still be promoted first on the ASC pass, not buried"
    );
    assert_eq!(items[1].queue_id, "fresh-add");
}
