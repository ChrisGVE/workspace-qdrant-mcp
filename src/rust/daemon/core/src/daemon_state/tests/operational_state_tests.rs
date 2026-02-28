//! Operational state set/get, upsert, component isolation, project scope,
//! and missing key tests.

use super::*;

#[tokio::test]
async fn test_operational_state_set_get_roundtrip() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_test.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Set a global value
    set_operational_state(pool, "last_log_prune", "daemon", "2026-02-17T00:00:00Z", None)
        .await.unwrap();

    // Get it back
    let val = get_operational_state(pool, "last_log_prune", "daemon", None)
        .await.unwrap();
    assert_eq!(val, Some("2026-02-17T00:00:00Z".to_string()));
}

#[tokio::test]
async fn test_operational_state_upsert() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_upsert.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Set initial value
    set_operational_state(pool, "key1", "daemon", "value1", None)
        .await.unwrap();

    // Overwrite with upsert
    set_operational_state(pool, "key1", "daemon", "value2", None)
        .await.unwrap();

    let val = get_operational_state(pool, "key1", "daemon", None)
        .await.unwrap();
    assert_eq!(val, Some("value2".to_string()));
}

#[tokio::test]
async fn test_operational_state_component_isolation() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_isolation.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Same key, different components
    set_operational_state(pool, "version", "daemon", "1.0", None).await.unwrap();
    set_operational_state(pool, "version", "server", "2.0", None).await.unwrap();
    set_operational_state(pool, "version", "cli", "3.0", None).await.unwrap();

    assert_eq!(
        get_operational_state(pool, "version", "daemon", None).await.unwrap(),
        Some("1.0".to_string())
    );
    assert_eq!(
        get_operational_state(pool, "version", "server", None).await.unwrap(),
        Some("2.0".to_string())
    );
    assert_eq!(
        get_operational_state(pool, "version", "cli", None).await.unwrap(),
        Some("3.0".to_string())
    );
}

#[tokio::test]
async fn test_operational_state_project_scoped() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_project.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Global entry
    set_operational_state(pool, "status", "daemon", "running", None).await.unwrap();
    // Project-scoped entry
    set_operational_state(pool, "status", "daemon", "indexing", Some("proj-1")).await.unwrap();

    // Global lookup should NOT return project-scoped value
    let global = get_operational_state(pool, "status", "daemon", None).await.unwrap();
    assert_eq!(global, Some("running".to_string()));

    // Project lookup should return project-scoped value
    let proj = get_operational_state(pool, "status", "daemon", Some("proj-1")).await.unwrap();
    assert_eq!(proj, Some("indexing".to_string()));

    // Unknown project returns None
    let unknown = get_operational_state(pool, "status", "daemon", Some("proj-999")).await.unwrap();
    assert_eq!(unknown, None);
}

#[tokio::test]
async fn test_operational_state_missing_key() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_missing.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    let val = get_operational_state(pool, "nonexistent", "daemon", None).await.unwrap();
    assert_eq!(val, None);
}
