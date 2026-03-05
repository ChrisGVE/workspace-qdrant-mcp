//! Checkpoint rollback operation tests

use super::super::*;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_rollback_delete_file() {
    let dir = std::env::temp_dir().join("test_rollback_delete");
    let _ = std::fs::create_dir_all(&dir);
    let file_path = dir.join("to_delete.txt");
    std::fs::write(&file_path, "temporary data").unwrap();

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm
        .create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::DeleteFile {
                path: file_path.clone(),
            }],
        )
        .await
        .unwrap();

    cm.rollback_checkpoint(&ckpt_id).await.unwrap();
    assert!(
        !file_path.exists(),
        "File should have been deleted by rollback"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_restore_file() {
    let dir = std::env::temp_dir().join("test_rollback_restore");
    let _ = std::fs::create_dir_all(&dir);

    let original = dir.join("original.txt");
    let backup = dir.join("backup.txt");
    std::fs::write(&original, "modified").unwrap();
    std::fs::write(&backup, "original content").unwrap();

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm
        .create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::RestoreFile {
                original_path: original.clone(),
                backup_path: backup.clone(),
            }],
        )
        .await
        .unwrap();

    cm.rollback_checkpoint(&ckpt_id).await.unwrap();
    let content = std::fs::read_to_string(&original).unwrap();
    assert_eq!(content, "original content");
    assert!(!backup.exists(), "Backup should be cleaned up");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_remove_from_collection_no_storage_client() {
    let dir = std::env::temp_dir().join("test_rollback_remove_no_sc");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm
        .create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::RemoveFromCollection {
                document_id: "doc-123".into(),
                collection: "projects".into(),
            }],
        )
        .await
        .unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(
        result.is_ok(),
        "rollback_checkpoint should succeed even if individual actions fail"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_revert_index_no_storage_client() {
    let dir = std::env::temp_dir().join("test_rollback_revert_no_sc");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let snapshot = serde_json::json!({
        "collection": "projects",
        "indexes": ["field1", "field2"]
    });

    let ckpt_id = cm
        .create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::RevertIndexChanges {
                index_snapshot: snapshot,
            }],
        )
        .await
        .unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_custom_handler_registered() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = std::env::temp_dir().join("test_rollback_custom");
    let _ = std::fs::create_dir_all(&dir);

    let executed = Arc::new(AtomicBool::new(false));
    let executed_clone = executed.clone();

    struct TestHandler {
        executed: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl CustomRollbackHandler for TestHandler {
        async fn execute(&self, _data: &serde_json::Value) -> Result<(), String> {
            self.executed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    cm.register_custom_handler(
        "test_action",
        Arc::new(TestHandler {
            executed: executed_clone,
        }),
    )
    .await;

    let ckpt_id = cm
        .create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::Custom {
                action_type: "test_action".into(),
                data: serde_json::json!({"key": "value"}),
            }],
        )
        .await
        .unwrap();

    cm.rollback_checkpoint(&ckpt_id).await.unwrap();
    assert!(
        executed.load(Ordering::SeqCst),
        "Custom handler should have been executed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_custom_handler_not_registered() {
    let dir = std::env::temp_dir().join("test_rollback_custom_unreg");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm
        .create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::Custom {
                action_type: "unregistered_action".into(),
                data: serde_json::json!({}),
            }],
        )
        .await
        .unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(
        result.is_ok(),
        "rollback_checkpoint succeeds even with failed actions"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_multiple_actions_continue_on_failure() {
    let dir = std::env::temp_dir().join("test_rollback_multi");
    let _ = std::fs::create_dir_all(&dir);

    let file_to_delete = dir.join("should_be_deleted.txt");
    std::fs::write(&file_to_delete, "data").unwrap();

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm
        .create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![
                RollbackAction::RemoveFromCollection {
                    document_id: "doc-456".into(),
                    collection: "projects".into(),
                },
                RollbackAction::DeleteFile {
                    path: file_to_delete.clone(),
                },
            ],
        )
        .await
        .unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(result.is_ok());
    assert!(
        !file_to_delete.exists(),
        "DeleteFile should execute even when other actions fail"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_storage_configuration() {
    let dir = std::env::temp_dir().join("test_rollback_storage_cfg");
    let _ = std::fs::create_dir_all(&dir);

    let pipeline = Pipeline::new(2);
    let cm = pipeline.checkpoint_manager();
    assert!(cm.storage_client.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_custom_handler_registry() {
    struct NoopHandler;

    #[async_trait::async_trait]
    impl CustomRollbackHandler for NoopHandler {
        async fn execute(&self, _data: &serde_json::Value) -> Result<(), String> {
            Ok(())
        }
    }

    let dir = std::env::temp_dir().join("test_custom_registry");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));

    {
        let handlers = cm.custom_handlers.read().await;
        assert!(handlers.is_empty());
    }

    cm.register_custom_handler("noop", Arc::new(NoopHandler))
        .await;
    {
        let handlers = cm.custom_handlers.read().await;
        assert_eq!(handlers.len(), 1);
        assert!(handlers.contains_key("noop"));
    }

    cm.register_custom_handler("another", Arc::new(NoopHandler))
        .await;
    {
        let handlers = cm.custom_handlers.read().await;
        assert_eq!(handlers.len(), 2);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
