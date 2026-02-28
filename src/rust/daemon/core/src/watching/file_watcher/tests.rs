use super::*;
use std::path::PathBuf;

#[tokio::test]
async fn test_config_default() {
    let config = EnhancedWatcherConfig::default();
    assert_eq!(config.debounce_delay_ms, 1000);
    assert!(config.enable_file_id_cache);
}

#[tokio::test]
async fn test_watch_event_variants() {
    // Test that all event variants can be created
    let events = vec![
        WatchEvent::Created {
            path: PathBuf::from("/test"),
            is_directory: false,
        },
        WatchEvent::Modified {
            path: PathBuf::from("/test"),
            is_directory: false,
        },
        WatchEvent::Deleted {
            path: PathBuf::from("/test"),
            is_directory: false,
        },
        WatchEvent::Renamed {
            old_path: PathBuf::from("/old"),
            new_path: PathBuf::from("/new"),
            is_directory: false,
        },
        WatchEvent::CrossFilesystemMove {
            deleted_path: PathBuf::from("/deleted"),
            is_directory: true,
        },
        WatchEvent::RootRenamed {
            old_path: PathBuf::from("/old"),
            new_path: PathBuf::from("/new"),
            tenant_id: "tenant-123".to_string(),
        },
        WatchEvent::Error {
            path: Some(PathBuf::from("/error")),
            message: "test error".to_string(),
        },
    ];

    assert_eq!(events.len(), 7);
}
