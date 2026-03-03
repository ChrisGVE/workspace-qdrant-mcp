//! Unit tests for move detection and rename correlation.

use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;

use super::correlator::MoveCorrelator;
use super::types::{MoveCorrelatorConfig, RenameAction};

#[test]
fn test_simple_rename_same_directory() {
    let mut correlator = MoveCorrelator::new();

    let old_path = PathBuf::from("/project/old_name.txt");
    let new_path = PathBuf::from("/project/new_name.txt");

    // Simulate MOVED_FROM
    let result = correlator.handle_moved_from(old_path.clone(), false, Some(12345));
    assert_eq!(result, RenameAction::Pending);

    // Simulate MOVED_TO with same file ID
    let result = correlator.handle_moved_to(new_path.clone(), false, Some(12345));

    assert!(matches!(result, RenameAction::SimpleRename { .. }));
    if let RenameAction::SimpleRename { old_path: op, new_path: np, is_directory } = result {
        assert_eq!(op, old_path);
        assert_eq!(np, new_path);
        assert!(!is_directory);
    }
}

#[test]
fn test_intra_filesystem_move() {
    let mut correlator = MoveCorrelator::new();

    let old_path = PathBuf::from("/project/src/file.txt");
    let new_path = PathBuf::from("/project/dest/file.txt");

    // Simulate MOVED_FROM
    let result = correlator.handle_moved_from(old_path.clone(), false, Some(12345));
    assert_eq!(result, RenameAction::Pending);

    // Simulate MOVED_TO with same file ID
    let result = correlator.handle_moved_to(new_path.clone(), false, Some(12345));

    assert!(matches!(result, RenameAction::IntraFilesystemMove { .. }));
    if let RenameAction::IntraFilesystemMove { old_path: op, new_path: np, is_directory } = result {
        assert_eq!(op, old_path);
        assert_eq!(np, new_path);
        assert!(!is_directory);
    }
}

#[test]
fn test_directory_rename() {
    let mut correlator = MoveCorrelator::new();

    let old_path = PathBuf::from("/project/old_folder");
    let new_path = PathBuf::from("/project/new_folder");

    // Simulate MOVED_FROM for directory
    let result = correlator.handle_moved_from(old_path.clone(), true, Some(12345));
    assert_eq!(result, RenameAction::Pending);

    // Simulate MOVED_TO with same file ID
    let result = correlator.handle_moved_to(new_path.clone(), true, Some(12345));

    assert!(matches!(result, RenameAction::SimpleRename { .. }));
    if let RenameAction::SimpleRename { old_path: op, new_path: np, is_directory } = result {
        assert_eq!(op, old_path);
        assert_eq!(np, new_path);
        assert!(is_directory);
    }
}

#[test]
fn test_combined_rename_event() {
    let mut correlator = MoveCorrelator::new();

    let old_path = PathBuf::from("/project/old.txt");
    let new_path = PathBuf::from("/project/new.txt");

    let result = correlator.handle_rename_event(old_path.clone(), new_path.clone(), false);

    assert!(matches!(result, RenameAction::SimpleRename { .. }));
}

#[test]
fn test_expired_moves_become_cross_filesystem() {
    let config = MoveCorrelatorConfig {
        correlation_timeout_secs: 0, // Immediate timeout for testing
        ..Default::default()
    };
    let mut correlator = MoveCorrelator::with_config(config);

    let old_path = PathBuf::from("/project/moved_file.txt");

    // Simulate MOVED_FROM
    let result = correlator.handle_moved_from(old_path.clone(), false, Some(12345));
    assert_eq!(result, RenameAction::Pending);

    // Wait for timeout
    sleep(Duration::from_millis(10));

    // Get expired moves
    let expired = correlator.get_expired_moves();

    assert_eq!(expired.len(), 1);
    assert!(matches!(expired[0], RenameAction::CrossFilesystemMove { .. }));
    if let RenameAction::CrossFilesystemMove { deleted_path, is_directory } = &expired[0] {
        assert_eq!(deleted_path, &old_path);
        assert!(!is_directory);
    }
}

#[test]
fn test_stats() {
    let mut correlator = MoveCorrelator::new();

    correlator.handle_moved_from(PathBuf::from("/a"), false, Some(1));
    correlator.handle_moved_from(PathBuf::from("/b"), false, Some(2));
    correlator.handle_moved_from(PathBuf::from("/c"), false, None);

    let stats = correlator.stats();
    assert_eq!(stats.pending_by_id, 2);
    assert_eq!(stats.pending_by_path, 1);
}

#[test]
fn test_clear() {
    let mut correlator = MoveCorrelator::new();

    correlator.handle_moved_from(PathBuf::from("/a"), false, Some(1));
    correlator.handle_moved_from(PathBuf::from("/b"), false, None);

    correlator.clear();

    let stats = correlator.stats();
    assert_eq!(stats.pending_by_id, 0);
    assert_eq!(stats.pending_by_path, 0);
}
