//! Folder processing strategy.
//!
//! Handles `ItemType::Folder` queue items: directory scanning (progressive
//! single-level enumeration), folder deletion (cascading to tracked files),
//! and folder update/add (treated as rescan).

mod delete;
mod scan;
mod strategy;

pub use strategy::FolderStrategy;

#[cfg(test)]
mod tests {
    use crate::unified_queue_schema::{ItemType, QueueOperation};

    use super::strategy::FolderStrategy;
    use crate::strategies::ProcessingStrategy;

    #[test]
    fn test_folder_strategy_handles_folder_items() {
        let strategy = FolderStrategy;
        assert!(strategy.handles(&ItemType::Folder, &QueueOperation::Scan));
        assert!(strategy.handles(&ItemType::Folder, &QueueOperation::Add));
        assert!(strategy.handles(&ItemType::Folder, &QueueOperation::Delete));
    }

    #[test]
    fn test_folder_strategy_rejects_non_folder_items() {
        let strategy = FolderStrategy;
        assert!(!strategy.handles(&ItemType::File, &QueueOperation::Scan));
        assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Tenant, &QueueOperation::Delete));
    }

    #[test]
    fn test_folder_strategy_name() {
        let strategy = FolderStrategy;
        assert_eq!(strategy.name(), "folder");
    }
}
