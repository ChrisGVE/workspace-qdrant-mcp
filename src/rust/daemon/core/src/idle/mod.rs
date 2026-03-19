//! Idle state taxonomy and maintenance scheduler.
//!
//! Formalizes the daemon's idle states and provides a framework for running
//! maintenance tasks (orphan cleanup, filesystem reconciliation, etc.) during
//! idle periods with automatic yield when real work arrives.

mod scheduler;
mod task;

pub mod tasks;

pub use scheduler::{MaintenanceScheduler, MaintenanceTaskStatus};
pub use task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};

/// What the daemon can do right now.
///
/// Determined each loop iteration based on queue depth, Qdrant availability,
/// and memory pressure. Maintenance tasks declare which states they can run in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleState {
    /// Queue empty, Qdrant available — full maintenance window.
    FullIdle,
    /// Queue has items but Qdrant circuit is open — SQLite+filesystem only.
    QdrantDownIdle,
    /// Under memory pressure — light bookkeeping only.
    ResourceConstrained,
    /// Actively processing queue items.
    Active,
}

impl IdleState {
    /// Determine the current idle state from system conditions.
    pub fn determine(queue_depth: i64, qdrant_available: bool, memory_pressure: bool) -> Self {
        if memory_pressure {
            return IdleState::ResourceConstrained;
        }
        if queue_depth > 0 && qdrant_available {
            return IdleState::Active;
        }
        if !qdrant_available {
            return IdleState::QdrantDownIdle;
        }
        IdleState::FullIdle
    }

    /// Can any maintenance task run in this state?
    pub fn allows_maintenance(&self) -> bool {
        !matches!(self, IdleState::Active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_idle() {
        assert_eq!(IdleState::determine(0, true, false), IdleState::FullIdle);
    }

    #[test]
    fn test_active_when_work_and_qdrant() {
        assert_eq!(IdleState::determine(10, true, false), IdleState::Active);
    }

    #[test]
    fn test_qdrant_down_idle() {
        assert_eq!(
            IdleState::determine(5, false, false),
            IdleState::QdrantDownIdle
        );
    }

    #[test]
    fn test_qdrant_down_empty_queue() {
        // Queue empty but Qdrant down — can still do SQLite maintenance
        assert_eq!(
            IdleState::determine(0, false, false),
            IdleState::QdrantDownIdle
        );
    }

    #[test]
    fn test_memory_pressure_overrides() {
        assert_eq!(
            IdleState::determine(0, true, true),
            IdleState::ResourceConstrained
        );
        assert_eq!(
            IdleState::determine(10, true, true),
            IdleState::ResourceConstrained
        );
    }

    #[test]
    fn test_allows_maintenance() {
        assert!(IdleState::FullIdle.allows_maintenance());
        assert!(IdleState::QdrantDownIdle.allows_maintenance());
        assert!(IdleState::ResourceConstrained.allows_maintenance());
        assert!(!IdleState::Active.allows_maintenance());
    }
}
