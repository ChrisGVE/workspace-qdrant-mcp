//! Priority queue item for the task execution pipeline

use std::cmp::Ordering;

use super::super::PriorityTask;

/// Priority queue implementation for tasks.
/// Uses reverse ordering so highest priority comes first.
pub(crate) struct TaskQueueItem {
    pub(crate) task: PriorityTask,
    pub(crate) sequence: u64,
}

impl PartialEq for TaskQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.task.context.priority == other.task.context.priority
            && self.sequence == other.sequence
    }
}

impl Eq for TaskQueueItem {}

impl PartialOrd for TaskQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaskQueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (higher priority first)
        match self.task.context.priority.cmp(&other.task.context.priority) {
            Ordering::Equal => {
                // If priorities are equal, use sequence number (FIFO)
                other.sequence.cmp(&self.sequence)
            }
            other_order => other_order,
        }
    }
}
