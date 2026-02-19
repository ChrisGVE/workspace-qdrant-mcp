//! Strategy Pattern: interchangeable processing algorithms behind stable interfaces.
//!
//! Each `ProcessingStrategy` handles a specific combination of (item_type, op)
//! from the unified queue. The queue processor dispatches to the appropriate
//! strategy via the `StrategyRegistry`.
//!
//! # Submodules
//! - `processing/` -- per item_type strategies (file, text, folder, tenant, etc.)
//!
//! # Future submodules
//! - `scan` -- directory scanning strategies
//! - `retry` -- retry policy strategies

pub mod processing;

use async_trait::async_trait;

use crate::context::ProcessingContext;
use crate::unified_queue_processor::UnifiedProcessorError;
use crate::unified_queue_schema::{ItemType, QueueOperation, UnifiedQueueItem};

/// Trait for queue item processing strategies.
///
/// Each strategy handles one or more (item_type, op) combinations.
/// The `processing_loop` dispatches to the matching strategy via
/// `StrategyRegistry::dispatch()`.
#[async_trait]
pub trait ProcessingStrategy: Send + Sync {
    /// Check if this strategy handles the given item type and operation.
    fn handles(&self, item_type: &ItemType, op: &QueueOperation) -> bool;

    /// Process a single queue item using the shared processing context.
    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError>;

    /// Human-readable name for logging and metrics.
    fn name(&self) -> &'static str;
}

/// Registry of processing strategies for dispatch.
///
/// Strategies are checked in registration order; the first match wins.
pub struct StrategyRegistry {
    strategies: Vec<Box<dyn ProcessingStrategy>>,
}

impl StrategyRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Register a strategy. Strategies are matched in registration order.
    pub fn register(&mut self, strategy: Box<dyn ProcessingStrategy>) {
        self.strategies.push(strategy);
    }

    /// Find the first strategy that handles the given item type and operation.
    pub fn dispatch(
        &self,
        item_type: &ItemType,
        op: &QueueOperation,
    ) -> Option<&dyn ProcessingStrategy> {
        self.strategies
            .iter()
            .find(|s| s.handles(item_type, op))
            .map(|s| s.as_ref())
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy strategy for testing the registry.
    struct DummyStrategy {
        target_type: ItemType,
    }

    #[async_trait]
    impl ProcessingStrategy for DummyStrategy {
        fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
            *item_type == self.target_type
        }

        async fn process(
            &self,
            _ctx: &ProcessingContext,
            _item: &UnifiedQueueItem,
        ) -> Result<(), UnifiedProcessorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "dummy"
        }
    }

    #[test]
    fn test_registry_dispatch_match() {
        let mut registry = StrategyRegistry::new();
        registry.register(Box::new(DummyStrategy {
            target_type: ItemType::File,
        }));
        registry.register(Box::new(DummyStrategy {
            target_type: ItemType::Text,
        }));

        let found = registry.dispatch(&ItemType::File, &QueueOperation::Add);
        assert!(found.is_some());
        assert_eq!(found.unwrap().name(), "dummy");
    }

    #[test]
    fn test_registry_dispatch_no_match() {
        let registry = StrategyRegistry::new();
        let found = registry.dispatch(&ItemType::File, &QueueOperation::Add);
        assert!(found.is_none());
    }

    #[test]
    fn test_registry_dispatch_first_wins() {
        let mut registry = StrategyRegistry::new();
        registry.register(Box::new(DummyStrategy {
            target_type: ItemType::File,
        }));
        registry.register(Box::new(DummyStrategy {
            target_type: ItemType::File,
        }));

        // Should find the first one
        let found = registry.dispatch(&ItemType::File, &QueueOperation::Add);
        assert!(found.is_some());
    }
}
