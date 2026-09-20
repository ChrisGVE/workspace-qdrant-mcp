//! Tests for how the loop scores a finished batch (GitHub #295).
//!
//! Only the pure scoring step is covered: `process_batch` itself needs a live
//! queue, a storage client and an embedding provider, so its outcome is not a
//! property of this code.

#[cfg(test)]
mod tests {
    use super::super::batch_processing::BatchOutcome;
    use super::super::loop_core::apply_batch_outcome;
    use super::super::loop_state::LoopState;
    use crate::unified_queue_processor::config::UnifiedProcessorConfig;

    fn state() -> LoopState {
        LoopState::new(&UnifiedProcessorConfig::default(), 0)
    }

    #[test]
    fn a_fully_parked_batch_is_not_scored_as_a_dispatch() {
        // Every item was parked because the embedding provider is down. No
        // work moved, so the loop must see a non-dispatch and back off.
        let mut s = state();
        s.last_poll_dispatched = true;
        apply_batch_outcome(&mut s, &BatchOutcome::parked());
        assert!(
            !s.last_poll_dispatched,
            "a parked batch moved nothing and is not progress"
        );
    }

    #[test]
    fn a_dispatch_that_moved_no_tenant_still_counts_as_progress() {
        // The guard against the naive fix: a batch of deletes processes items
        // and reports no tenant, which is nothing like a parked batch.
        let mut s = state();
        s.last_poll_dispatched = false;
        apply_batch_outcome(&mut s, &BatchOutcome::dispatched(Default::default()));
        assert!(
            s.last_poll_dispatched,
            "an empty tenant set is not an empty batch"
        );
    }

    #[test]
    fn a_parked_batch_does_not_consume_the_recovery_ramp() {
        let mut s = state();
        s.recovery_ramp_remaining = 3;
        apply_batch_outcome(&mut s, &BatchOutcome::parked());
        assert_eq!(
            s.recovery_ramp_remaining, 3,
            "the ramp counts batches that ran"
        );
        apply_batch_outcome(&mut s, &BatchOutcome::dispatched(Default::default()));
        assert_eq!(s.recovery_ramp_remaining, 2);
    }
}
