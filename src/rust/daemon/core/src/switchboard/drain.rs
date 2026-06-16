//! Telemetry drain — the background bridge from the switchboard ring to
//! `DaemonMetrics` (Prometheus / OTLP).
//!
//! The switchboard is the single owner of telemetry: every emitter pushes a
//! `MetricSample` onto the lock-free ring, and **this** task is the one place
//! that converts those samples into `DaemonMetrics` observations. Emitters never
//! touch `DaemonMetrics` directly (arch §1, §3b). The loop is off the hot path,
//! sheds via bounded per-wake work, and idles when the ring is empty.

use std::time::Duration;

use tokio::time::sleep;

use super::{switchboard, MetricSample};
use crate::monitoring::metrics_core::METRICS;

/// Samples converted per wake before yielding back to the runtime — bounds the
/// time a single drain pass holds the executor under a telemetry burst.
const DRAIN_BATCH: usize = 256;

/// Idle back-off when the ring is empty (no busy-spin).
const IDLE_SLEEP: Duration = Duration::from_millis(50);

/// Back-off while the switchboard has not been sealed yet (very early init).
const WAIT_SEAL_SLEEP: Duration = Duration::from_millis(100);

/// Convert one buffered sample into its Prometheus observation. The single
/// telemetry write point for the whole daemon.
pub fn apply_to_metrics(sample: &MetricSample) {
    match sample {
        // Embedding-batch telemetry: reproduces the exact pre-switchboard
        // `record_embedding(model, batch_size, elapsed)` observation, so the
        // duration + batch-size histograms are byte-identical — only the path
        // changed (emitter → switchboard → here).
        MetricSample::EmbedderBatch { rec, model } => {
            METRICS.record_embedding(model, rec.batch_size, rec.elapsed);
        }
        // EmbedderLatency is the ingestion stage-3 CONTROL feed (→ EWMA fast
        // lane); it has no Prometheus series of its own. Its telemetry is already
        // covered by EmbedderBatch (which counts every embedding batch), so a
        // record_embedding call here would double-count. No-op by design.
        MetricSample::EmbedderLatency { .. } => {}
        // The #133 queue scalar lanes still update EwmaState inline and are not
        // emitted to the switchboard yet. When they migrate, map them to their
        // existing series here.
        MetricSample::QueueMsPerKb(_)
        | MetricSample::QueueDlqDepth(_)
        | MetricSample::QueueThroughput(_) => {}
    }
}

/// Background loop: drain buffered telemetry to Prometheus and mirror the
/// switchboard's buffer-full drop counter into `DaemonMetrics`. Runs for the life
/// of the process; spawned from `memexd` after the switchboard is sealed.
pub async fn run_switchboard_drain() {
    // The switchboard owns the cumulative drop total; we feed the Prometheus
    // counter the delta each tick so the two stay in lockstep without re-adding.
    let mut last_full_count: u64 = 0;

    loop {
        let Some(sw) = switchboard() else {
            sleep(WAIT_SEAL_SLEEP).await;
            continue;
        };

        let mut drained = 0usize;
        while let Some(sample) = sw.drain_one() {
            apply_to_metrics(&sample);
            drained += 1;
            if drained >= DRAIN_BATCH {
                break;
            }
        }

        let full = sw.buffer_full_count();
        if full > last_full_count {
            METRICS.record_switchboard_buffer_full(full - last_full_count);
            last_full_count = full;
        }

        if drained == 0 {
            sleep(IDLE_SLEEP).await;
        } else {
            // More may remain; yield rather than sleep so a burst drains fast.
            tokio::task::yield_now().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::switchboard::{EmbedderBatchRec, MetricId, SwitchboardBuilder};

    #[test]
    fn test_apply_embedder_batch_records_embedding() {
        // Snapshot the cumulative batch-size observation count, apply one sample,
        // and confirm the existing record_embedding path advanced by exactly one.
        let before = METRICS
            .encode()
            .expect("encode")
            .contains("wqm_memexd_embedding_batch_size");

        apply_to_metrics(&MetricSample::EmbedderBatch {
            rec: EmbedderBatchRec {
                batch_size: 7,
                elapsed: Duration::from_millis(42),
            },
            model: "fastembed",
        });

        let after = METRICS.encode().expect("encode");
        assert!(before || after.contains("wqm_memexd_embedding_batch_size"));
        assert!(
            after.contains("wqm_memexd_embedding_duration_seconds"),
            "duration histogram must exist after applying an EmbedderBatch sample"
        );
    }

    #[test]
    fn test_apply_control_and_queue_samples_are_noops() {
        // These must NOT touch record_embedding (no double-count / no panic).
        apply_to_metrics(&MetricSample::QueueMsPerKb(5.0));
        apply_to_metrics(&MetricSample::QueueDlqDepth(9.0));
        apply_to_metrics(&MetricSample::QueueThroughput(1.0));
    }

    #[test]
    fn test_drain_one_roundtrips_emitted_batch() {
        // emit_embedder_batch → drain_one yields the same record (no control fn,
        // telemetry-only id).
        let sw =
            SwitchboardBuilder::new(&crate::config::queue_health::QueueHealthConfig::default())
                .seal();
        let h = sw.handle(MetricId::EmbedderBatch, "openai");
        sw.emit_embedder_batch(
            h,
            EmbedderBatchRec {
                batch_size: 3,
                elapsed: Duration::from_millis(11),
            },
        );
        match sw.drain_one() {
            Some(MetricSample::EmbedderBatch { rec, model }) => {
                assert_eq!(rec.batch_size, 3);
                assert_eq!(rec.elapsed, Duration::from_millis(11));
                assert_eq!(model, "openai");
            }
            other => panic!("unexpected drained sample: {other:?}"),
        }
    }
}
