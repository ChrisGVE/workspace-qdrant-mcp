//! The single event type that flows through the switchboard.
//!
//! `MetricSample` is the **one** event both sinks consume — the telemetry buffer
//! and the control fn-pointer. There is deliberately no separate
//! `ControlEvent`/`TelemetryEvent` split (arch doc §2, §5b). One variant per
//! `MetricId`, each carrying that id's typed record with raw values (no
//! pre-normalization — arch §1.5). `MetricSample` is `Copy`: the hot path never
//! heap-allocates.

use std::time::Duration;

use super::MetricId;

/// Co-measured fields for one embedding call. Raw values (arch §1.5).
///
/// `embed_ms` is `u128` (`Instant::elapsed().as_millis()`); it is converted to
/// `f64` only at the control store (`(embed_ms as f64).to_bits()`, arch §9), so
/// the read side recovers IEEE-754 bits. `source_bytes` is the summed length of
/// the chunk texts that were embedded.
#[derive(Debug, Clone, Copy)]
pub struct EmbedLatencyRec {
    pub embed_ms: u128,
    pub source_bytes: usize,
}

/// Telemetry record for one `generate_embeddings_batch` call: the values the
/// existing `DaemonMetrics::record_embedding` path consumes. `elapsed` is carried
/// as a `Copy` `Duration` so the drain reproduces the duration histogram with no
/// precision loss; `batch_size` feeds the batch-size histogram.
#[derive(Debug, Clone, Copy)]
pub struct EmbedderBatchRec {
    pub batch_size: usize,
    pub elapsed: Duration,
}

/// THE event type through the switchboard — telemetry buffer AND control fn.
///
/// One variant per `MetricId`. `model` rides along as the stable telemetry label
/// (never affects routing — arch §1.6). `Copy`, so no per-emit allocation.
#[derive(Debug, Clone, Copy)]
pub enum MetricSample {
    EmbedderLatency {
        rec: EmbedLatencyRec,
        model: &'static str,
    },
    /// Per-byte processing cost (ms/KB) — a ratio, carried as `f64` so the EWMA
    /// lane keeps sub-integer precision.
    QueueMsPerKb(f64),
    /// Dead-letter-queue per-poll **delta-rate** (counts/poll), carried as `f64`
    /// so a *draining* DLQ (negative delta) is preserved — the A3 probe needs the
    /// sign to distinguish draining (Green) from growing (Red). The lane smooths
    /// this rate; the absolute count is read separately for the emptiness test.
    QueueDlqDepth(f64),
    QueueThroughput(f64),
    EmbedderBatch {
        rec: EmbedderBatchRec,
        model: &'static str,
    },
}

impl MetricSample {
    /// The routing key. Matches the emitting handle's id (debug-asserted at
    /// emit). The exhaustive `match` makes a new `MetricId` variant a compile
    /// error here until its sample variant is mapped.
    #[inline]
    pub fn id(&self) -> MetricId {
        match self {
            MetricSample::EmbedderLatency { .. } => MetricId::EmbedderLatency,
            MetricSample::QueueMsPerKb(_) => MetricId::QueueMsPerKb,
            MetricSample::QueueDlqDepth(_) => MetricId::QueueDlqDepth,
            MetricSample::QueueThroughput(_) => MetricId::QueueThroughput,
            MetricSample::EmbedderBatch { .. } => MetricId::EmbedderBatch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_matches_variant() {
        let s = MetricSample::EmbedderLatency {
            rec: EmbedLatencyRec {
                embed_ms: 100,
                source_bytes: 1000,
            },
            model: "fastembed",
        };
        assert_eq!(s.id(), MetricId::EmbedderLatency);
        assert_eq!(MetricSample::QueueMsPerKb(7.0).id(), MetricId::QueueMsPerKb);
        assert_eq!(
            MetricSample::QueueDlqDepth(7.0).id(),
            MetricId::QueueDlqDepth
        );
        assert_eq!(
            MetricSample::QueueThroughput(1.0).id(),
            MetricId::QueueThroughput
        );
    }

    #[test]
    fn test_sample_is_copy() {
        let s = MetricSample::QueueDlqDepth(42.0);
        let s2 = s; // Copy — `s` still usable below.
        assert_eq!(s.id(), s2.id());
    }
}
