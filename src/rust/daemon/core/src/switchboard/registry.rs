//! Metric id registry — the single source of truth for every metric kind.
//!
//! `MetricId` is a `#[repr(usize)]` enum whose discriminant (`id as usize`) is
//! the array index into the routing table (`routing.rs`). No hash, no map: the
//! id IS the routing key. Adding a metric is a compile-everywhere change — a new
//! enum variant forces a new `MetricSample` variant (`sample.rs`) and a matching
//! `RoutingEntry`, so divergence is a compile error rather than a silent gap.
//!
//! `MetricId` must never carry runtime state, contain routing logic, or be
//! mutated after init (see arch doc §2, §5a).

/// Every metric kind that flows through the switchboard. The discriminant is the
/// routing-table index (`id as usize`); keep the explicit `= N` assignments
/// contiguous from 0 so the index space stays dense.
///
/// Discriminants 0 (`EmbedderLatency`) and 4 (`EmbedderBatch`) are
/// **telemetry-load-bearing and frozen** — they back the shipped Prometheus
/// embedding series and must not shift (#133 F2a, no telemetry regression).
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricId {
    /// Embedding-call latency + source size. Record-shaped (`EmbedLatencyRec`).
    /// Control metric (#133 embedder lane).
    EmbedderLatency = 0,
    /// Per-byte processing cost (ms per KB). Scalar `f64`. Control metric — the
    /// A1 trend probe's signal (#133 F2a, was the `QueueItemMs` stub).
    QueueMsPerKb = 1,
    /// Dead-letter-queue depth (count). Scalar `u64`. Control metric — the A3
    /// delta-rate probe's signal (#133 F2a, was the `QueueKb` stub). NOT
    /// persisted (re-seeded per poll from a live count).
    QueueDlqDepth = 2,
    /// Drain throughput (bytes/s). Scalar `f64`. Control metric — the drain
    /// (F5) probe's denominator.
    QueueThroughput = 3,
    /// Embedding-batch telemetry: duration + batch size for one
    /// `generate_embeddings_batch` call. Record-shaped (`EmbedderBatchRec`).
    /// **Telemetry-only** (no control fn, no slow lane) — routes the pre-existing
    /// `record_embedding` Prometheus path through the switchboard so the hub owns
    /// all telemetry. Distinct from `EmbedderLatency`, which is the ingestion
    /// stage-3 *control* feed; this id covers **every** embedding batch call.
    EmbedderBatch = 4,
}

/// Number of `MetricId` variants — the routing-table and descriptor length.
/// Keep in lockstep with the enum (the `DESCRIPTORS` array length is checked
/// against this at compile time by its type annotation).
pub const METRIC_COUNT: usize = 5;

impl MetricId {
    /// Every `MetricId` in discriminant order — lets callers enumerate the id
    /// space (e.g. to derive the persist allow-list from `DESCRIPTORS`).
    pub const ALL: [MetricId; METRIC_COUNT] = [
        MetricId::EmbedderLatency,
        MetricId::QueueMsPerKb,
        MetricId::QueueDlqDepth,
        MetricId::QueueThroughput,
        MetricId::EmbedderBatch,
    ];

    /// The stable variant-name string used as the `control_baseline.metric_id`
    /// key (persistence) and in logs. Kept exhaustive so a new variant is a
    /// compile error here.
    pub const fn variant_name(self) -> &'static str {
        match self {
            MetricId::EmbedderLatency => "EmbedderLatency",
            MetricId::QueueMsPerKb => "QueueMsPerKb",
            MetricId::QueueDlqDepth => "QueueDlqDepth",
            MetricId::QueueThroughput => "QueueThroughput",
            MetricId::EmbedderBatch => "EmbedderBatch",
        }
    }
}

/// Static metadata for one metric kind: zone path, base name, unit, and whether
/// its slow lane is persisted to `control_baseline`. Indexed by `MetricId as
/// usize` (same order as the enum).
pub struct MetricDescriptor {
    pub zone: &'static str,
    pub name: &'static str,
    pub unit: &'static str,
    /// Whether this metric's slow-lane baseline is persisted to
    /// `control_baseline`. `true` only for control ids that carry a meaningful
    /// long-run baseline: `EmbedderLatency`, `QueueMsPerKb`, `QueueThroughput`.
    /// `false` for `QueueDlqDepth` (re-seeded per poll from a live count) and
    /// `EmbedderBatch` (telemetry-only, no slow lane). This flag is the single
    /// gate the persist task consults for BOTH the flush and the prune
    /// allow-list (DATA-05/DATA-08), so the two can never diverge.
    pub persist: bool,
}

/// One descriptor per `MetricId`, in discriminant order.
pub const DESCRIPTORS: [MetricDescriptor; METRIC_COUNT] = [
    MetricDescriptor {
        zone: "embedding",
        name: "latency",
        unit: "ms",
        persist: true,
    },
    MetricDescriptor {
        zone: "queue",
        name: "ms_per_kb",
        unit: "ms",
        persist: true,
    },
    MetricDescriptor {
        zone: "queue",
        name: "dlq_depth",
        unit: "count",
        persist: false,
    },
    MetricDescriptor {
        zone: "queue",
        name: "throughput",
        unit: "items_s",
        persist: true,
    },
    MetricDescriptor {
        zone: "embedding",
        name: "batch",
        unit: "ms",
        persist: false,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discriminants_are_dense_and_ordered() {
        assert_eq!(MetricId::EmbedderLatency as usize, 0);
        assert_eq!(MetricId::QueueMsPerKb as usize, 1);
        assert_eq!(MetricId::QueueDlqDepth as usize, 2);
        assert_eq!(MetricId::QueueThroughput as usize, 3);
        assert_eq!(MetricId::EmbedderBatch as usize, 4);
    }

    #[test]
    fn test_telemetry_load_bearing_ids_unchanged() {
        // #133 F2a: the two shipped Prometheus-backed ids must not shift.
        assert_eq!(MetricId::EmbedderLatency as usize, 0);
        assert_eq!(MetricId::EmbedderBatch as usize, 4);
    }

    #[test]
    fn test_descriptor_count_matches_metric_count() {
        assert_eq!(DESCRIPTORS.len(), METRIC_COUNT);
        assert_eq!(MetricId::ALL.len(), METRIC_COUNT);
    }

    #[test]
    fn test_variant_name_matches_all_order() {
        // ALL is in discriminant order, so its index equals `as usize`.
        for (i, id) in MetricId::ALL.iter().enumerate() {
            assert_eq!(*id as usize, i);
        }
        assert_eq!(MetricId::QueueMsPerKb.variant_name(), "QueueMsPerKb");
        assert_eq!(MetricId::QueueDlqDepth.variant_name(), "QueueDlqDepth");
    }

    #[test]
    fn test_persist_flags_set_correctly() {
        // Persist only the three control ids with a meaningful long-run baseline.
        assert!(DESCRIPTORS[MetricId::EmbedderLatency as usize].persist);
        assert!(DESCRIPTORS[MetricId::QueueMsPerKb as usize].persist);
        assert!(DESCRIPTORS[MetricId::QueueThroughput as usize].persist);
        assert!(!DESCRIPTORS[MetricId::QueueDlqDepth as usize].persist);
        assert!(!DESCRIPTORS[MetricId::EmbedderBatch as usize].persist);
    }
}
