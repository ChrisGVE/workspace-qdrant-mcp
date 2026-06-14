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
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricId {
    /// Embedding-call latency + source size. Record-shaped (`EmbedLatencyRec`).
    /// Control metric (#133 task 11 embedder lane).
    EmbedderLatency = 0,
    /// Per-item processing time (ms). Scalar. #133 tasks 8–10, migrated later.
    QueueItemMs = 1,
    /// Per-item memory footprint (KB). Scalar. #133 tasks 8–10, migrated later.
    QueueKb = 2,
    /// Drain throughput (items/s). Scalar. #133 tasks 8–10, migrated later.
    QueueThroughput = 3,
}

/// Number of `MetricId` variants — the routing-table and descriptor length.
/// Keep in lockstep with the enum (the `DESCRIPTORS` array length is checked
/// against this at compile time by its type annotation).
pub const METRIC_COUNT: usize = 4;

/// Static metadata for one metric kind: zone path, base name, unit. Indexed by
/// `MetricId as usize` (same order as the enum).
pub struct MetricDescriptor {
    pub zone: &'static str,
    pub name: &'static str,
    pub unit: &'static str,
}

/// One descriptor per `MetricId`, in discriminant order.
pub const DESCRIPTORS: [MetricDescriptor; METRIC_COUNT] = [
    MetricDescriptor {
        zone: "embedding",
        name: "latency",
        unit: "ms",
    },
    MetricDescriptor {
        zone: "queue",
        name: "item_ms",
        unit: "ms",
    },
    MetricDescriptor {
        zone: "queue",
        name: "memory_kb",
        unit: "kb",
    },
    MetricDescriptor {
        zone: "queue",
        name: "throughput",
        unit: "items_s",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discriminants_are_dense_and_ordered() {
        assert_eq!(MetricId::EmbedderLatency as usize, 0);
        assert_eq!(MetricId::QueueItemMs as usize, 1);
        assert_eq!(MetricId::QueueKb as usize, 2);
        assert_eq!(MetricId::QueueThroughput as usize, 3);
    }

    #[test]
    fn test_descriptor_count_matches_metric_count() {
        assert_eq!(DESCRIPTORS.len(), METRIC_COUNT);
    }
}
