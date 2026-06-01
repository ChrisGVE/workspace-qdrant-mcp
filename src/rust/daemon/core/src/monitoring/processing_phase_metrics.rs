//! Phase-2 processing-phase metrics (PRD D5).
//!
//! Per-phase duration histogram and item counter for the file-processing
//! pipeline. The pipeline records internal phase names (`parse`, `embed`,
//! `graph`, `upsert`, `fts5`); [`canonical_phase`] maps the five user-facing
//! phases to their canonical D5 labels and ignores any other internal phase.
//!
//! These are **processing-layer** metrics, not graph metrics — there is
//! deliberately NO `layer` label (D5).

use once_cell::sync::Lazy;
use prometheus::{HistogramVec, IntCounterVec, Opts};

use crate::monitoring::{METRICS, PROCESSING_DURATION_BUCKETS};

/// Canonical D5 processing phases (fixed enum → bounded `phase` cardinality).
pub const CANONICAL_PHASES: [&str; 5] = [
    "chunk",
    "embed",
    "qdrant_upsert",
    "search_index",
    "graph_extract",
];

/// Map an internal pipeline phase name to its canonical D5 phase label.
/// Returns `None` for internal phases that are not part of the canonical
/// five (e.g. `extract`, `keyword`, `tier2_tagging`) — those are not emitted.
pub fn canonical_phase(internal: &str) -> Option<&'static str> {
    match internal {
        "parse" => Some("chunk"),
        "embed" => Some("embed"),
        "upsert" => Some("qdrant_upsert"),
        "fts5" => Some("search_index"),
        "graph" => Some("graph_extract"),
        _ => None,
    }
}

struct ProcessingPhaseMetrics {
    duration_seconds: HistogramVec,
    items_total: IntCounterVec,
}

impl ProcessingPhaseMetrics {
    fn new() -> Self {
        let duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "wqm_memexd_processing_phase_duration_seconds",
                "Duration of each file-processing phase in seconds",
            )
            .buckets(PROCESSING_DURATION_BUCKETS.to_vec()),
            &["phase", "tenant_id", "collection"],
        )
        .expect("metric can be created");
        let items_total = IntCounterVec::new(
            Opts::new(
                "wqm_memexd_processing_phase_items_total",
                "Total items processed per file-processing phase",
            ),
            &["phase", "tenant_id"],
        )
        .expect("metric can be created");

        let r = &METRICS.registry;
        let _ = r.register(Box::new(duration_seconds.clone()));
        let _ = r.register(Box::new(items_total.clone()));

        Self {
            duration_seconds,
            items_total,
        }
    }
}

static PROCESSING_PHASE: Lazy<ProcessingPhaseMetrics> = Lazy::new(ProcessingPhaseMetrics::new);

/// Record one completed processing phase: observe its duration and increment
/// the item counter. `internal_phase` is the pipeline's internal phase name;
/// non-canonical phases are silently ignored. No-op when telemetry is off.
pub fn record_processing_phase(
    internal_phase: &str,
    tenant_id: &str,
    collection: &str,
    duration_secs: f64,
) {
    if !METRICS.is_enabled() {
        return;
    }
    let Some(phase) = canonical_phase(internal_phase) else {
        return;
    };
    let m = &*PROCESSING_PHASE;
    m.duration_seconds
        .with_label_values(&[phase, tenant_id, collection])
        .observe(duration_secs);
    m.items_total.with_label_values(&[phase, tenant_id]).inc();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_phase_maps_five_and_ignores_others() {
        assert_eq!(canonical_phase("parse"), Some("chunk"));
        assert_eq!(canonical_phase("embed"), Some("embed"));
        assert_eq!(canonical_phase("upsert"), Some("qdrant_upsert"));
        assert_eq!(canonical_phase("fts5"), Some("search_index"));
        assert_eq!(canonical_phase("graph"), Some("graph_extract"));
        assert_eq!(canonical_phase("keyword"), None);
        assert_eq!(canonical_phase("extract"), None);
        assert_eq!(CANONICAL_PHASES.len(), 5);
    }

    #[test]
    fn record_uses_bounded_phase_labels() {
        // Record a canonical and a non-canonical phase; only the canonical one
        // produces a series, and its phase label is from the canonical set.
        record_processing_phase("upsert", "tenant_a", "projects", 0.012);
        record_processing_phase("keyword", "tenant_a", "projects", 0.5);

        let families = METRICS.registry.gather();
        let dur = families
            .iter()
            .find(|f| f.get_name() == "wqm_memexd_processing_phase_duration_seconds")
            .expect("duration family present");
        for metric in dur.get_metric() {
            let phase = metric
                .get_label()
                .iter()
                .find(|l| l.get_name() == "phase")
                .map(|l| l.get_value())
                .unwrap_or("");
            assert!(
                CANONICAL_PHASES.contains(&phase),
                "unexpected phase label: {phase}"
            );
        }
    }

    #[test]
    fn duration_metric_has_no_layer_label() {
        record_processing_phase("embed", "tenant_b", "projects", 0.1);
        let families = METRICS.registry.gather();
        let dur = families
            .iter()
            .find(|f| f.get_name() == "wqm_memexd_processing_phase_duration_seconds")
            .expect("duration family present");
        for metric in dur.get_metric() {
            for label in metric.get_label() {
                assert_ne!(
                    label.get_name(),
                    "layer",
                    "processing_phase_duration_seconds must not carry a layer label"
                );
            }
        }
    }
}
