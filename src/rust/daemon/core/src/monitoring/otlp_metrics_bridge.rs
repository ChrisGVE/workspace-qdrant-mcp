//! Bridge the daemon's Prometheus registry onto the OpenTelemetry metrics SDK.
//!
//! Task 88 (Phase-2 OTLP metrics path). The daemon's canonical metric surface
//! is the Prometheus pull endpoint (`/metrics`, port 6337); all collectors live
//! in [`crate::monitoring::metrics_core::METRICS`]'s [`prometheus::Registry`].
//! When the operator opts into the additive OTLP **metrics** push path
//! (`telemetry.otlp.enabled` AND `telemetry.otlp.metrics_enabled`), an
//! [`opentelemetry_sdk::metrics::SdkMeterProvider`] is installed globally and
//! this bridge forwards the registry's gauge and counter families onto OTel
//! observable instruments so real daemon data flows over OTLP — not a stub.
//!
//! ## What is forwarded
//! - **Gauge** families → `f64_observable_gauge` (current value).
//! - **Counter** families → `f64_observable_counter` (cumulative value; OTel
//!   cumulative temporality matches Prometheus counter semantics).
//! - **Histogram / Summary** families are intentionally NOT forwarded: the OTel
//!   observable-instrument API records a single scalar per observation and
//!   cannot reproduce bucket boundaries. Latency distributions remain available
//!   on the Prometheus pull endpoint.
//!
//! ## One gather per export tick
//! OTel 0.31 exposes no batch-observer API, so each forwarded family gets its
//! own `with_callback`. To avoid an O(N²) `Registry::gather()` (one full
//! serialize per family per export), the callbacks share a short-TTL snapshot:
//! the first callback of an export tick gathers once and caches the result for
//! [`SNAPSHOT_TTL`]; the remaining callbacks in that tick reuse it. Exports are
//! ~60s apart, far longer than the TTL, so each tick triggers exactly one
//! gather.
//!
//! ## Coverage note
//! Instruments are created from the registered families at install time. The
//! caller installs this bridge after the daemon's metric subsystems have been
//! exercised; families that register lazily *after* install (e.g. the State-DB
//! vacuum gauge, absent until the first VACUUM) are not retroactively bridged.
//! This is acceptable for the additive Phase-2 path — the Prometheus pull
//! endpoint always exposes the full set, and the install log reports the count.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use opentelemetry::metrics::AsyncInstrument;
use opentelemetry::KeyValue;
use prometheus::proto::MetricType;

use crate::monitoring::metrics_core::METRICS;

/// How long a gathered snapshot is reused across instrument callbacks. Must be
/// well under the OTLP export interval (~60s) but longer than the burst of
/// callbacks within a single export tick (sub-millisecond).
const SNAPSHOT_TTL: Duration = Duration::from_secs(5);

/// A single observable sample: pre-resolved attributes and the scalar value.
type FamilySamples = Vec<(Vec<KeyValue>, f64)>;

/// Cached per-tick snapshot of the gauge/counter samples, keyed by family name.
static SNAPSHOT: Lazy<Mutex<Option<(Instant, Arc<HashMap<String, FamilySamples>>)>>> =
    Lazy::new(|| Mutex::new(None));

/// Install OTel observable instruments mirroring the Prometheus gauge/counter
/// families. Must be called after a global meter provider is installed
/// (see [`crate::tracing_otel::init_meter_provider`]) and after the daemon's
/// metric subsystems have registered their collectors.
///
/// Returns the number of instruments registered (one per forwarded family).
pub fn install_global_bridge() -> usize {
    let meter = opentelemetry::global::meter("memexd");
    let families = METRICS.registry.gather();
    let mut count = 0usize;

    for family in &families {
        let name = family.get_name().to_string();
        match family.get_field_type() {
            MetricType::GAUGE => {
                let family_name = name.clone();
                // The returned handle may be dropped: the SDK meter retains the
                // callback for the lifetime of the provider.
                let _gauge = meter
                    .f64_observable_gauge(name)
                    .with_callback(move |observer| observe_family(observer, &family_name))
                    .build();
                count += 1;
            }
            MetricType::COUNTER => {
                let family_name = name.clone();
                let _counter = meter
                    .f64_observable_counter(name)
                    .with_callback(move |observer| observe_family(observer, &family_name))
                    .build();
                count += 1;
            }
            // Histograms/summaries stay pull-only (see module docs).
            _ => {}
        }
    }

    count
}

/// Observe every sample of `family_name` from the current export-tick snapshot.
fn observe_family(observer: &dyn AsyncInstrument<f64>, family_name: &str) {
    let snapshot = current_snapshot();
    if let Some(samples) = snapshot.get(family_name) {
        for (attributes, value) in samples {
            observer.observe(*value, attributes);
        }
    }
}

/// Return the gauge/counter sample snapshot, gathering the registry at most once
/// per [`SNAPSHOT_TTL`]. Returns an empty snapshot when the metrics kill-switch
/// is off, so a runtime-disabled registry stops feeding OTLP rather than
/// re-exporting stale values.
fn current_snapshot() -> Arc<HashMap<String, FamilySamples>> {
    let mut guard = match SNAPSHOT.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some((taken_at, snapshot)) = guard.as_ref() {
        if taken_at.elapsed() < SNAPSHOT_TTL {
            return Arc::clone(snapshot);
        }
    }

    let snapshot = Arc::new(build_snapshot());
    *guard = Some((Instant::now(), Arc::clone(&snapshot)));
    snapshot
}

/// Gather the registry once and pre-resolve gauge/counter samples by family.
fn build_snapshot() -> HashMap<String, FamilySamples> {
    let mut map: HashMap<String, FamilySamples> = HashMap::new();
    if !METRICS.is_enabled() {
        return map;
    }
    for family in METRICS.registry.gather() {
        let ty = family.get_field_type();
        if ty != MetricType::GAUGE && ty != MetricType::COUNTER {
            continue;
        }
        let mut samples: FamilySamples = Vec::new();
        for metric in family.get_metric() {
            let attributes: Vec<KeyValue> = metric
                .get_label()
                .iter()
                .map(|label| {
                    KeyValue::new(label.get_name().to_string(), label.get_value().to_string())
                })
                .collect();
            let value = match ty {
                MetricType::GAUGE => metric.get_gauge().get_value(),
                MetricType::COUNTER => metric.get_counter().get_value(),
                _ => continue,
            };
            samples.push((attributes, value));
        }
        map.insert(family.get_name().to_string(), samples);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_returns_some_instruments() {
        // METRICS registers gauge + counter families eagerly, so the bridge
        // forwards a non-zero number of instruments. Runs against whatever
        // global meter is installed (a no-op provider in unit tests); the call
        // must not panic and must report the forwarded families. The callbacks'
        // observe path is exercised end-to-end (against a real periodic reader)
        // by the OTLP metrics integration test.
        let n = install_global_bridge();
        assert!(
            n > 0,
            "expected at least one gauge/counter family to be forwarded"
        );
    }

    #[test]
    fn snapshot_is_reused_within_ttl() {
        // Two calls inside the TTL must return the same cached Arc (one gather).
        let a = current_snapshot();
        let b = current_snapshot();
        assert!(
            Arc::ptr_eq(&a, &b),
            "snapshot should be reused within the TTL"
        );
    }
}
