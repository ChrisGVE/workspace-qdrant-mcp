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
//!   on the Prometheus pull endpoint. (Exemplar-linked histograms are a
//!   dashboard concern wired separately and require a trace store.)
//!
//! ## Coverage note
//! Instruments are created from a snapshot of registered families at install
//! time. Nearly all daemon collectors register eagerly at `METRICS` init, so
//! they are covered. A handful of lazily-registered gauges (e.g. the State-DB
//! vacuum timestamp, absent until the first VACUUM) that appear only after
//! install are not retroactively bridged. This is acceptable for the additive
//! Phase-2 path; the Prometheus pull endpoint always exposes the full set.

use opentelemetry::metrics::AsyncInstrument;
use opentelemetry::KeyValue;
use prometheus::proto::MetricType;

use crate::monitoring::metrics_core::METRICS;

/// Install OTel observable instruments mirroring the Prometheus gauge/counter
/// families. Must be called after a global meter provider is installed
/// (see [`crate::tracing_otel::init_meter_provider`]).
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
                    .with_callback(move |observer| {
                        observe_family(observer, &family_name, MetricType::GAUGE);
                    })
                    .build();
                count += 1;
            }
            MetricType::COUNTER => {
                let family_name = name.clone();
                let _counter = meter
                    .f64_observable_counter(name)
                    .with_callback(move |observer| {
                        observe_family(observer, &family_name, MetricType::COUNTER);
                    })
                    .build();
                count += 1;
            }
            // Histograms/summaries stay pull-only (see module docs).
            _ => {}
        }
    }

    count
}

/// Read every sample of `family_name` from the live registry and observe it on
/// `observer`, carrying the Prometheus label pairs as OTel attributes.
fn observe_family(observer: &dyn AsyncInstrument<f64>, family_name: &str, ty: MetricType) {
    for family in METRICS.registry.gather() {
        if family.get_name() != family_name {
            continue;
        }
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
            observer.observe(value, &attributes);
        }
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_returns_some_instruments() {
        // METRICS registers gauge + counter families eagerly, so the bridge
        // forwards a non-zero number of instruments. This runs against whatever
        // global meter is installed (a no-op provider in unit tests); the call
        // must not panic and must report the forwarded families. The per-family
        // `observe_family` callbacks are exercised end-to-end (against a real
        // periodic reader) by the OTLP metrics integration test.
        let n = install_global_bridge();
        assert!(
            n > 0,
            "expected at least one gauge/counter family to be forwarded"
        );
    }
}
