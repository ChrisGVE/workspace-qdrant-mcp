//! OpenTelemetry tracing setup for the daemon-server boundary.
//!
//! Implements distributed tracing across the daemon and MCP server using
//! OpenTelemetry standards. Supports both a no-op development mode (no
//! exporter) and OTLP (production) export over either gRPC (tonic) or
//! HTTP/protobuf (reqwest).
//!
//! Task 412.19: initial OpenTelemetry tracing setup.
//! Task 57: upgrade to opentelemetry 0.31 / opentelemetry-otlp 0.31 /
//!   tracing-opentelemetry 0.32. The 0.21 -> 0.31 migration replaced the
//!   `new_exporter()` / `Config` / `runtime::Tokio` APIs with the
//!   `SpanExporter::builder()` and `SdkTracerProvider::builder()` builders,
//!   and removed the global `shutdown_tracer_provider()` in favour of a
//!   retained `provider.shutdown()` handle.

mod exporter;
mod propagation;

use std::env;
use std::sync::Mutex;
use std::time::Duration;

use once_cell::sync::Lazy;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::Subscriber;
use tracing_subscriber::registry::LookupSpan;

pub use exporter::OtlpTransport;
pub use propagation::{current_traceparent, link_current_to_traceparent};

/// Globally retained tracer provider handle.
///
/// 0.31 removed `opentelemetry::global::shutdown_tracer_provider()`; flushing
/// now requires calling `SdkTracerProvider::shutdown()` on a retained handle.
/// [`otel_layer`] stores the provider here so [`shutdown_tracer`] can flush
/// buffered spans on daemon exit.
static TRACER_PROVIDER: Lazy<Mutex<Option<SdkTracerProvider>>> = Lazy::new(|| Mutex::new(None));

/// Globally retained meter provider handle (Task 88, OTLP metrics path).
///
/// Mirrors [`TRACER_PROVIDER`]: [`init_meter_provider`] installs the provider
/// globally and retains it here so [`shutdown_meter_provider`] can flush the
/// final periodic export on daemon exit.
static METER_PROVIDER: Lazy<Mutex<Option<SdkMeterProvider>>> = Lazy::new(|| Mutex::new(None));

/// Default OTLP metrics export interval (matches the 60s metrics-snapshot
/// cadence used by the Prometheus history writer).
pub const DEFAULT_OTLP_METRICS_INTERVAL: Duration = Duration::from_secs(60);

/// OpenTelemetry configuration.
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Service name for traces.
    pub service_name: String,
    /// Service version.
    pub service_version: String,
    /// OTLP endpoint (if `None`, uses a no-op provider with no exporter).
    pub otlp_endpoint: Option<String>,
    /// Sampling ratio (0.0 to 1.0).
    pub sampling_ratio: f64,
    /// Enable trace context propagation in gRPC metadata.
    pub propagate_context: bool,
    /// OTLP exporter protocol: `grpc` (tonic) or `http/protobuf` (reqwest/http).
    pub otlp_protocol: String,
    /// OTLP metadata/header entries (e.g. `authorization=Bearer ...`).
    pub otlp_headers: std::collections::HashMap<String, String>,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            service_name: "memexd".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            otlp_endpoint: env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok(),
            sampling_ratio: 1.0,
            propagate_context: true,
            otlp_protocol: "grpc".to_string(),
            otlp_headers: std::collections::HashMap::new(),
        }
    }
}

impl OtelConfig {
    /// Build an [`OtelConfig`] from the daemon's telemetry configuration.
    ///
    /// Only returns `Some` when `otlp.enabled` is true — callers should treat
    /// `None` as "do not initialize OTLP".
    pub fn from_telemetry(
        telemetry: &crate::config::TelemetryConfig,
        service_version: impl Into<String>,
    ) -> Option<Self> {
        if !telemetry.otlp.enabled {
            return None;
        }
        Some(Self {
            service_name: telemetry.service_name.clone(),
            service_version: service_version.into(),
            otlp_endpoint: Some(telemetry.otlp.endpoint.clone()),
            sampling_ratio: telemetry.otlp.sample_rate,
            propagate_context: true,
            otlp_protocol: telemetry.otlp.protocol.as_str().to_string(),
            otlp_headers: telemetry.otlp.headers.clone(),
        })
    }

    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(name) = env::var("OTEL_SERVICE_NAME") {
            config.service_name = name;
        }

        if let Ok(ratio) = env::var("OTEL_TRACES_SAMPLER_ARG") {
            if let Ok(r) = ratio.parse::<f64>() {
                config.sampling_ratio = r.clamp(0.0, 1.0);
            }
        }

        config
    }
}

/// Initialize an OpenTelemetry tracer provider for the supplied config.
///
/// When `otlp_endpoint` is set, an OTLP [`SdkTracerProvider`] is built with a
/// batch span processor over the configured transport (gRPC or HTTP/protobuf).
/// When it is `None`, a no-op provider with no exporter is returned so the rest
/// of the pipeline still has a valid provider with zero export overhead.
pub fn init_tracer_provider(
    config: &OtelConfig,
) -> Result<SdkTracerProvider, exporter::OtelInitError> {
    let resource = exporter::build_resource(config);
    let sampler = exporter::build_sampler(config.sampling_ratio);

    match &config.otlp_endpoint {
        Some(endpoint) => {
            let transport = exporter::select_transport(&config.otlp_protocol);
            tracing::info!(
                endpoint = %endpoint,
                transport = %transport.as_str(),
                "Initializing OpenTelemetry OTLP exporter"
            );
            let span_exporter =
                exporter::build_span_exporter(transport, endpoint, &config.otlp_headers)?;
            Ok(SdkTracerProvider::builder()
                .with_sampler(sampler)
                .with_resource(resource)
                .with_batch_exporter(span_exporter)
                .build())
        }
        None => {
            tracing::info!("Initializing OpenTelemetry with no exporter (development mode)");
            Ok(SdkTracerProvider::builder()
                .with_sampler(sampler)
                .with_resource(resource)
                .build())
        }
    }
}

/// Create an OpenTelemetry tracing layer for use with `tracing-subscriber`.
///
/// This layer bridges the `tracing` crate with OpenTelemetry, allowing all
/// `#[tracing::instrument]` spans to be exported as OpenTelemetry traces. The
/// built provider is set globally and retained for [`shutdown_tracer`] so
/// buffered spans are flushed on exit.
pub fn otel_layer<S>(
    config: &OtelConfig,
) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::SdkTracer>>
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    match init_tracer_provider(config) {
        Ok(provider) => {
            let tracer = provider.tracer(config.service_name.clone());
            // Set the global provider so spans created outside the tracing
            // bridge are still exported, and retain a handle for shutdown.
            opentelemetry::global::set_tracer_provider(provider.clone());
            // Install the W3C trace-context propagator so `traceparent` headers
            // are injected/extracted across the daemon<->MCP boundary (Task 58,
            // fixes C6 together with the ParentBased sampler).
            if config.propagate_context {
                opentelemetry::global::set_text_map_propagator(
                    opentelemetry_sdk::propagation::TraceContextPropagator::new(),
                );
            }
            store_provider(provider);
            Some(tracing_opentelemetry::layer().with_tracer(tracer))
        }
        Err(e) => {
            tracing::error!("Failed to initialize OpenTelemetry tracer: {}", e);
            None
        }
    }
}

/// Store the retained provider handle, replacing any previous one.
fn store_provider(provider: SdkTracerProvider) {
    if let Ok(mut guard) = TRACER_PROVIDER.lock() {
        *guard = Some(provider);
    }
}

/// Shutdown OpenTelemetry and flush remaining traces.
///
/// 0.31 removed the global `shutdown_tracer_provider()`; we flush by calling
/// `shutdown()` on the retained [`SdkTracerProvider`] handle. A no-op when no
/// provider was ever installed (OTLP disabled), preserving the zero-overhead
/// disabled path.
pub fn shutdown_tracer() {
    let provider = TRACER_PROVIDER.lock().ok().and_then(|mut g| g.take());
    if let Some(provider) = provider {
        if let Err(e) = provider.shutdown() {
            tracing::warn!("Error shutting down OpenTelemetry tracer provider: {}", e);
        } else {
            tracing::info!("OpenTelemetry tracer provider shut down");
        }
    }
}

/// Initialize and globally install the OTLP **metrics** export path (Task 88).
///
/// Builds an [`SdkMeterProvider`] with a [`PeriodicReader`] over the configured
/// OTLP transport, installs it as the global meter provider, retains it for
/// [`shutdown_meter_provider`], and bridges the daemon's Prometheus
/// gauge/counter registry onto OTel observable instruments so real data flows.
///
/// This is the additive Phase-2 path: the Prometheus pull endpoint remains the
/// primary metric transport. Returns `Ok(true)` when a provider was installed,
/// `Ok(false)` when no OTLP endpoint was configured (nothing to export).
///
/// [`PeriodicReader`]: opentelemetry_sdk::metrics::PeriodicReader
pub fn init_meter_provider(
    config: &OtelConfig,
    interval: Duration,
) -> Result<bool, exporter::OtelInitError> {
    match exporter::build_meter_provider(config, interval)? {
        Some(provider) => {
            opentelemetry::global::set_meter_provider(provider.clone());
            if let Ok(mut guard) = METER_PROVIDER.lock() {
                *guard = Some(provider);
            }
            let bridged = crate::monitoring::otlp_metrics_bridge::install_global_bridge();
            tracing::info!(
                bridged_instruments = bridged,
                "OTLP metrics export path installed (additive to Prometheus pull)"
            );
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Shutdown the OTLP metrics provider and flush the final export (Task 88).
///
/// A no-op when no provider was ever installed (OTLP metrics disabled),
/// preserving the zero-overhead disabled path. Mirrors [`shutdown_tracer`].
pub fn shutdown_meter_provider() {
    let provider = METER_PROVIDER.lock().ok().and_then(|mut g| g.take());
    if let Some(provider) = provider {
        if let Err(e) = provider.shutdown() {
            tracing::warn!("Error shutting down OpenTelemetry meter provider: {}", e);
        } else {
            tracing::info!("OpenTelemetry meter provider shut down");
        }
    }
}

// Note: gRPC context propagation functions (extract_context_from_metadata,
// inject_context_into_metadata) are implemented in the grpc crate where
// tonic types are available.

/// Get the current trace ID as a string for logging correlation.
///
/// This can be included in log messages to correlate logs with traces.
pub fn current_trace_id() -> Option<String> {
    use opentelemetry::trace::TraceContextExt;

    let context = opentelemetry::Context::current();
    let span = context.span();
    let span_context = span.span_context();

    if span_context.is_valid() {
        Some(span_context.trace_id().to_string())
    } else {
        None
    }
}

/// Get the current span ID as a string for logging correlation.
pub fn current_span_id() -> Option<String> {
    use opentelemetry::trace::TraceContextExt;

    let context = opentelemetry::Context::current();
    let span = context.span();
    let span_context = span.span_context();

    if span_context.is_valid() {
        Some(span_context.span_id().to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OtelConfig::default();
        assert_eq!(config.service_name, "memexd");
        assert_eq!(config.sampling_ratio, 1.0);
        assert!(config.propagate_context);
    }

    #[test]
    fn test_config_from_env() {
        // Test that from_env doesn't panic.
        let config = OtelConfig::from_env();
        assert!(!config.service_name.is_empty());
    }

    #[test]
    fn test_trace_id_functions() {
        // Without an active span, both should return None.
        assert!(current_trace_id().is_none());
        assert!(current_span_id().is_none());
    }

    #[test]
    fn from_telemetry_disabled_returns_none() {
        let t = crate::config::TelemetryConfig::default();
        assert!(!t.otlp.enabled);
        assert!(OtelConfig::from_telemetry(&t, "1.2.3").is_none());
    }

    #[test]
    fn from_telemetry_enabled_maps_fields() {
        let mut t = crate::config::TelemetryConfig::default();
        t.service_name = "daemon-test".to_string();
        t.otlp.enabled = true;
        t.otlp.endpoint = "http://collector.example:4317".to_string();
        t.otlp.protocol = crate::config::OtlpProtocol::HttpProtobuf;
        t.otlp.sample_rate = 0.25;
        t.otlp.headers.insert("x-trace".into(), "yes".into());

        let o = OtelConfig::from_telemetry(&t, "9.9.9").expect("enabled config maps");
        assert_eq!(o.service_name, "daemon-test");
        assert_eq!(o.service_version, "9.9.9");
        assert_eq!(
            o.otlp_endpoint.as_deref(),
            Some("http://collector.example:4317")
        );
        assert_eq!(o.otlp_protocol, "http/protobuf");
        assert_eq!(o.sampling_ratio, 0.25);
        assert_eq!(
            o.otlp_headers.get("x-trace").map(String::as_str),
            Some("yes")
        );
    }

    #[test]
    fn init_no_exporter_provider_succeeds() {
        let config = OtelConfig {
            otlp_endpoint: None,
            ..OtelConfig::default()
        };
        let provider = init_tracer_provider(&config).expect("no-op provider builds");
        // Flushing a no-op provider must not error.
        provider.shutdown().expect("shutdown no-op provider");
    }

    #[test]
    fn shutdown_without_provider_is_noop() {
        // Must not panic when no provider was ever installed.
        shutdown_tracer();
    }

    #[test]
    fn init_meter_provider_without_endpoint_returns_false() {
        // No endpoint => nothing to export => no global install (Ok(false)),
        // keeping the disabled path zero-overhead and the global meter untouched.
        let config = OtelConfig {
            otlp_endpoint: None,
            ..OtelConfig::default()
        };
        let installed =
            init_meter_provider(&config, Duration::from_secs(1)).expect("no-endpoint is Ok");
        assert!(!installed);
    }

    #[test]
    fn shutdown_meter_provider_without_provider_is_noop() {
        // Must not panic when no meter provider was ever installed.
        shutdown_meter_provider();
    }
}
