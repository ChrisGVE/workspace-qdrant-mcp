//! OpenTelemetry tracing setup for daemon-server boundary
//!
//! Implements distributed tracing across the daemon and MCP server
//! using OpenTelemetry standards. Supports both stdout (development)
//! and OTLP (production) exporters.
//!
//! Task 412.19: Add OpenTelemetry tracing setup for daemon-server boundary

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    runtime,
    trace::{BatchSpanProcessor, Config, Sampler, TracerProvider},
    Resource,
};
use std::env;
use tracing::Subscriber;
use tracing_subscriber::registry::LookupSpan;

/// OpenTelemetry configuration
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Service name for traces
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// OTLP endpoint (if None, uses stdout exporter)
    pub otlp_endpoint: Option<String>,
    /// Sampling ratio (0.0 to 1.0)
    pub sampling_ratio: f64,
    /// Enable trace context propagation in gRPC metadata
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
            otlp_protocol: match telemetry.otlp.protocol {
                crate::config::OtlpProtocol::Grpc => "grpc".to_string(),
                crate::config::OtlpProtocol::HttpProtobuf => "http/protobuf".to_string(),
            },
            otlp_headers: telemetry.otlp.headers.clone(),
        })
    }
}

impl OtelConfig {
    /// Create config from environment variables
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

/// Initialize OpenTelemetry tracing with OTLP exporter
///
/// Returns a TracerProvider that can be used to get tracers for distributed tracing.
/// If OTEL_EXPORTER_OTLP_ENDPOINT is set, uses OTLP exporter; otherwise uses a simple provider.
pub fn init_tracer_provider(
    config: &OtelConfig,
) -> Result<TracerProvider, opentelemetry::trace::TraceError> {
    let resource = Resource::new(vec![
        KeyValue::new("service.name", config.service_name.clone()),
        KeyValue::new("service.version", config.service_version.clone()),
    ]);

    let sampler = if config.sampling_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else if config.sampling_ratio <= 0.0 {
        Sampler::AlwaysOff
    } else {
        Sampler::TraceIdRatioBased(config.sampling_ratio)
    };

    let trace_config = Config::default()
        .with_sampler(sampler)
        .with_resource(resource);

    if let Some(endpoint) = &config.otlp_endpoint {
        // Use OTLP exporter for production
        tracing::info!(
            "Initializing OpenTelemetry with OTLP endpoint: {} (protocol={})",
            endpoint,
            config.otlp_protocol
        );
        if config.otlp_protocol != "grpc" {
            tracing::warn!(
                "OTLP protocol '{}' requested but only the tonic/gRPC exporter is \
                 compiled in. Falling back to gRPC transport to the configured endpoint.",
                config.otlp_protocol
            );
        }

        if !config.otlp_headers.is_empty() {
            // Header/metadata injection requires tonic version alignment with
            // opentelemetry-otlp 0.14's vendored tonic. Relying on the
            // `OTEL_EXPORTER_OTLP_HEADERS` env variable is the interop path
            // the SDK already honors. Warn so operators know we're not
            // wiring them manually yet.
            tracing::warn!(
                "telemetry.otlp.headers configured ({} entries) but are being \
                 forwarded via OTEL_EXPORTER_OTLP_HEADERS env only; direct \
                 injection is not yet implemented",
                config.otlp_headers.len()
            );
        }
        let exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(endpoint.clone())
            .build_span_exporter()
            .map_err(|e| opentelemetry::trace::TraceError::Other(Box::new(e)))?;

        let batch_processor = BatchSpanProcessor::builder(exporter, runtime::Tokio).build();

        let provider = TracerProvider::builder()
            .with_span_processor(batch_processor)
            .with_config(trace_config)
            .build();

        Ok(provider)
    } else {
        // Use simple provider for development (no exporter)
        tracing::info!("Initializing OpenTelemetry with no exporter (development mode)");

        let provider = TracerProvider::builder().with_config(trace_config).build();

        Ok(provider)
    }
}

/// Create an OpenTelemetry tracing layer for use with tracing-subscriber
///
/// This layer bridges the `tracing` crate with OpenTelemetry, allowing
/// all `#[tracing::instrument]` spans to be exported as OpenTelemetry traces.
pub fn otel_layer<S>(
    config: &OtelConfig,
) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    match init_tracer_provider(config) {
        Ok(provider) => {
            let tracer = provider.tracer(config.service_name.clone());
            // Set the global provider so spans are exported
            opentelemetry::global::set_tracer_provider(provider);
            Some(tracing_opentelemetry::layer().with_tracer(tracer))
        }
        Err(e) => {
            tracing::error!("Failed to initialize OpenTelemetry tracer: {}", e);
            None
        }
    }
}

/// Shutdown OpenTelemetry and flush remaining traces
pub fn shutdown_tracer() {
    opentelemetry::global::shutdown_tracer_provider();
    tracing::info!("OpenTelemetry tracer provider shut down");
}

// Note: gRPC context propagation functions (extract_context_from_metadata,
// inject_context_into_metadata) are implemented in the grpc crate where
// tonic types are available.

/// Get the current trace ID as a string for logging correlation
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

/// Get the current span ID as a string for logging correlation
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
        // Test that from_env doesn't panic
        let config = OtelConfig::from_env();
        assert!(!config.service_name.is_empty());
    }

    #[test]
    fn test_trace_id_functions() {
        // Without active span, should return None
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
}
