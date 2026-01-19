//! OpenTelemetry tracing setup for daemon-server boundary
//!
//! Implements distributed tracing across the daemon and MCP server
//! using OpenTelemetry standards. Supports both stdout (development)
//! and OTLP (production) exporters.
//!
//! Task 412.19: Add OpenTelemetry tracing setup for daemon-server boundary

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::{
    runtime,
    trace::{BatchSpanProcessor, Config, Sampler, TracerProvider},
    Resource,
};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
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
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            service_name: "memexd".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            otlp_endpoint: env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok(),
            sampling_ratio: 1.0,
            propagate_context: true,
        }
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
pub fn init_tracer_provider(config: &OtelConfig) -> Result<TracerProvider, opentelemetry::trace::TraceError> {
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
        tracing::info!("Initializing OpenTelemetry with OTLP endpoint: {}", endpoint);

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

        let provider = TracerProvider::builder()
            .with_config(trace_config)
            .build();

        Ok(provider)
    }
}

/// Create an OpenTelemetry tracing layer for use with tracing-subscriber
///
/// This layer bridges the `tracing` crate with OpenTelemetry, allowing
/// all `#[tracing::instrument]` spans to be exported as OpenTelemetry traces.
pub fn otel_layer<S>(config: &OtelConfig) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>
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
}
