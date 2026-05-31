//! OTLP exporter construction for [`super`].
//!
//! Isolates the opentelemetry-otlp 0.31 builder API (gRPC via tonic,
//! HTTP/protobuf via reqwest), resource/sampler construction, programmatic
//! header injection, and the metrics-export scaffolding (kept disabled by
//! default for Phase 2, Task 88). Splitting this out keeps `tracing_otel.rs`
//! under the project's 500-line file limit.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry_otlp::{SpanExporter, WithExportConfig, WithHttpConfig, WithTonicConfig};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::Resource;

/// Error raised while building the OTLP exporter or provider.
///
/// Replaces the 0.21-era `opentelemetry::trace::TraceError` (removed in 0.31).
#[derive(Debug)]
pub enum OtelInitError {
    /// The exporter builder failed (transport setup, TLS, bad endpoint, ...).
    Exporter(opentelemetry_otlp::ExporterBuildError),
}

impl fmt::Display for OtelInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OtelInitError::Exporter(e) => write!(f, "OTLP exporter build failed: {e}"),
        }
    }
}

impl std::error::Error for OtelInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OtelInitError::Exporter(e) => Some(e),
        }
    }
}

impl From<opentelemetry_otlp::ExporterBuildError> for OtelInitError {
    fn from(e: opentelemetry_otlp::ExporterBuildError) -> Self {
        OtelInitError::Exporter(e)
    }
}

/// The selected OTLP wire transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtlpTransport {
    /// Protobuf over gRPC (tonic).
    Grpc,
    /// Protobuf over HTTP (`http/protobuf`, reqwest).
    HttpProtobuf,
}

impl OtlpTransport {
    /// Canonical lowercase wire-name used in logs and config.
    pub fn as_str(self) -> &'static str {
        match self {
            OtlpTransport::Grpc => "grpc",
            OtlpTransport::HttpProtobuf => "http/protobuf",
        }
    }
}

/// Map a configured protocol string onto an [`OtlpTransport`].
///
/// Unknown values fall back to gRPC (the historical default) with a warning;
/// a valid `http/protobuf` is honored (no silent gRPC fallback).
pub fn select_transport(protocol: &str) -> OtlpTransport {
    match protocol.trim() {
        "http/protobuf" | "http-protobuf" => OtlpTransport::HttpProtobuf,
        "grpc" => OtlpTransport::Grpc,
        other => {
            tracing::warn!(
                "Unknown OTLP protocol '{}'; falling back to gRPC transport",
                other
            );
            OtlpTransport::Grpc
        }
    }
}

/// Build the OTLP resource (service identity attributes).
pub fn build_resource(config: &super::OtelConfig) -> Resource {
    Resource::builder()
        .with_attributes([
            KeyValue::new("service.name", config.service_name.clone()),
            KeyValue::new("service.version", config.service_version.clone()),
        ])
        .build()
}

/// Build a sampler from a sampling ratio in `[0.0, 1.0]`.
pub fn build_sampler(ratio: f64) -> Sampler {
    if ratio >= 1.0 {
        Sampler::AlwaysOn
    } else if ratio <= 0.0 {
        Sampler::AlwaysOff
    } else {
        Sampler::TraceIdRatioBased(ratio)
    }
}

/// Build a span exporter for the selected transport with header injection.
///
/// The HTTP/protobuf path is genuinely wired (no silent gRPC fallback): when
/// `transport` is [`OtlpTransport::HttpProtobuf`] the reqwest-backed exporter
/// is used. Static headers from config are injected into the appropriate
/// transport (tonic metadata for gRPC, HTTP headers for the HTTP exporter).
pub fn build_span_exporter(
    transport: OtlpTransport,
    endpoint: &str,
    headers: &HashMap<String, String>,
) -> Result<SpanExporter, OtelInitError> {
    if !headers.is_empty() {
        tracing::info!(
            "Injecting {} static OTLP header(s) into the {} exporter",
            headers.len(),
            transport.as_str()
        );
    }
    let exporter = match transport {
        OtlpTransport::Grpc => SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint.to_string())
            .with_metadata(build_tonic_metadata(headers))
            .build()?,
        OtlpTransport::HttpProtobuf => SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint.to_string())
            .with_headers(headers.clone())
            .build()?,
    };
    Ok(exporter)
}

/// Convert config headers into tonic gRPC metadata for the OTLP exporter.
///
/// Entries whose names/values are not valid ASCII gRPC metadata are skipped
/// with a warning rather than aborting initialization.
fn build_tonic_metadata(headers: &HashMap<String, String>) -> tonic_otlp::metadata::MetadataMap {
    use tonic_otlp::metadata::{MetadataKey, MetadataMap, MetadataValue};

    let mut map = MetadataMap::with_capacity(headers.len());
    for (key, value) in headers {
        let key_lc = key.to_ascii_lowercase();
        match (
            MetadataKey::from_bytes(key_lc.as_bytes()),
            MetadataValue::try_from(value.as_str()),
        ) {
            (Ok(k), Ok(v)) => {
                map.insert(k, v);
            }
            _ => {
                tracing::warn!(
                    "Skipping invalid OTLP gRPC header '{}' (invalid metadata key or value)",
                    key
                );
            }
        }
    }
    map
}

/// Build an OTLP [`SdkMeterProvider`] for metrics export (Phase 2 scaffolding).
///
/// Wired but intentionally NOT installed by default: the Prometheus pull
/// endpoint remains the canonical metric surface for this daemon. Task 88
/// will gate this behind config and install it globally. Kept here so the
/// 0.31 metrics builder API is exercised and ready.
///
/// Returns `None` when no endpoint is configured.
#[allow(dead_code)]
pub fn build_meter_provider(
    config: &super::OtelConfig,
    interval: Duration,
) -> Result<Option<SdkMeterProvider>, OtelInitError> {
    let Some(endpoint) = config.otlp_endpoint.as_deref() else {
        return Ok(None);
    };
    let transport = select_transport(&config.otlp_protocol);
    let metric_exporter = match transport {
        OtlpTransport::Grpc => opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint.to_string())
            .with_metadata(build_tonic_metadata(&config.otlp_headers))
            .build()?,
        OtlpTransport::HttpProtobuf => opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_endpoint(endpoint.to_string())
            .with_headers(config.otlp_headers.clone())
            .build()?,
    };
    let reader = PeriodicReader::builder(metric_exporter)
        .with_interval(interval)
        .build();
    Ok(Some(
        SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(build_resource(config))
            .build(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_mapping() {
        assert_eq!(select_transport("grpc"), OtlpTransport::Grpc);
        assert_eq!(
            select_transport("http/protobuf"),
            OtlpTransport::HttpProtobuf
        );
        assert_eq!(
            select_transport("http-protobuf"),
            OtlpTransport::HttpProtobuf
        );
        // Unknown protocol falls back to gRPC.
        assert_eq!(select_transport("garbage"), OtlpTransport::Grpc);
    }

    #[test]
    fn transport_as_str_roundtrips() {
        assert_eq!(OtlpTransport::Grpc.as_str(), "grpc");
        assert_eq!(OtlpTransport::HttpProtobuf.as_str(), "http/protobuf");
    }

    #[test]
    fn sampler_bounds() {
        assert!(matches!(build_sampler(1.0), Sampler::AlwaysOn));
        assert!(matches!(build_sampler(2.0), Sampler::AlwaysOn));
        assert!(matches!(build_sampler(0.0), Sampler::AlwaysOff));
        assert!(matches!(build_sampler(-1.0), Sampler::AlwaysOff));
        assert!(matches!(
            build_sampler(0.5),
            Sampler::TraceIdRatioBased(r) if (r - 0.5).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn tonic_metadata_skips_invalid_headers() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer abc".to_string());
        // A control character (newline) is invalid in an HTTP/2 header value, so
        // tonic's MetadataValue rejects it and the entry must be skipped, not
        // panic. (Latin-1/obs-text bytes like 'ï' are actually *accepted* by
        // HeaderValue, so they would not exercise the skip path.)
        headers.insert("x-bad".to_string(), "bad\nvalue".to_string());
        let map = build_tonic_metadata(&headers);
        assert!(map.get("authorization").is_some());
        assert!(map.get("x-bad").is_none());
    }
}
