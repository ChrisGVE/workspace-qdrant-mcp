//! Telemetry, metrics, and OTLP/Prometheus export configuration

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

fn default_collection_interval() -> u64 {
    60
}
fn default_history_retention() -> usize {
    120
}
fn default_telemetry_enabled() -> bool {
    true
}

/// Observability configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Collection interval in seconds
    #[serde(default = "default_collection_interval")]
    pub collection_interval: u64,

    /// Basic metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Detailed telemetry configuration
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            collection_interval: default_collection_interval(),
            metrics: MetricsConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

impl ObservabilityConfig {
    /// Validate configuration settings.
    ///
    /// `collection_interval` must be in the range [1, 86400] seconds (1 s – 24 h).
    /// Chains telemetry subconfig validation.
    pub fn validate(&self) -> Result<(), String> {
        if self.collection_interval == 0 {
            return Err("collection_interval must be at least 1 second".to_string());
        }
        if self.collection_interval > 86_400 {
            return Err("collection_interval must not exceed 86400 seconds (24 hours)".to_string());
        }
        self.telemetry
            .validate()
            .map_err(|e| format!("telemetry: {e}"))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsConfig {
    #[serde(default)]
    pub enabled: bool,
}

fn default_service_name() -> String {
    "memexd".to_string()
}

fn default_prometheus_port() -> u16 {
    6337
}

fn default_prometheus_bind() -> String {
    "0.0.0.0".to_string()
}

fn default_otlp_endpoint() -> String {
    "http://localhost:4318".to_string()
}

fn default_otlp_protocol() -> OtlpProtocol {
    OtlpProtocol::HttpProtobuf
}

fn default_otlp_sample_rate() -> f64 {
    1.0
}

fn default_trace_tier() -> String {
    "off".to_string()
}

fn default_attribute_cardinality_cap() -> usize {
    40
}

/// Runtime tracing controls (PRD B4). Governs the trace cost-gate tier, hot-path
/// instrumentation, and per-span attribute cardinality. Independent of OTLP
/// export (`otlp`): these bound span/attribute *construction* cost on hot paths,
/// the OTLP sampler bounds *export*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Trace verbosity tier: `off` | `hot` | `full` (default `off`). Maps to
    /// [`crate::tracing_gate::TraceTier`]; consult [`TracingConfig::effective_tier`].
    #[serde(default = "default_trace_tier")]
    pub tier: String,

    /// Whether `#[instrument]`-style spans on hot paths are constructed. The
    /// per-path gate ([`crate::tracing_gate::if_tier`]) keys off `tier`; this is
    /// an additional global opt-out for hot-path span work.
    #[serde(default)]
    pub instrument_hot_paths: bool,

    /// Upper bound on distinct attribute values recorded per span dimension, to
    /// cap trace-backend cardinality (default 40).
    #[serde(default = "default_attribute_cardinality_cap")]
    pub attribute_cardinality_cap: usize,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            tier: default_trace_tier(),
            instrument_hot_paths: false,
            attribute_cardinality_cap: default_attribute_cardinality_cap(),
        }
    }
}

impl TracingConfig {
    /// Parse `tier` into the runtime [`TraceTier`](crate::tracing_gate::TraceTier),
    /// falling back to [`TraceTier::Off`](crate::tracing_gate::TraceTier::Off) for
    /// an unrecognized value. Call [`TracingConfig::validate`] first to surface a
    /// bad tier as a config error rather than a silent fallback.
    pub fn effective_tier(&self) -> crate::tracing_gate::TraceTier {
        crate::tracing_gate::TraceTier::parse(&self.tier)
            .unwrap_or(crate::tracing_gate::TraceTier::Off)
    }

    pub fn validate(&self) -> Result<(), String> {
        if crate::tracing_gate::TraceTier::parse(&self.tier).is_none() {
            return Err(format!(
                "telemetry.tracing.tier must be one of off|hot|full, got {:?}",
                self.tier
            ));
        }
        if self.attribute_cardinality_cap == 0 {
            return Err(
                "telemetry.tracing.attribute_cardinality_cap must be at least 1".to_string(),
            );
        }
        Ok(())
    }
}

/// OTLP wire protocol. Matches the `OTEL_EXPORTER_OTLP_PROTOCOL`
/// environment variable convention (lowercase with slashes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OtlpProtocol {
    /// Protobuf-over-HTTP (serde tag: `http/protobuf`). Default.
    #[serde(rename = "http/protobuf")]
    HttpProtobuf,
    /// Protobuf-over-gRPC (serde tag: `grpc`).
    #[serde(rename = "grpc")]
    Grpc,
}

impl OtlpProtocol {
    pub fn as_str(&self) -> &'static str {
        match self {
            OtlpProtocol::HttpProtobuf => "http/protobuf",
            OtlpProtocol::Grpc => "grpc",
        }
    }

    /// Parse from the canonical OTLP string representation used in
    /// `OTEL_EXPORTER_OTLP_PROTOCOL` and YAML config.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "http/protobuf" | "http-protobuf" => Some(OtlpProtocol::HttpProtobuf),
            "grpc" => Some(OtlpProtocol::Grpc),
            _ => None,
        }
    }
}

/// Prometheus HTTP endpoint configuration.
///
/// When enabled, the daemon serves the Prometheus text exposition format
/// at `GET /metrics` and `200 OK` at `GET /health`, both bound to
/// `{bind}:{port}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusExportConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_prometheus_port")]
    pub port: u16,

    #[serde(default = "default_prometheus_bind")]
    pub bind: String,
}

impl Default for PrometheusExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: default_prometheus_port(),
            bind: default_prometheus_bind(),
        }
    }
}

impl PrometheusExportConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled && self.port == 0 {
            return Err("telemetry.prometheus.port must be non-zero when enabled".to_string());
        }
        if self.bind.trim().is_empty() {
            return Err("telemetry.prometheus.bind must not be empty".to_string());
        }
        Ok(())
    }
}

/// OTLP push exporter configuration for traces (and optionally metrics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpExportConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Enable the additive OTLP **metrics** push path (Task 88, Phase 2).
    ///
    /// Independent of `enabled` (which gates traces) so an operator can ship
    /// traces over OTLP without also pushing metrics. Metrics push requires
    /// `enabled` to be true as well (the OTLP endpoint is shared). The
    /// Prometheus pull endpoint remains the primary metric transport
    /// regardless; this path is additive (exemplars, future multi-host).
    #[serde(default)]
    pub metrics_enabled: bool,

    #[serde(default = "default_otlp_endpoint")]
    pub endpoint: String,

    #[serde(default = "default_otlp_protocol")]
    pub protocol: OtlpProtocol,

    #[serde(default = "default_otlp_sample_rate")]
    pub sample_rate: f64,

    #[serde(default)]
    pub headers: HashMap<String, String>,
}

impl Default for OtlpExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            metrics_enabled: false,
            endpoint: default_otlp_endpoint(),
            protocol: default_otlp_protocol(),
            sample_rate: default_otlp_sample_rate(),
            headers: HashMap::new(),
        }
    }
}

impl OtlpExportConfig {
    /// Whether the additive OTLP metrics push path should be installed.
    ///
    /// Requires both `enabled` (OTLP on, shared endpoint) and `metrics_enabled`
    /// (operator opted into the metrics push). Traces are gated by `enabled`
    /// alone; metrics need the extra opt-in (Task 88).
    pub fn metrics_export_enabled(&self) -> bool {
        self.enabled && self.metrics_enabled
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.sample_rate) {
            return Err(format!(
                "telemetry.otlp.sample_rate must be within [0.0, 1.0], got {}",
                self.sample_rate
            ));
        }
        if self.enabled && self.endpoint.trim().is_empty() {
            return Err("telemetry.otlp.endpoint must not be empty when enabled".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_history_retention")]
    pub history_retention: usize,

    #[serde(default = "default_telemetry_enabled")]
    pub cpu_usage: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub memory_usage: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub latency: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub queue_depth: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub throughput: bool,

    /// OpenTelemetry `service.name` resource attribute. Used by both the
    /// OTLP exporter resource and as the `job` label on Prometheus scrapes
    /// when self-labeling is configured.
    #[serde(default = "default_service_name")]
    pub service_name: String,

    /// Prometheus pull-based metrics exposition.
    #[serde(default)]
    pub prometheus: PrometheusExportConfig,

    /// OTLP push exporter for traces (and optionally metrics).
    #[serde(default)]
    pub otlp: OtlpExportConfig,

    /// Runtime tracing controls (cost-gate tier, hot-path instrumentation,
    /// attribute cardinality cap).
    #[serde(default)]
    pub tracing: TracingConfig,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            history_retention: default_history_retention(),
            cpu_usage: default_telemetry_enabled(),
            memory_usage: default_telemetry_enabled(),
            latency: default_telemetry_enabled(),
            queue_depth: default_telemetry_enabled(),
            throughput: default_telemetry_enabled(),
            service_name: default_service_name(),
            prometheus: PrometheusExportConfig::default(),
            otlp: OtlpExportConfig::default(),
            tracing: TracingConfig::default(),
        }
    }
}

impl TelemetryConfig {
    /// Apply environment variable overrides matching the OTEL_* conventions
    /// used by the TypeScript MCP server (for consistency across services).
    ///
    /// Precedence (highest → lowest):
    ///   1. Environment variables (applied here)
    ///   2. YAML configuration
    ///   3. Compiled-in defaults
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("OTEL_SERVICE_NAME") {
            if !val.trim().is_empty() {
                self.service_name = val;
            }
        }

        if let Ok(val) = env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            if !val.trim().is_empty() {
                // Set the endpoint only. Unlike the OTel CLI convention, a bare
                // endpoint does NOT enable OTLP export: it stays opt-in via
                // `telemetry.otlp.enabled` (YAML) or the explicit WQM_OTLP_ENABLED
                // / WQM_OTLP_METRICS_ENABLED env gates. This prevents a stray or
                // misconfigured endpoint env (e.g. an unreachable collector, or
                // http/protobuf pointed at the gRPC port) from silently
                // activating a failing exporter whose BatchSpanProcessor then
                // floods the logs with "Spans emitted after Shutdown" (#85).
                self.otlp.endpoint = val;
            }
        }

        // Explicit opt-in for OTLP export. Mirrors the metrics gate below and the
        // YAML `otlp.enabled` flag; keeps OTLP off-by-default unless the operator
        // asks for it, independent of whether an endpoint env is present (#85).
        if let Ok(val) = env::var("WQM_OTLP_ENABLED") {
            match val.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => self.otlp.enabled = true,
                "0" | "false" | "no" | "off" => self.otlp.enabled = false,
                _ => {}
            }
        }

        if let Ok(val) = env::var("OTEL_EXPORTER_OTLP_PROTOCOL") {
            if let Some(protocol) = OtlpProtocol::parse(&val) {
                self.otlp.protocol = protocol;
            }
        }

        // Opt into the additive OTLP metrics push path (Task 88). Truthy values
        // also imply OTLP itself is on, since the endpoint is shared.
        if let Ok(val) = env::var("WQM_OTLP_METRICS_ENABLED") {
            match val.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => {
                    self.otlp.metrics_enabled = true;
                    self.otlp.enabled = true;
                }
                "0" | "false" | "no" | "off" => self.otlp.metrics_enabled = false,
                _ => {}
            }
        }

        self.apply_otlp_header_overrides();
        self.apply_sample_rate_override();
        self.apply_prometheus_overrides();
        self.apply_tracing_overrides();
    }

    fn apply_tracing_overrides(&mut self) {
        // Trace cost-gate tier (B1/B4). WQM_TRACE_TIER overrides the YAML/default
        // tier; an unparseable value is left for validate() to reject rather
        // than silently dropped, keeping env > YAML > default precedence.
        if let Ok(val) = std::env::var(crate::tracing_gate::TRACE_TIER_ENV) {
            if !val.trim().is_empty() {
                self.tracing.tier = val.trim().to_string();
            }
        }
    }

    fn apply_otlp_header_overrides(&mut self) {
        if let Ok(val) = std::env::var("OTEL_EXPORTER_OTLP_HEADERS") {
            for kv in val.split(',') {
                let kv = kv.trim();
                if kv.is_empty() {
                    continue;
                }
                if let Some((k, v)) = kv.split_once('=') {
                    self.otlp
                        .headers
                        .insert(k.trim().to_string(), v.trim().to_string());
                }
            }
        }
    }

    fn apply_sample_rate_override(&mut self) {
        if let Ok(val) = std::env::var("OTEL_TRACES_SAMPLER_ARG") {
            if let Ok(parsed) = val.parse::<f64>() {
                if (0.0..=1.0).contains(&parsed) {
                    self.otlp.sample_rate = parsed;
                }
            }
        }
    }

    fn apply_prometheus_overrides(&mut self) {
        // Prometheus endpoint overrides (WQM_-prefixed to avoid colliding
        // with OTel conventions that don't cover pull-based exposition).
        if let Ok(val) = std::env::var("WQM_PROMETHEUS_ENABLED") {
            self.prometheus.enabled = matches!(val.to_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Ok(val) = std::env::var("WQM_PROMETHEUS_PORT") {
            if let Ok(port) = val.parse::<u16>() {
                self.prometheus.port = port;
            }
        }
        // WQM_METRICS_PORT is the canonical metrics-port override (E1, §12 Q5);
        // applied last so it wins over the legacy WQM_PROMETHEUS_PORT alias.
        if let Ok(val) = std::env::var("WQM_METRICS_PORT") {
            if let Ok(port) = val.parse::<u16>() {
                self.prometheus.port = port;
            }
        }
        if let Ok(val) = std::env::var("WQM_PROMETHEUS_BIND") {
            if !val.trim().is_empty() {
                self.prometheus.bind = val;
            }
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.service_name.trim().is_empty() {
            return Err("telemetry.service_name must not be empty".to_string());
        }
        self.prometheus.validate()?;
        self.otlp.validate()?;
        self.tracing.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::observability::monitoring::MonitoringConfig;

    #[test]
    fn test_monitoring_config_defaults() {
        let config = MonitoringConfig::default();
        assert_eq!(config.check_interval_hours, 24);
        assert!(config.check_on_startup);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_monitoring_config_validation() {
        let mut config = MonitoringConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid check_interval_hours
        config.check_interval_hours = 0;
        assert!(config.validate().is_err());
        config.check_interval_hours = 8761;
        assert!(config.validate().is_err());
        config.check_interval_hours = 24;

        // Valid again
        assert!(config.validate().is_ok());
    }

    // ── TelemetryConfig export-surface tests ────────────────────────────
    //
    // The env_overrides tests mutate process-wide environment state, so
    // they share a `serial_test` group to serialize access.

    use serial_test::serial;

    const OTEL_ENV_VARS: &[&str] = &[
        "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_TRACES_SAMPLER_ARG",
        "WQM_PROMETHEUS_ENABLED",
        "WQM_PROMETHEUS_PORT",
        "WQM_PROMETHEUS_BIND",
        "WQM_METRICS_PORT",
        "WQM_TRACE_TIER",
        "WQM_OTLP_METRICS_ENABLED",
        "WQM_OTLP_ENABLED",
    ];

    struct TelemetryEnvGuard;
    impl Drop for TelemetryEnvGuard {
        fn drop(&mut self) {
            for key in OTEL_ENV_VARS {
                std::env::remove_var(key);
            }
        }
    }

    #[test]
    fn test_telemetry_defaults_are_opt_in() {
        let t = TelemetryConfig::default();
        assert!(!t.enabled);
        assert!(!t.prometheus.enabled);
        assert_eq!(t.prometheus.port, 6337);
        assert_eq!(t.prometheus.bind, "0.0.0.0");
        assert!(!t.otlp.enabled);
        assert_eq!(t.otlp.endpoint, "http://localhost:4318");
        assert_eq!(t.otlp.protocol, OtlpProtocol::HttpProtobuf);
        assert!((t.otlp.sample_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(t.service_name, "memexd");
    }

    #[test]
    fn test_telemetry_defaults_validate() {
        TelemetryConfig::default().validate().unwrap();
    }

    #[test]
    fn test_telemetry_validation_rejects_invalid_sample_rate() {
        let mut t = TelemetryConfig::default();
        t.otlp.sample_rate = 1.5;
        assert!(t.validate().is_err());
        t.otlp.sample_rate = -0.1;
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_telemetry_validation_rejects_empty_service_name() {
        let mut t = TelemetryConfig::default();
        t.service_name = "   ".to_string();
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_telemetry_validation_rejects_enabled_prometheus_on_port_zero() {
        let mut t = TelemetryConfig::default();
        t.prometheus.enabled = true;
        t.prometheus.port = 0;
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_telemetry_validation_rejects_enabled_otlp_with_empty_endpoint() {
        let mut t = TelemetryConfig::default();
        t.otlp.enabled = true;
        t.otlp.endpoint = "".to_string();
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_otlp_metrics_disabled_by_default() {
        let t = TelemetryConfig::default();
        assert!(!t.otlp.metrics_enabled);
        // Even with traces on, metrics export stays off until opted in.
        assert!(!t.otlp.metrics_export_enabled());
    }

    #[test]
    fn test_otlp_metrics_export_enabled_requires_both_flags() {
        let mut t = TelemetryConfig::default();
        t.otlp.metrics_enabled = true;
        // metrics_enabled alone is insufficient — otlp.enabled gates the shared endpoint.
        assert!(!t.otlp.metrics_export_enabled());
        t.otlp.enabled = true;
        assert!(t.otlp.metrics_export_enabled());
    }

    #[test]
    #[serial]
    fn test_env_override_otlp_metrics_enabled() {
        let _g = TelemetryEnvGuard;
        std::env::set_var("WQM_OTLP_METRICS_ENABLED", "true");

        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();

        assert!(t.otlp.metrics_enabled);
        // Enabling metrics push implies OTLP itself is on (shared endpoint).
        assert!(t.otlp.enabled);
        assert!(t.otlp.metrics_export_enabled());
    }

    #[test]
    fn test_otlp_protocol_parse() {
        assert_eq!(
            OtlpProtocol::parse("http/protobuf"),
            Some(OtlpProtocol::HttpProtobuf)
        );
        assert_eq!(OtlpProtocol::parse("grpc"), Some(OtlpProtocol::Grpc));
        assert_eq!(OtlpProtocol::parse("unknown"), None);
    }

    #[test]
    #[serial]
    fn test_env_overrides_service_name_and_endpoint() {
        let _g = TelemetryEnvGuard;
        std::env::set_var("OTEL_SERVICE_NAME", "custom-service");
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318");

        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();

        assert_eq!(t.service_name, "custom-service");
        assert_eq!(t.otlp.endpoint, "http://collector:4318");
        // A bare endpoint env must NOT enable OTLP — export stays opt-in so a
        // stray/misconfigured endpoint can't activate a failing exporter (#85).
        assert!(!t.otlp.enabled);
    }

    #[test]
    #[serial]
    fn test_wqm_otlp_enabled_is_explicit_gate() {
        let _g = TelemetryEnvGuard;

        // Endpoint present but no explicit enable → stays disabled (#85).
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318");
        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();
        assert!(!t.otlp.enabled);

        // Explicit opt-in enables it.
        std::env::set_var("WQM_OTLP_ENABLED", "true");
        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();
        assert!(t.otlp.enabled);

        // Explicit opt-out wins over a truthy metrics flag is not asserted here;
        // a plain "off" disables.
        std::env::set_var("WQM_OTLP_ENABLED", "off");
        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();
        assert!(!t.otlp.enabled);
    }

    #[test]
    #[serial]
    fn test_env_overrides_protocol_and_headers() {
        let _g = TelemetryEnvGuard;
        std::env::set_var("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc");
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "authorization=Bearer abc,x-tenant=acme",
        );

        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();

        assert_eq!(t.otlp.protocol, OtlpProtocol::Grpc);
        assert_eq!(
            t.otlp.headers.get("authorization"),
            Some(&"Bearer abc".to_string())
        );
        assert_eq!(t.otlp.headers.get("x-tenant"), Some(&"acme".to_string()));
    }

    #[test]
    #[serial]
    fn test_env_overrides_sample_rate_bounds() {
        let _g = TelemetryEnvGuard;
        let mut t = TelemetryConfig::default();

        std::env::set_var("OTEL_TRACES_SAMPLER_ARG", "0.25");
        t.apply_env_overrides();
        assert!((t.otlp.sample_rate - 0.25).abs() < 1e-9);

        // Out-of-range values are silently ignored so a misconfigured env
        // can't poison the validated config.
        std::env::set_var("OTEL_TRACES_SAMPLER_ARG", "2.0");
        t.apply_env_overrides();
        assert!((t.otlp.sample_rate - 0.25).abs() < 1e-9);
    }

    #[test]
    #[serial]
    fn test_env_overrides_prometheus_toggle_and_port() {
        let _g = TelemetryEnvGuard;
        std::env::set_var("WQM_PROMETHEUS_ENABLED", "true");
        std::env::set_var("WQM_PROMETHEUS_PORT", "9999");
        std::env::set_var("WQM_PROMETHEUS_BIND", "127.0.0.1");

        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();

        assert!(t.prometheus.enabled);
        assert_eq!(t.prometheus.port, 9999);
        assert_eq!(t.prometheus.bind, "127.0.0.1");
    }

    #[test]
    #[serial]
    fn test_env_overrides_do_not_clobber_unset_fields() {
        let _g = TelemetryEnvGuard;
        let mut t = TelemetryConfig::default();
        t.otlp.endpoint = "http://set-in-yaml:4318".to_string();
        t.otlp.sample_rate = 0.5;

        // Nothing in env → YAML values survive.
        t.apply_env_overrides();
        assert_eq!(t.otlp.endpoint, "http://set-in-yaml:4318");
        assert!((t.otlp.sample_rate - 0.5).abs() < 1e-9);
    }

    // ── E1: canonical metrics port ──────────────────────────────────────

    #[test]
    fn test_metrics_port_default_is_canonical_6337() {
        assert_eq!(TelemetryConfig::default().prometheus.port, 6337);
    }

    #[test]
    #[serial]
    fn test_env_override_metrics_port_precedence() {
        let _g = TelemetryEnvGuard;
        // WQM_METRICS_PORT is canonical and wins over WQM_PROMETHEUS_PORT.
        std::env::set_var("WQM_PROMETHEUS_PORT", "8888");
        std::env::set_var("WQM_METRICS_PORT", "9999");

        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();
        assert_eq!(t.prometheus.port, 9999);
    }

    #[test]
    #[serial]
    fn test_env_override_metrics_port_alone() {
        let _g = TelemetryEnvGuard;
        std::env::set_var("WQM_METRICS_PORT", "7777");
        let mut t = TelemetryConfig::default();
        t.apply_env_overrides();
        assert_eq!(t.prometheus.port, 7777);
    }

    // ── TracingConfig (B4) ──────────────────────────────────────────────

    #[test]
    fn test_tracing_defaults_are_off() {
        let t = TelemetryConfig::default();
        assert_eq!(t.tracing.tier, "off");
        assert!(!t.tracing.instrument_hot_paths);
        assert_eq!(t.tracing.attribute_cardinality_cap, 40);
        assert_eq!(
            t.tracing.effective_tier(),
            crate::tracing_gate::TraceTier::Off
        );
    }

    #[test]
    fn test_tracing_defaults_validate() {
        TelemetryConfig::default().validate().unwrap();
    }

    #[test]
    fn test_tracing_validation_rejects_unknown_tier() {
        let mut t = TelemetryConfig::default();
        t.tracing.tier = "loud".to_string();
        let err = t.validate().unwrap_err();
        assert!(err.contains("tier"), "error should mention tier: {err}");
    }

    #[test]
    fn test_tracing_validation_rejects_zero_cardinality_cap() {
        let mut t = TelemetryConfig::default();
        t.tracing.attribute_cardinality_cap = 0;
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_tracing_effective_tier_maps_values() {
        use crate::tracing_gate::TraceTier;
        let mut t = TelemetryConfig::default();
        t.tracing.tier = "hot".to_string();
        assert_eq!(t.tracing.effective_tier(), TraceTier::Hot);
        t.tracing.tier = "FULL".to_string();
        assert_eq!(t.tracing.effective_tier(), TraceTier::Full);
    }

    #[test]
    #[serial]
    fn test_env_override_trace_tier_precedence() {
        let _g = TelemetryEnvGuard;
        // YAML/default value present; env should win (env > YAML > default).
        let mut t = TelemetryConfig::default();
        t.tracing.tier = "hot".to_string();

        std::env::set_var("WQM_TRACE_TIER", "full");
        t.apply_env_overrides();
        assert_eq!(t.tracing.tier, "full");
        assert_eq!(
            t.tracing.effective_tier(),
            crate::tracing_gate::TraceTier::Full
        );
    }

    #[test]
    #[serial]
    fn test_env_override_trace_tier_unset_keeps_yaml() {
        let _g = TelemetryEnvGuard;
        let mut t = TelemetryConfig::default();
        t.tracing.tier = "hot".to_string();

        // No WQM_TRACE_TIER in env → YAML value survives.
        t.apply_env_overrides();
        assert_eq!(t.tracing.tier, "hot");
    }

    // ── ObservabilityConfig::validate ────────────────────────────────────────

    #[test]
    fn test_observability_config_validate_default_ok() {
        assert!(ObservabilityConfig::default().validate().is_ok());
    }

    #[test]
    fn test_observability_config_validate_rejects_zero_interval() {
        let config = ObservabilityConfig {
            collection_interval: 0,
            ..ObservabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_observability_config_validate_rejects_over_86400() {
        let config = ObservabilityConfig {
            collection_interval: 86_401,
            ..ObservabilityConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_observability_config_validate_accepts_boundary_values() {
        let min = ObservabilityConfig {
            collection_interval: 1,
            ..ObservabilityConfig::default()
        };
        assert!(min.validate().is_ok());

        let max = ObservabilityConfig {
            collection_interval: 86_400,
            ..ObservabilityConfig::default()
        };
        assert!(max.validate().is_ok());
    }

    #[test]
    fn test_observability_config_validate_chains_telemetry() {
        let mut config = ObservabilityConfig::default();
        config.telemetry.service_name = "  ".to_string(); // whitespace → invalid
        let err = config.validate().unwrap_err();
        assert!(
            err.contains("telemetry:"),
            "error should be prefixed with 'telemetry:': {err}"
        );
    }
}
