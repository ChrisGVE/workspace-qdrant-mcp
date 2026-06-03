//! Environment-variable overrides for [`DaemonConfig`] (WI-a2).
//!
//! Expressed declaratively via the shared engine
//! ([`wqm_common::config::EnvOverride`] + [`wqm_common::config::apply_env_overrides`])
//! so the daemon shares one override mechanism with the CLI and MCP server.
//!
//! All variables use the `WORKSPACE_QDRANT_` prefix. Invalid numeric/bool
//! values are ignored (the file/default value is kept) — consistent with the
//! CLI and MCP adoptions of the same engine.

use std::path::PathBuf;

use wqm_common::config::{apply_env_overrides as apply_shared, EnvGetter, EnvOverride};

use crate::config::DaemonConfig;
use crate::storage::TransportMode;

const ENV_PREFIX: &str = "WORKSPACE_QDRANT_";

/// Build the prefixed env-var name.
fn var(name: &str) -> String {
    format!("{ENV_PREFIX}{name}")
}

/// Apply all `WORKSPACE_QDRANT_*` overrides plus the telemetry `OTEL_*` /
/// `WQM_PROMETHEUS_*` overrides, reading from the process environment.
pub fn apply_env_overrides(config: DaemonConfig) -> DaemonConfig {
    apply_env_overrides_with(config, &|key| std::env::var(key).ok())
}

/// Apply the overrides using an injected environment getter (testable without
/// mutating process-global env vars).
pub(crate) fn apply_env_overrides_with(
    mut config: DaemonConfig,
    getter: &EnvGetter,
) -> DaemonConfig {
    let specs = build_specs();
    apply_shared(&mut config, getter, &specs);
    // Telemetry export settings follow OTEL_* / WQM_PROMETHEUS_* conventions;
    // the logic lives on the config struct so tests and external callers can
    // reuse it without depending on this module.
    config.observability.telemetry.apply_env_overrides();
    config
}

/// The declarative override spec list. Order is irrelevant here — each spec
/// targets a distinct field with a single source var.
fn build_specs() -> Vec<EnvOverride<DaemonConfig>> {
    vec![
        // ── General ─────────────────────────────────────────────────────────
        EnvOverride::single(var("LOG_FILE"), |c: &mut DaemonConfig, val| {
            c.log_file = Some(PathBuf::from(val));
        }),
        EnvOverride::single(var("MAX_CONCURRENT_TASKS"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.max_concurrent_tasks = Some(n);
            }
        }),
        EnvOverride::single(var("DEFAULT_TIMEOUT_MS"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.default_timeout_ms = Some(n);
            }
        }),
        EnvOverride::single(var("ENABLE_PREEMPTION"), |c: &mut DaemonConfig, val| {
            if let Ok(b) = val.parse() {
                c.enable_preemption = b;
            }
        }),
        EnvOverride::single(var("CHUNK_SIZE"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.chunk_size = n;
            }
        }),
        EnvOverride::single(var("LOG_LEVEL"), |c: &mut DaemonConfig, val| {
            c.log_level = val;
        }),
        EnvOverride::single(var("ENABLE_METRICS"), |c: &mut DaemonConfig, val| {
            if let Ok(b) = val.parse() {
                c.observability.metrics.enabled = b;
            }
        }),
        EnvOverride::single(var("METRICS_INTERVAL_SECS"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.observability.collection_interval = n;
            }
        }),
        // ── Qdrant ──────────────────────────────────────────────────────────
        EnvOverride::single(var("QDRANT__URL"), |c: &mut DaemonConfig, val| {
            c.qdrant.url = val;
        }),
        EnvOverride::single(
            var("QDRANT__TRANSPORT"),
            |c: &mut DaemonConfig, val| match val.to_lowercase().as_str() {
                "grpc" => c.qdrant.transport = TransportMode::Grpc,
                "http" => c.qdrant.transport = TransportMode::Http,
                _ => {}
            },
        ),
        EnvOverride::single(var("QDRANT__TIMEOUT_MS"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.qdrant.timeout_ms = n;
            }
        }),
        EnvOverride::single(var("QDRANT__MAX_RETRIES"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.qdrant.max_retries = n;
            }
        }),
        EnvOverride::single(
            var("QDRANT__RETRY_DELAY_MS"),
            |c: &mut DaemonConfig, val| {
                if let Ok(n) = val.parse() {
                    c.qdrant.retry_delay_ms = n;
                }
            },
        ),
        EnvOverride::single(var("QDRANT__POOL_SIZE"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.qdrant.pool_size = n;
            }
        }),
        EnvOverride::single(var("QDRANT__TLS"), |c: &mut DaemonConfig, val| {
            if let Ok(b) = val.parse() {
                c.qdrant.tls = b;
            }
        }),
        EnvOverride::single(
            var("QDRANT__DENSE_VECTOR_SIZE"),
            |c: &mut DaemonConfig, val| {
                if let Ok(n) = val.parse() {
                    c.qdrant.dense_vector_size = n;
                }
            },
        ),
        // ── Auto-ingestion ──────────────────────────────────────────────────
        EnvOverride::single(
            var("AUTO_INGESTION__ENABLED"),
            |c: &mut DaemonConfig, val| {
                if let Ok(b) = val.parse() {
                    c.auto_ingestion.enabled = b;
                }
            },
        ),
        EnvOverride::single(
            var("AUTO_INGESTION__AUTO_CREATE_WATCHES"),
            |c: &mut DaemonConfig, val| {
                if let Ok(b) = val.parse() {
                    c.auto_ingestion.auto_create_watches = b;
                }
            },
        ),
        EnvOverride::single(
            var("AUTO_INGESTION__TARGET_COLLECTION_SUFFIX"),
            |c: &mut DaemonConfig, val| {
                c.auto_ingestion.target_collection_suffix = val;
            },
        ),
        EnvOverride::single(
            var("AUTO_INGESTION__MAX_FILES_PER_BATCH"),
            |c: &mut DaemonConfig, val| {
                if let Ok(n) = val.parse() {
                    c.auto_ingestion.max_files_per_batch = n;
                }
            },
        ),
        // ── Daemon endpoint ─────────────────────────────────────────────────
        EnvOverride::single(var("DAEMON_HOST"), |c: &mut DaemonConfig, val| {
            c.daemon_endpoint.host = val;
        }),
        EnvOverride::single(var("DAEMON_PORT"), |c: &mut DaemonConfig, val| {
            if let Ok(n) = val.parse() {
                c.daemon_endpoint.grpc_port = n;
            }
        }),
        EnvOverride::single(var("DAEMON_AUTH_TOKEN"), |c: &mut DaemonConfig, val| {
            c.daemon_endpoint.auth_token = Some(val);
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an injected env getter from `(key, value)` pairs — avoids mutating
    /// process-global env (which would race across parallel tests).
    fn getter_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    fn apply(pairs: &[(&str, &str)]) -> DaemonConfig {
        apply_env_overrides_with(DaemonConfig::default(), &getter_from(pairs))
    }

    #[test]
    fn string_override_applies() {
        let cfg = apply(&[("WORKSPACE_QDRANT_LOG_LEVEL", "debug")]);
        assert_eq!(cfg.log_level, "debug");
    }

    #[test]
    fn numeric_override_applies() {
        let cfg = apply(&[("WORKSPACE_QDRANT_CHUNK_SIZE", "2048")]);
        assert_eq!(cfg.chunk_size, 2048);
    }

    #[test]
    fn invalid_numeric_is_ignored() {
        let default_chunk = DaemonConfig::default().chunk_size;
        let cfg = apply(&[("WORKSPACE_QDRANT_CHUNK_SIZE", "not-a-number")]);
        assert_eq!(cfg.chunk_size, default_chunk, "invalid value keeps default");
    }

    #[test]
    fn qdrant_url_and_transport_override() {
        let cfg = apply(&[
            ("WORKSPACE_QDRANT_QDRANT__URL", "http://envhost:6333"),
            ("WORKSPACE_QDRANT_QDRANT__TRANSPORT", "http"),
        ]);
        assert_eq!(cfg.qdrant.url, "http://envhost:6333");
        assert!(matches!(cfg.qdrant.transport, TransportMode::Http));
    }

    #[test]
    fn daemon_endpoint_overrides() {
        let cfg = apply(&[
            ("WORKSPACE_QDRANT_DAEMON_HOST", "10.0.0.5"),
            ("WORKSPACE_QDRANT_DAEMON_PORT", "60061"),
        ]);
        assert_eq!(cfg.daemon_endpoint.host, "10.0.0.5");
        assert_eq!(cfg.daemon_endpoint.grpc_port, 60061);
    }

    #[test]
    fn absent_env_preserves_defaults() {
        let before = DaemonConfig::default();
        let after = apply(&[]);
        assert_eq!(after.chunk_size, before.chunk_size);
        assert_eq!(after.log_level, before.log_level);
        assert_eq!(after.qdrant.url, before.qdrant.url);
    }
}
