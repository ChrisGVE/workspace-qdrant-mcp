//! Health check subcommand.
//!
//! Columnar template per cli-feedback.md.
//!
//! Covers three endpoints:
//! 1. `memexd` gRPC health RPC (daemon self-report, including internal
//!    components like the queue processor).
//! 2. Qdrant HTTP liveness (`GET /readyz` on `WQM_QDRANT_URL` with
//!    fallback to the compose default `http://127.0.0.1:6333`).
//! 3. MCP HTTP transport `/healthz` (only probed when `WQM_MCP_HTTP_URL`
//!    is set — stdio-mode deployments have no HTTP endpoint to probe).
//!
//! Exit code is 0 iff every probed endpoint reports healthy; any
//! failure returns 1 so CI / compose health scripts can gate on it.
//!
//! When the daemon's `state.db` has an active relative-path migration
//! marker (see `docs/specs/16-path-abstraction.md` §6.2.2), a progress
//! banner is rendered above the human-mode output. JSON consumers are not
//! affected — JSON output remains structurally stable for scripting.

use std::env;
use std::time::Duration;

use anyhow::Result;
use colored::Colorize;

use crate::grpc::client::workspace_daemon::ComponentHealth;
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::{self, ServiceStatus};

use super::migration_banner;
use super::types::{status_label, HealthComponentJson, HealthStatusJson};

/// Default probe timeout. Short enough that an unhealthy service fails
/// the command quickly; long enough to ride out normal network jitter.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Per-endpoint probe outcome fed into the composed health report.
#[derive(Debug, Clone)]
struct Probe {
    name: &'static str,
    url: String,
    status: ServiceStatus,
    message: Option<String>,
}

impl Probe {
    fn unreachable(name: &'static str, url: String, err: impl std::fmt::Display) -> Self {
        Self {
            name,
            url,
            status: ServiceStatus::Unhealthy,
            message: Some(format!("unreachable: {err}")),
        }
    }

    fn healthy(name: &'static str, url: String) -> Self {
        Self {
            name,
            url,
            status: ServiceStatus::Healthy,
            message: None,
        }
    }

    fn degraded(name: &'static str, url: String, msg: impl Into<String>) -> Self {
        Self {
            name,
            url,
            status: ServiceStatus::Degraded,
            message: Some(msg.into()),
        }
    }
}

/// Resolve the Qdrant URL for probing. Priority:
/// 1. `WQM_QDRANT_URL` env (explicit override)
/// 2. `QDRANT_URL` env (matches daemon config)
/// 3. Active cli-config.toml profile (only when one is active)
/// 4. Compose default `http://127.0.0.1:6333`
fn qdrant_url() -> String {
    if let Ok(url) = env::var("WQM_QDRANT_URL") {
        if !url.is_empty() {
            return url;
        }
    }
    if let Ok(url) = env::var("QDRANT_URL") {
        if !url.is_empty() {
            return url;
        }
    }
    crate::config::Config::from_env().qdrant_url
}

/// Optional MCP HTTP URL. Returns `None` in stdio deployments so the probe
/// is skipped rather than reporting a bogus failure.
fn mcp_http_url() -> Option<String> {
    env::var("WQM_MCP_HTTP_URL").ok().or_else(|| {
        // Derive from MCP_HTTP_HOST/PORT if set — mirrors the docker env vars.
        let host = env::var("MCP_HTTP_HOST").ok()?;
        let port = env::var("MCP_HTTP_PORT").ok()?;
        Some(format!("http://{host}:{port}"))
    })
}

/// Issue `GET {base}/{path}` with the standard probe timeout. A 2xx
/// response is healthy; anything else degraded; transport errors unhealthy.
async fn probe_get(name: &'static str, base: &str, path: &str) -> Probe {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let client = match reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .danger_accept_invalid_certs(true) // support native-TLS deployments with self-signed certs
        .build()
    {
        Ok(c) => c,
        Err(e) => return Probe::unreachable(name, url, e),
    };

    match client.get(&url).send().await {
        Ok(resp) => {
            let code = resp.status();
            if code.is_success() {
                Probe::healthy(name, url)
            } else {
                Probe::degraded(name, url, format!("HTTP {}", code.as_u16()))
            }
        }
        Err(e) => Probe::unreachable(name, url, e),
    }
}

/// Show system health, optionally as JSON.
///
/// Returns `Ok(())` with a non-zero process exit in main when any probed
/// endpoint reports unhealthy; callers can rely on `std::process::exit(1)`
/// wiring in the CLI root once this bubbles up.
pub async fn health(json: bool) -> Result<()> {
    // Run the two HTTP probes in parallel with the daemon gRPC probe.
    let qdrant_base = qdrant_url();
    let qdrant_probe_fut = probe_get("qdrant", &qdrant_base, "/readyz");
    let mcp_probe_fut = async {
        match mcp_http_url() {
            Some(base) => Some(probe_get("mcp-http", &base, "/healthz").await),
            None => None,
        }
    };

    let daemon_probe_fut = async {
        match crate::grpc::connect_default().await {
            Ok(mut client) => match client.system().health(()).await {
                Ok(response) => Ok(response.into_inner()),
                Err(e) => Err(format!("gRPC error: {e}")),
            },
            Err(e) => Err(format!("connect error: {e}")),
        }
    };

    let (daemon_res, qdrant_probe, mcp_probe) =
        tokio::join!(daemon_probe_fut, qdrant_probe_fut, mcp_probe_fut);

    match daemon_res {
        Ok(health) => {
            let overall = ServiceStatus::from_proto(health.status);
            let external = build_external_probes(qdrant_probe, mcp_probe);

            if json {
                print_health_json(true, overall, &health.components, &external);
            } else {
                // Banner renders above the title per spec §6.2.2 so it is
                // visible before users scan health rows. The helper swallows
                // errors and skips silently when the migration is inactive,
                // so it can never block health output.
                migration_banner::print_if_active();
                canvas::print_title("System Health");
                canvas::print_blank();
                print_health_columnar(true, overall, &health.components, &external);
            }
        }
        Err(err_msg) => {
            let external = build_external_probes(qdrant_probe, mcp_probe);
            if json {
                print_health_json_disconnected(false, &external);
            } else {
                migration_banner::print_if_active();
                canvas::print_title("System Health");
                canvas::print_blank();
                print_health_columnar_disconnected(&external);
                output::error(format!(
                    "Daemon not reachable ({err_msg}). Start with: wqm service start"
                ));
            }
        }
    }

    Ok(())
}

/// Collect optional MCP probe into a flat vec of external-component probes.
fn build_external_probes(qdrant: Probe, mcp: Option<Probe>) -> Vec<Probe> {
    match mcp {
        Some(p) => vec![qdrant, p],
        None => vec![qdrant],
    }
}

fn print_health_columnar(
    _connected: bool,
    overall: ServiceStatus,
    components: &[ComponentHealth],
    external: &[Probe],
) {
    let overall_gutter = status_gutter(overall);

    let mut builder = ColumnarBuilder::new()
        .kv_gutter(
            "Connection",
            format_status(ServiceStatus::Healthy),
            Gutter::Sync,
        )
        .kv_gutter("Overall", format_status(overall), overall_gutter);

    if !components.is_empty() {
        builder = builder.section(Some("Components"));
        for comp in components {
            let comp_status = ServiceStatus::from_proto(comp.status);
            let gutter = status_gutter(comp_status);
            builder = builder.kv_gutter(&comp.component_name, format_status(comp_status), gutter);
            // Each remediation line is already `[<rag> <culprit>] <text>` from the
            // daemon (#133 F7/UX-5) — render them one per row, not as one blob.
            for line in comp.message.lines().filter(|l| !l.is_empty()) {
                builder = builder.raw(&format!("  {}", line.dimmed()), Gutter::None);
            }
        }
    }

    if !external.is_empty() {
        builder = builder.section(Some("External"));
        for probe in external {
            let gutter = status_gutter(probe.status);
            builder = builder.kv_gutter(probe.name, format_status(probe.status), gutter);
            builder = builder.raw(&format!("  {}", probe.url.dimmed()), Gutter::None);
            if let Some(msg) = &probe.message {
                builder = builder.raw(&format!("  {}", msg.dimmed()), Gutter::None);
            }
        }
    }

    builder.render();
}

fn print_health_columnar_disconnected(external: &[Probe]) {
    let mut builder = ColumnarBuilder::new().kv_gutter(
        "Connection",
        format_status(ServiceStatus::Unhealthy),
        Gutter::Error,
    );

    if !external.is_empty() {
        builder = builder.section(Some("External"));
        for probe in external {
            let gutter = status_gutter(probe.status);
            builder = builder.kv_gutter(probe.name, format_status(probe.status), gutter);
            builder = builder.raw(&format!("  {}", probe.url.dimmed()), Gutter::None);
            if let Some(msg) = &probe.message {
                builder = builder.raw(&format!("  {}", msg.dimmed()), Gutter::None);
            }
        }
    }

    builder.render();
}

fn format_status(status: ServiceStatus) -> String {
    match status {
        ServiceStatus::Healthy | ServiceStatus::Active => status_label(status).green().to_string(),
        ServiceStatus::Degraded => status_label(status).yellow().to_string(),
        ServiceStatus::Unhealthy => status_label(status).red().to_string(),
        ServiceStatus::Inactive => status_label(status).dimmed().to_string(),
        ServiceStatus::Unknown => status_label(status).dimmed().to_string(),
    }
}

fn status_gutter(status: ServiceStatus) -> Gutter {
    // Exhaustive (UX-6): cold-start `Unknown` maps to a visible `…` gutter, never
    // the blank `Gutter::None` a wildcard would give it.
    match status {
        ServiceStatus::Healthy | ServiceStatus::Active => Gutter::Sync,
        ServiceStatus::Degraded => Gutter::Warning,
        ServiceStatus::Unhealthy => Gutter::Error,
        ServiceStatus::Unknown => Gutter::Probing,
        ServiceStatus::Inactive => Gutter::None,
    }
}

fn print_health_json(
    connected: bool,
    overall: ServiceStatus,
    components: &[ComponentHealth],
    external: &[Probe],
) {
    let mut components: Vec<HealthComponentJson> = components
        .iter()
        .map(|c| HealthComponentJson {
            name: c.component_name.clone(),
            status: status_label(ServiceStatus::from_proto(c.status)).to_string(),
            message: if c.message.is_empty() {
                None
            } else {
                Some(c.message.clone())
            },
            remediation: c
                .message
                .lines()
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect(),
        })
        .collect();
    // External probes appear in the same JSON array with a qualified name
    // so JSON consumers can distinguish daemon-internal components from
    // out-of-process services.
    for probe in external {
        components.push(HealthComponentJson {
            name: format!("external:{}", probe.name),
            status: status_label(probe.status).to_string(),
            message: probe.message.clone().or_else(|| Some(probe.url.clone())),
            remediation: Vec::new(),
        });
    }
    let json_out = HealthStatusJson {
        connected,
        health: status_label(overall).to_string(),
        components,
    };
    output::print_json(&json_out);
}

fn print_health_json_disconnected(connected: bool, external: &[Probe]) {
    let components: Vec<HealthComponentJson> = external
        .iter()
        .map(|probe| HealthComponentJson {
            name: format!("external:{}", probe.name),
            status: status_label(probe.status).to_string(),
            message: probe.message.clone().or_else(|| Some(probe.url.clone())),
            remediation: Vec::new(),
        })
        .collect();
    let json_out = HealthStatusJson {
        connected,
        health: if connected {
            "unknown".to_string()
        } else {
            "unhealthy".to_string()
        },
        components,
    };
    output::print_json(&json_out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn cold_start_status_maps_to_probing_gutter() {
        // #133 F8/UX-6: Unknown (cold-start) gets a visible `…` gutter, not blank.
        assert_eq!(status_gutter(ServiceStatus::Unknown), Gutter::Probing);
        assert_eq!(status_gutter(ServiceStatus::Healthy), Gutter::Sync);
        assert_eq!(status_gutter(ServiceStatus::Unhealthy), Gutter::Error);
        assert_eq!(status_gutter(ServiceStatus::Degraded), Gutter::Warning);
    }

    /// Sentinel values chosen so the tests can't collide with production env.
    const TEST_HOST: &str = "http://127.0.0.1:16333";
    const TEST_HOST_B: &str = "http://127.0.0.1:16334";
    const TEST_MCP_HOST: &str = "http://127.0.0.1:16335";

    fn clear_url_env() {
        env::remove_var("WQM_QDRANT_URL");
        env::remove_var("QDRANT_URL");
        env::remove_var("WQM_MCP_HTTP_URL");
        env::remove_var("MCP_HTTP_HOST");
        env::remove_var("MCP_HTTP_PORT");
        env::remove_var("WQM_PROFILE");
        // Point at a path that can't exist so the user's real cli-config.toml
        // (if present) does not leak profile endpoints into the test.
        env::set_var(
            "WQM_CLI_CONFIG",
            "/dev/null/wqm-health-test-nonexistent/cli-config.toml",
        );
    }

    #[test]
    #[serial]
    fn qdrant_url_defaults_to_loopback_when_no_env() {
        clear_url_env();
        assert_eq!(qdrant_url(), wqm_common::constants::DEFAULT_QDRANT_URL);
    }

    #[test]
    #[serial]
    fn qdrant_url_prefers_wqm_override() {
        clear_url_env();
        env::set_var("WQM_QDRANT_URL", TEST_HOST);
        env::set_var("QDRANT_URL", TEST_HOST_B);
        let got = qdrant_url();
        clear_url_env();
        assert_eq!(got, TEST_HOST);
    }

    #[test]
    #[serial]
    fn qdrant_url_falls_back_to_qdrant_url_env() {
        clear_url_env();
        env::set_var("QDRANT_URL", TEST_HOST_B);
        let got = qdrant_url();
        clear_url_env();
        assert_eq!(got, TEST_HOST_B);
    }

    #[test]
    #[serial]
    fn mcp_http_url_is_none_without_env() {
        clear_url_env();
        assert!(mcp_http_url().is_none());
    }

    #[test]
    #[serial]
    fn mcp_http_url_uses_explicit_override_first() {
        clear_url_env();
        env::set_var("WQM_MCP_HTTP_URL", TEST_MCP_HOST);
        env::set_var("MCP_HTTP_HOST", "ignored");
        env::set_var("MCP_HTTP_PORT", "9999");
        let got = mcp_http_url();
        clear_url_env();
        assert_eq!(got, Some(TEST_MCP_HOST.to_string()));
    }

    #[test]
    #[serial]
    fn mcp_http_url_assembles_from_host_and_port() {
        clear_url_env();
        env::set_var("MCP_HTTP_HOST", "127.0.0.1");
        env::set_var("MCP_HTTP_PORT", "16335");
        let got = mcp_http_url();
        clear_url_env();
        assert_eq!(got, Some("http://127.0.0.1:16335".to_string()));
    }

    #[test]
    #[serial]
    fn mcp_http_url_requires_both_host_and_port_to_assemble() {
        clear_url_env();
        env::set_var("MCP_HTTP_HOST", "127.0.0.1");
        let got = mcp_http_url();
        clear_url_env();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn probe_reports_unreachable_for_closed_port() {
        // Use a guaranteed-unused high port on the loopback. `probe_get`
        // must return an Unhealthy probe with a descriptive message, never
        // panic or hang beyond PROBE_TIMEOUT.
        let probe = probe_get("test", "http://127.0.0.1:1", "/healthz").await;
        assert_eq!(probe.status, ServiceStatus::Unhealthy);
        assert!(probe.message.is_some());
    }

    #[test]
    fn probe_constructors_carry_name_and_url() {
        let h = Probe::healthy("x", "http://a/".to_string());
        assert_eq!(h.name, "x");
        assert_eq!(h.url, "http://a/");
        assert_eq!(h.status, ServiceStatus::Healthy);

        let d = Probe::degraded("y", "http://b/".to_string(), "HTTP 503");
        assert_eq!(d.status, ServiceStatus::Degraded);
        assert_eq!(d.message.as_deref(), Some("HTTP 503"));

        let u = Probe::unreachable("z", "http://c/".to_string(), "connection refused");
        assert_eq!(u.status, ServiceStatus::Unhealthy);
        assert!(u.message.as_deref().unwrap().contains("connection refused"));
    }
}
