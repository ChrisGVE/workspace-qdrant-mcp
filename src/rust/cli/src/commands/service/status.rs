//! Service status subcommand
//!
//! Reports the daemon [`DaemonSource`](super::detect::DaemonSource) and any
//! source-specific diagnostics:
//!
//! | Source                  | Extra diagnostics                                |
//! |-------------------------|--------------------------------------------------|
//! | `LocalOnly { pid }`     | PID, uptime, RSS (best-effort via `ps`)          |
//! | `DockerOnly`            | `docker stats --no-stream memexd` (best-effort)  |
//! | `Both { pid }`          | Both local and docker sections                   |
//! | `RemoteOnly { addr }`   | gRPC reachability + address                      |
//! | `None`                  | "memexd not running"                             |
//!
//! On any source where a gRPC endpoint is expected to be reachable, an
//! optional scrape of `http://127.0.0.1:6337/metrics` is attempted with a
//! 500 ms timeout. Failure of the scrape never fails the command.

use anyhow::Result;
use colored::Colorize;
use serde::Serialize;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::grpc::client::DaemonClient;
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::{self, ServiceStatus};

use super::detect::{detect_daemon_source, DaemonSource};

/// URL used for the best-effort Prometheus-style metrics scrape.
const METRICS_URL: &str = "http://127.0.0.1:6337/metrics";

/// Upper bound for the metrics scrape. Kept tight so the command stays
/// responsive even when the daemon is down.
const METRICS_TIMEOUT: Duration = Duration::from_millis(500);

/// JSON-serializable service status
#[derive(Serialize, Clone, Debug)]
struct ServiceStatusJson {
    source: String,
    connected: bool,
    health: String,
    components: Vec<ComponentStatusJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local: Option<LocalInfoJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    docker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics_summary: Option<String>,
}

/// JSON-serializable component health
#[derive(Serialize, Clone, Debug)]
struct ComponentStatusJson {
    name: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// JSON-serializable local process info
#[derive(Serialize, Clone, Debug)]
struct LocalInfoJson {
    pid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    uptime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rss_kb: Option<u64>,
}

fn status_name(s: ServiceStatus) -> &'static str {
    match s {
        ServiceStatus::Healthy => "healthy",
        ServiceStatus::Degraded => "degraded",
        ServiceStatus::Unhealthy => "unhealthy",
        ServiceStatus::Active => "active",
        ServiceStatus::Inactive => "inactive",
        ServiceStatus::Unknown => "unknown",
    }
}

fn format_status(status: ServiceStatus) -> String {
    match status {
        ServiceStatus::Healthy | ServiceStatus::Active => status_name(status).green().to_string(),
        ServiceStatus::Degraded => status_name(status).yellow().to_string(),
        ServiceStatus::Unhealthy => status_name(status).red().to_string(),
        ServiceStatus::Inactive | ServiceStatus::Unknown => {
            status_name(status).dimmed().to_string()
        }
    }
}

fn status_gutter(status: ServiceStatus) -> Gutter {
    match status {
        ServiceStatus::Healthy | ServiceStatus::Active => Gutter::Sync,
        ServiceStatus::Degraded => Gutter::Warning,
        ServiceStatus::Unhealthy => Gutter::Error,
        _ => Gutter::None,
    }
}

/// Local process diagnostics gathered best-effort via `ps`.
///
/// Returns `None` fields whenever the probe fails; callers must tolerate
/// missing data rather than failing the command.
#[derive(Debug, Default, Clone)]
struct LocalInfo {
    uptime: Option<String>,
    rss_kb: Option<u64>,
}

/// Collect platform-best-effort local process info.
///
/// On Unix we shell out to `ps -o etime=,rss= -p <pid>` because it is
/// available everywhere we support without pulling a new dependency. On
/// non-Unix targets we simply return empty info.
fn local_info(pid: u32) -> LocalInfo {
    #[cfg(unix)]
    {
        let out = Command::new("ps")
            .args(["-o", "etime=,rss=", "-p", &pid.to_string()])
            .stderr(Stdio::null())
            .output();
        let Ok(out) = out else {
            return LocalInfo::default();
        };
        if !out.status.success() {
            return LocalInfo::default();
        }
        let line = String::from_utf8_lossy(&out.stdout);
        let trimmed = line.trim();
        // `ps` emits: "<etime> <rss>". etime format varies (DD-HH:MM:SS,
        // HH:MM:SS, or MM:SS). Split on whitespace from the right so the
        // uptime stays atomic.
        let mut parts = trimmed.rsplitn(2, char::is_whitespace);
        let rss = parts.next().and_then(|s| s.trim().parse::<u64>().ok());
        let etime = parts.next().map(|s| s.trim().to_string());
        LocalInfo {
            uptime: etime.filter(|s| !s.is_empty()),
            rss_kb: rss,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        LocalInfo::default()
    }
}

/// Best-effort Prometheus scrape summary.
///
/// Returns a short compact string ("N metrics reachable") when the endpoint
/// responds within the timeout; returns `None` on any failure so callers
/// can simply skip the line.
async fn scrape_metrics_summary() -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(METRICS_TIMEOUT)
        .build()
        .ok()?;
    let resp = client.get(METRICS_URL).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.text().await.ok()?;
    // Count non-comment, non-empty lines as a compact summary signal.
    let count = body
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .count();
    Some(format!("{count} metric samples reachable at {METRICS_URL}"))
}

/// Best-effort docker-stats snapshot.
///
/// Returns a trimmed single-line summary from `docker stats --no-stream
/// memexd` when docker is available. `None` on any failure.
fn docker_stats_summary() -> Option<String> {
    let out = Command::new("docker")
        .args([
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}",
            "memexd",
        ])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// Render the health response when a daemon is reachable via gRPC.
async fn handle_connected(mut client: DaemonClient, json: bool) -> Result<Health> {
    match client.system().health(()).await {
        Ok(response) => {
            let health = response.into_inner();
            let overall = ServiceStatus::from_proto(health.status);
            let components: Vec<ComponentStatusJson> = health
                .components
                .iter()
                .map(|c| ComponentStatusJson {
                    name: c.component_name.clone(),
                    status: status_name(ServiceStatus::from_proto(c.status)).to_string(),
                    message: if c.message.is_empty() {
                        None
                    } else {
                        Some(c.message.clone())
                    },
                })
                .collect();
            if !json {
                render_health_columnar(&health, overall);
            }
            Ok(Health {
                overall,
                components,
            })
        }
        Err(e) => {
            if !json {
                output::warning(format!("Could not get health: {}", e));
            }
            Ok(Health {
                overall: ServiceStatus::Unknown,
                components: Vec::new(),
            })
        }
    }
}

/// Columnar render of the gRPC health response.
fn render_health_columnar(
    health: &crate::grpc::client::workspace_daemon::HealthResponse,
    overall: ServiceStatus,
) {
    let overall_gutter = status_gutter(overall);
    let mut builder = ColumnarBuilder::new()
        .kv_gutter(
            "Connection",
            format_status(ServiceStatus::Healthy),
            Gutter::Sync,
        )
        .kv_gutter("Health", format_status(overall), overall_gutter);

    if !health.components.is_empty() {
        builder = builder.section(Some("Components"));
        for comp in &health.components {
            let comp_status = ServiceStatus::from_proto(comp.status);
            let gutter = status_gutter(comp_status);
            builder = builder.kv_gutter(&comp.component_name, format_status(comp_status), gutter);
            if !comp.message.is_empty() {
                builder = builder.raw(&format!("  {}", comp.message.dimmed()), Gutter::None);
            }
        }
    }

    builder.render();
}

#[derive(Debug, Clone)]
struct Health {
    overall: ServiceStatus,
    components: Vec<ComponentStatusJson>,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            overall: ServiceStatus::Unknown,
            components: Vec::new(),
        }
    }
}

/// Returns true when we expect something to answer on the daemon gRPC port.
fn expect_grpc(source: &DaemonSource) -> bool {
    !matches!(source, DaemonSource::None)
}

/// Probe gRPC health; returns (connected, health).
async fn probe_health(source: &DaemonSource, json: bool) -> Result<(bool, Health)> {
    if !expect_grpc(source) {
        return Ok((false, Health::default()));
    }
    match DaemonClient::connect_default().await {
        Ok(client) => {
            let health = handle_connected(client, json).await?;
            Ok((true, health))
        }
        Err(_) => {
            if !json && matches!(source, DaemonSource::RemoteOnly { .. }) {
                output::warning("Remote gRPC endpoint not reachable");
            }
            Ok((false, Health::default()))
        }
    }
}

/// Extract and render local-process info for `LocalOnly`/`Both` sources.
fn collect_local(source: &DaemonSource, json: bool) -> Option<LocalInfoJson> {
    let pid = match source {
        DaemonSource::LocalOnly { pid } | DaemonSource::Both { pid } => *pid,
        _ => return None,
    };
    let info = local_info(pid);
    if !json {
        output::kv("PID", pid.to_string());
        if let Some(uptime) = &info.uptime {
            output::kv("Uptime", uptime);
        }
        if let Some(rss) = info.rss_kb {
            output::kv("RSS", format!("{} KB", rss));
        }
    }
    Some(LocalInfoJson {
        pid,
        uptime: info.uptime,
        rss_kb: info.rss_kb,
    })
}

/// Extract and render docker-stats info for `DockerOnly`/`Both` sources.
fn collect_docker(source: &DaemonSource, json: bool) -> Option<String> {
    if !matches!(source, DaemonSource::DockerOnly | DaemonSource::Both { .. }) {
        return None;
    }
    let summary = docker_stats_summary()?;
    if !json {
        output::kv("Docker", &summary);
    }
    Some(summary)
}

/// Show daemon status, optionally as JSON
pub async fn execute(json: bool) -> Result<()> {
    let source = detect_daemon_source().await;

    if !json {
        canvas::print_title("Daemon Status");
        canvas::print_blank();
        output::kv("Source", source.to_string());
    }

    let (connected, health) = probe_health(&source, json).await?;
    let local_json = collect_local(&source, json);
    let docker_json = collect_docker(&source, json);

    if matches!(source, DaemonSource::None) && !json {
        output::info("memexd not running");
    }

    let metrics_summary = if expect_grpc(&source) {
        scrape_metrics_summary().await
    } else {
        None
    };
    if let (Some(summary), false) = (&metrics_summary, json) {
        output::kv("Metrics", summary);
    }

    if json {
        output::print_json(&ServiceStatusJson {
            source: source.to_string(),
            connected,
            health: status_name(health.overall).to_string(),
            components: health.components,
            local: local_json,
            docker: docker_json,
            metrics_summary,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_info_returns_default_for_absurd_pid() {
        // u32::MAX is virtually guaranteed to not map to a real process;
        // `ps` will exit non-zero and we must return the default.
        let info = local_info(u32::MAX);
        assert!(info.uptime.is_none());
        assert!(info.rss_kb.is_none());
    }

    #[test]
    #[cfg(unix)]
    fn local_info_populates_for_self() {
        let info = local_info(std::process::id());
        // `ps` should at minimum return an RSS value for our own process.
        assert!(info.rss_kb.is_some(), "rss should be reported for self");
        assert!(info.uptime.is_some(), "uptime should be reported for self");
    }

    #[tokio::test]
    async fn metrics_scrape_tolerates_closed_port() {
        // The metrics endpoint is not expected to respond in the test
        // environment; the helper must return `None` rather than panic.
        let result = scrape_metrics_summary().await;
        // Allow either: some local stack may actually have 6337 open.
        // What matters is no panic and no hang.
        let _ = result;
    }

    #[test]
    fn status_name_covers_all_variants() {
        assert_eq!(status_name(ServiceStatus::Healthy), "healthy");
        assert_eq!(status_name(ServiceStatus::Degraded), "degraded");
        assert_eq!(status_name(ServiceStatus::Unhealthy), "unhealthy");
        assert_eq!(status_name(ServiceStatus::Active), "active");
        assert_eq!(status_name(ServiceStatus::Inactive), "inactive");
        assert_eq!(status_name(ServiceStatus::Unknown), "unknown");
    }
}
