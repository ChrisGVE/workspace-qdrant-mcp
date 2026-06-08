//! Data layer for the Service view.
//!
//! Two sources feed the view: synchronous SQLite reads (queue depth, DLQ,
//! indexed docs/chunks, watcher state) gathered on each tick, and a background
//! thread that probes live signals the render thread must never block on —
//! daemon gRPC health, Qdrant HTTP health, and the process memory footprint
//! scraped from the metrics endpoint.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::data::db::connect_readonly;

/// Metrics endpoint exposing the memexd Prometheus gauges.
const METRICS_URL: &str = "http://127.0.0.1:6337/metrics";

/// The honest memory gauge (dirty physical footprint).
const FOOTPRINT_METRIC: &str = "wqm_memexd_process_footprint_bytes";

/// Synchronous, SQLite-derived service status.
#[derive(Debug, Clone)]
pub struct ServiceStatus {
    pub db_readable: bool,
    pub qdrant_url: String,
    pub schema_version: i64,
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_failed: i64,
    pub queue_done: i64,
    pub queue_total: i64,
    pub dlq_count: i64,
    pub total_docs: i64,
    pub total_chunks: i64,
    pub watchers_active: i64,
    pub watchers_paused: i64,
}

impl Default for ServiceStatus {
    fn default() -> Self {
        Self {
            db_readable: false,
            qdrant_url: "unknown".to_string(),
            schema_version: 0,
            queue_pending: 0,
            queue_in_progress: 0,
            queue_failed: 0,
            queue_done: 0,
            queue_total: 0,
            dlq_count: 0,
            total_docs: 0,
            total_chunks: 0,
            watchers_active: 0,
            watchers_paused: 0,
        }
    }
}

/// Read all SQLite-derived service status in one pass.
pub fn fetch_service_status() -> ServiceStatus {
    let mut status = ServiceStatus::default();

    let conn = match connect_readonly() {
        Ok(c) => {
            status.db_readable = true;
            c
        }
        Err(_) => return status,
    };

    if let Ok(mut stmt) = conn.prepare("SELECT status, COUNT(*) FROM unified_queue GROUP BY status")
    {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        }) {
            for (state, count) in rows.flatten() {
                status.queue_total += count;
                match state.as_str() {
                    "pending" => status.queue_pending = count,
                    "in_progress" => status.queue_in_progress = count,
                    "failed" => status.queue_failed = count,
                    "done" => status.queue_done = count,
                    _ => {}
                }
            }
        }
    }

    if let Ok(count) = conn.query_row("SELECT COUNT(*) FROM dead_letter_queue", [], |row| {
        row.get::<_, i64>(0)
    }) {
        status.dlq_count = count;
    }

    if let Ok((docs, chunks)) = conn.query_row(
        "SELECT COUNT(file_id), COALESCE(SUM(chunk_count), 0) FROM tracked_files",
        [],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    ) {
        status.total_docs = docs;
        status.total_chunks = chunks;
    }

    if let Ok((paused, total)) = conn.query_row(
        "SELECT COALESCE(SUM(CASE WHEN is_paused = 1 THEN 1 ELSE 0 END), 0), COUNT(*) \
         FROM watch_folders WHERE parent_watch_id IS NULL",
        [],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    ) {
        status.watchers_paused = paused;
        status.watchers_active = total - paused;
    }

    if let Ok(url) = conn.query_row(
        "SELECT value FROM operational_state WHERE key = 'qdrant_url'",
        [],
        |row| row.get::<_, String>(0),
    ) {
        status.qdrant_url = url;
    }

    if let Ok(version) = conn.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
        row.get::<_, i64>(0)
    }) {
        status.schema_version = version;
    }

    status
}

/// Live signals probed off the render thread.
#[derive(Debug, Clone, Default)]
pub struct ServiceLive {
    /// Daemon gRPC health (None until the first probe completes).
    pub daemon_healthy: Option<bool>,
    /// Qdrant HTTP health (None until the first probe completes).
    pub qdrant_healthy: Option<bool>,
    /// Process dirty footprint in bytes, from the metrics endpoint.
    pub footprint_bytes: Option<u64>,
}

/// Spawn the background prober. Returns shared state the view reads each frame.
pub fn spawn_service_fetcher() -> Arc<Mutex<ServiceLive>> {
    let shared = Arc::new(Mutex::new(ServiceLive::default()));
    let shared_clone = Arc::clone(&shared);

    thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(_) => return,
        };

        rt.block_on(async move {
            loop {
                let live = probe_live().await;
                if let Ok(mut guard) = shared_clone.lock() {
                    *guard = live;
                }
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        });
    });

    shared
}

async fn probe_live() -> ServiceLive {
    let mut live = ServiceLive::default();

    live.daemon_healthy = Some(match crate::grpc::connect_default().await {
        Ok(mut client) => client.system().health(()).await.is_ok(),
        Err(_) => false,
    });

    if let Ok(client) = crate::commands::qdrant_helpers::build_qdrant_http_client() {
        let url = format!(
            "{}/collections",
            crate::commands::qdrant_helpers::qdrant_base_url()
        );
        live.qdrant_healthy = Some(
            client
                .get(&url)
                .timeout(Duration::from_secs(3))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false),
        );
    }

    live.footprint_bytes = scrape_footprint().await;

    live
}

/// Scrape the footprint gauge from the Prometheus metrics endpoint.
async fn scrape_footprint() -> Option<u64> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .ok()?;
    let body = client
        .get(METRICS_URL)
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()?;
    parse_footprint(&body)
}

/// Parse the footprint gauge value out of a Prometheus metrics body.
pub fn parse_footprint(body: &str) -> Option<u64> {
    body.lines()
        .find(|l| l.starts_with(FOOTPRINT_METRIC) && !l.starts_with('#'))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| v as u64)
}

/// Format a byte count as a human-readable size (e.g. "166.1 MB").
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_footprint_gauge() {
        let body = "# HELP wqm_memexd_process_footprint_bytes foo\n\
                    # TYPE wqm_memexd_process_footprint_bytes gauge\n\
                    wqm_memexd_process_footprint_bytes 174198784\n\
                    wqm_memexd_process_resident_memory_bytes 1049141248\n";
        assert_eq!(parse_footprint(body), Some(174198784));
    }

    #[test]
    fn footprint_absent_returns_none() {
        assert_eq!(parse_footprint("other_metric 5\n"), None);
    }

    #[test]
    fn formats_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(174_198_784), "166.1 MB");
    }

    #[test]
    fn default_status_is_unreachable() {
        let s = ServiceStatus::default();
        assert!(!s.db_readable);
        assert_eq!(s.queue_total, 0);
    }
}
