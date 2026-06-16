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

    // Prefer the daemon-recorded URL; fall back to the configured Qdrant base
    // URL so the panel never shows a bogus "unknown" when the daemon has not
    // written the operational_state row yet.
    status.qdrant_url = conn
        .query_row(
            "SELECT value FROM operational_state WHERE key = 'qdrant_url'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .filter(|u| !u.is_empty())
        .unwrap_or_else(crate::commands::qdrant_helpers::qdrant_base_url);

    if let Ok(version) = conn.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
        row.get::<_, i64>(0)
    }) {
        status.schema_version = version;
    }

    status
}

/// The daemon-owned databases that live in the data directory, in display order.
const DB_FILE_NAMES: [&str; 4] = ["state.db", "search.db", "graph.db", "daemon_state.db"];

/// One database file's on-disk size.
#[derive(Debug, Clone)]
pub struct DbFile {
    pub name: String,
    pub size: u64,
}

/// Storage footprint of the data directory: each database file's size, the
/// total, and the free space on the volume that holds them.
#[derive(Debug, Clone, Default)]
pub struct StorageInfo {
    /// Data directory path (with `~` abbreviation), for the panel title/line.
    pub data_dir: String,
    /// Per-database sizes (only files that exist).
    pub db_files: Vec<DbFile>,
    /// Sum of the database sizes.
    pub total_db_bytes: u64,
    /// Free bytes on the volume holding the data directory (None if unknown).
    pub free_bytes: Option<u64>,
}

/// Gather database file sizes and free disk space for the data directory.
pub fn fetch_storage() -> StorageInfo {
    let mut info = StorageInfo::default();
    let Ok(dir) = wqm_common::paths::get_data_dir() else {
        return info;
    };
    info.data_dir = abbreviate_home_path(&dir.to_string_lossy());

    for name in DB_FILE_NAMES {
        let path = dir.join(name);
        if let Ok(meta) = std::fs::metadata(&path) {
            let size = meta.len();
            info.total_db_bytes += size;
            info.db_files.push(DbFile {
                name: name.to_string(),
                size,
            });
        }
    }

    info.free_bytes = free_bytes_for(&dir);
    info
}

/// Free bytes available on the filesystem that holds `path`, via statvfs.
#[cfg(unix)]
fn free_bytes_for(path: &std::path::Path) -> Option<u64> {
    let stat = nix::sys::statvfs::statvfs(path).ok()?;
    Some(stat.blocks_available() as u64 * stat.fragment_size() as u64)
}

#[cfg(not(unix))]
fn free_bytes_for(_path: &std::path::Path) -> Option<u64> {
    None
}

/// Replace the home-directory prefix with `~` for compact display.
fn abbreviate_home_path(path: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home = home.to_string_lossy();
        if let Some(rest) = path.strip_prefix(home.as_ref()) {
            return format!("~{rest}");
        }
    }
    path.to_string()
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
    /// Functional queue-health verdict, decoded from the `queue_processor`
    /// gRPC component (#133 F9). `None` until the first probe; `Unknown` is the
    /// daemon-reported cold-start ("learning baseline"), distinct from
    /// `daemon_healthy == None` (the probe itself has not returned yet — UX-7).
    pub queue_verdict: Option<crate::output::ServiceStatus>,
    /// Per-line attributed remediation for a non-green queue verdict.
    pub queue_remediation: Vec<String>,
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

    match crate::grpc::connect_default().await {
        Ok(mut client) => match client.system().health(()).await {
            Ok(resp) => {
                live.daemon_healthy = Some(true);
                // Decode the functional queue-health verdict (#133 F9) from the
                // queue_processor component — verdict-Unknown is the daemon's
                // cold-start, NOT the probe-pending state.
                let resp = resp.into_inner();
                if let Some(comp) = resp
                    .components
                    .iter()
                    .find(|c| c.component_name == "queue_processor")
                {
                    live.queue_verdict =
                        Some(crate::output::ServiceStatus::from_proto(comp.status));
                    live.queue_remediation = comp
                        .message
                        .lines()
                        .filter(|l| !l.is_empty())
                        .map(str::to_string)
                        .collect();
                }
            }
            Err(_) => live.daemon_healthy = Some(false),
        },
        Err(_) => live.daemon_healthy = Some(false),
    }

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

    #[test]
    fn abbreviates_home_prefix() {
        if let Some(home) = dirs::home_dir() {
            let p = format!("{}/.local/share/workspace-qdrant", home.to_string_lossy());
            assert_eq!(abbreviate_home_path(&p), "~/.local/share/workspace-qdrant");
        }
        assert_eq!(abbreviate_home_path("/opt/data"), "/opt/data");
    }

    #[test]
    fn fetch_storage_does_not_panic_and_sums_sizes() {
        // Point the data dir at a temp directory with two fake DB files.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("state.db"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.path().join("search.db"), vec![0u8; 250]).unwrap();
        std::env::set_var("WQM_DATA_DIR", dir.path());

        let info = fetch_storage();
        std::env::remove_var("WQM_DATA_DIR");

        assert_eq!(info.total_db_bytes, 350);
        assert_eq!(info.db_files.len(), 2);
        // statvfs should report some free space on a real temp volume.
        #[cfg(unix)]
        assert!(info.free_bytes.is_some());
    }
}
