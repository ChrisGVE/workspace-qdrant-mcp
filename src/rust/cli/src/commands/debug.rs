//! Debug command - consolidated diagnostics
//!
//! Provides diagnostic tools for troubleshooting the daemon and services.
//! Subcommands: logs, errors, queue-errors, language

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use wqm_common::timestamps;
use clap::{Args, Subcommand, ValueEnum};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Debug command arguments
#[derive(Args)]
pub struct DebugArgs {
    #[command(subcommand)]
    command: DebugCommand,
}

/// Log component filter
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum LogComponent {
    /// Show logs from all components (merged and sorted by timestamp)
    #[default]
    All,
    /// Show daemon logs only
    Daemon,
    /// Show MCP server logs only
    McpServer,
}

/// Debug subcommands
#[derive(Subcommand)]
enum DebugCommand {
    /// View daemon and MCP server logs (merged from canonical paths)
    Logs {
        /// Number of lines to show (default: 50)
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,

        /// Follow log output (like tail -f)
        #[arg(short, long)]
        follow: bool,

        /// Filter by component
        #[arg(short, long, value_enum, default_value = "all")]
        component: LogComponent,

        /// Filter by MCP session ID
        #[arg(short, long)]
        session: Option<String>,

        /// Output in JSON format (raw log entries)
        #[arg(long)]
        json: bool,

        /// Show only ERROR and WARN level entries
        #[arg(short, long)]
        errors_only: bool,

        /// Show entries since time (e.g. '1h', '30m', '2d', or ISO 8601 timestamp)
        #[arg(long)]
        since: Option<String>,
    },

    /// Show recent errors from all sources
    Errors {
        /// Number of errors to show (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        count: usize,

        /// Filter by component (daemon, queue, lsp, grammar)
        #[arg(short, long)]
        component: Option<String>,
    },

    /// Show queue processing errors
    QueueErrors {
        /// Number of errors to show (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        count: usize,

        /// Filter by operation type (ingest, update, delete)
        #[arg(short, long)]
        operation: Option<String>,
    },

    /// Diagnose language support issues
    Language {
        /// Language to diagnose
        language: String,

        /// Run verbose diagnostics
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Execute debug command
pub async fn execute(args: DebugArgs) -> Result<()> {
    match args.command {
        DebugCommand::Logs {
            lines,
            follow,
            component,
            session,
            json,
            errors_only,
            since,
        } => logs(lines, follow, component, session, json, errors_only, since).await,
        DebugCommand::Errors { count, component } => errors(count, component).await,
        DebugCommand::QueueErrors { count, operation } => queue_errors(count, operation).await,
        DebugCommand::Language { language, verbose } => diagnose_language(&language, verbose).await,
    }
}

/// Returns the canonical OS-specific log directory for workspace-qdrant logs.
///
/// Delegates to `wqm_common::paths::get_canonical_log_dir()`.
fn get_canonical_log_dir() -> PathBuf {
    wqm_common::paths::get_canonical_log_dir()
}

/// Parse a relative time string (e.g. "1h", "30m", "2d", "10s") into a chrono Duration.
fn parse_relative_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("Empty time string");
    }

    // Try to split into numeric part and unit
    let (num_str, unit) = if s.ends_with("ms") {
        (&s[..s.len() - 2], "ms")
    } else {
        let split_pos = s
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(s.len());
        (&s[..split_pos], &s[split_pos..])
    };

    let num: f64 = num_str
        .parse()
        .with_context(|| format!("Invalid number in time string: '{}'", s))?;

    if num < 0.0 {
        anyhow::bail!("Negative time value: '{}'", s);
    }

    match unit {
        "s" | "sec" | "secs" => Ok(Duration::seconds(num as i64)),
        "m" | "min" | "mins" => Ok(Duration::minutes(num as i64)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Ok(Duration::hours(num as i64)),
        "d" | "day" | "days" => Ok(Duration::days(num as i64)),
        "w" | "week" | "weeks" => Ok(Duration::weeks(num as i64)),
        "ms" => Ok(Duration::milliseconds(num as i64)),
        "" => {
            // Default to seconds if no unit
            Ok(Duration::seconds(num as i64))
        }
        _ => anyhow::bail!("Unknown time unit '{}' in '{}'", unit, s),
    }
}

/// Parse a --since argument into a UTC cutoff timestamp.
///
/// Accepts:
/// - Relative: "1h", "30m", "2d", "10s", "1w"
/// - Absolute: ISO 8601 / RFC 3339 timestamps
fn parse_since(value: &str) -> Result<DateTime<Utc>> {
    // Try relative first (starts with a digit)
    if value
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        if let Ok(dur) = parse_relative_duration(value) {
            return Ok(Utc::now() - dur);
        }
    }

    // Try ISO 8601 / RFC 3339
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Ok(dt.with_timezone(&Utc));
    }
    // Try without timezone (assume UTC)
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return Ok(dt.and_utc());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc());
    }

    anyhow::bail!(
        "Cannot parse '{}'. Use relative (e.g. '1h', '30m', '2d') or ISO 8601 timestamp.",
        value
    )
}

/// Log levels ordered by severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    /// Parse a log level from a JSON value (string or numeric).
    fn from_json(val: &serde_json::Value) -> Option<Self> {
        match val {
            serde_json::Value::String(s) => Self::from_str(s),
            serde_json::Value::Number(n) => {
                // pino numeric levels: 10=trace, 20=debug, 30=info, 40=warn, 50=error
                let num = n.as_u64()?;
                match num {
                    0..=10 => Some(LogLevel::Trace),
                    11..=20 => Some(LogLevel::Debug),
                    21..=30 => Some(LogLevel::Info),
                    31..=40 => Some(LogLevel::Warn),
                    _ => Some(LogLevel::Error),
                }
            }
            _ => None,
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARN" | "WARNING" => Some(LogLevel::Warn),
            "ERROR" | "ERR" | "FATAL" => Some(LogLevel::Error),
            _ => None,
        }
    }

    fn is_error_or_warn(self) -> bool {
        matches!(self, LogLevel::Error | LogLevel::Warn)
    }
}

/// Represents a parsed log entry with timestamp and component info
#[derive(Debug)]
struct LogEntry {
    timestamp: String,
    parsed_time: Option<DateTime<Utc>>,
    level: Option<LogLevel>,
    component: String,
    session_id: Option<String>,
    raw_line: String,
}

impl LogEntry {
    /// Try to parse a JSON log line, extracting timestamp, level, and session_id.
    fn from_json_line(line: &str, component: &str) -> Option<Self> {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            let timestamp = json
                .get("timestamp")
                .or_else(|| json.get("time"))
                .or_else(|| json.get("ts"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let parsed_time = if !timestamp.is_empty() {
                DateTime::parse_from_rfc3339(&timestamp)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            } else {
                None
            };

            let level = json
                .get("level")
                .or_else(|| json.get("severity"))
                .and_then(LogLevel::from_json);

            let session_id = json
                .get("session_id")
                .and_then(|v| v.as_str())
                .map(String::from);

            Some(LogEntry {
                timestamp,
                parsed_time,
                level,
                component: component.to_string(),
                session_id,
                raw_line: line.to_string(),
            })
        } else {
            // Non-JSON line
            Some(LogEntry {
                timestamp: String::new(),
                parsed_time: None,
                level: None,
                component: component.to_string(),
                session_id: None,
                raw_line: line.to_string(),
            })
        }
    }
}

/// Log entry filter criteria.
struct LogFilter {
    errors_only: bool,
    since: Option<DateTime<Utc>>,
    session: Option<String>,
}

impl LogFilter {
    fn matches(&self, entry: &LogEntry) -> bool {
        if self.errors_only {
            if !entry.level.map(|l| l.is_error_or_warn()).unwrap_or(false) {
                return false;
            }
        }
        if let Some(cutoff) = &self.since {
            if let Some(ts) = &entry.parsed_time {
                if ts < cutoff {
                    return false;
                }
            }
            // Entries without a parseable timestamp are included (conservative)
        }
        if let Some(ref sid) = self.session {
            if !entry
                .session_id
                .as_ref()
                .map(|s| s.contains(sid))
                .unwrap_or(false)
            {
                return false;
            }
        }
        true
    }
}

/// Discover log files for a component, including rotated files.
///
/// Returns files sorted newest-first. Matches patterns:
/// - `<base>.jsonl` (current)
/// - `<base>.jsonl.1`, `<base>.jsonl.2`, ... (rotated by logroller)
fn discover_log_files(log_dir: &Path, base_name: &str) -> Vec<PathBuf> {
    let current = log_dir.join(format!("{}.jsonl", base_name));
    let mut files: Vec<PathBuf> = Vec::new();

    if current.exists() {
        files.push(current.clone());
    }

    // Look for rotated files: <base>.jsonl.1, <base>.jsonl.2, ...
    if let Ok(entries) = fs::read_dir(log_dir) {
        let prefix = format!("{}.jsonl.", base_name);
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&prefix) && entry.path() != current {
                // Check that suffix is numeric (not .gz etc. which we can't read yet)
                let suffix = &name_str[prefix.len()..];
                if suffix.chars().all(|c| c.is_ascii_digit()) {
                    files.push(entry.path());
                }
            }
        }
    }

    // Sort: current file first, then rotated by number (ascending = oldest first)
    // We want newest-first for efficient tail-N reading
    files.sort_by(|a, b| {
        let num_a = rotated_index(a);
        let num_b = rotated_index(b);
        num_a.cmp(&num_b)
    });

    files
}

/// Extract rotated file index (0 for current, N for .N suffix).
fn rotated_index(path: &Path) -> u32 {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

/// Read log entries from a file with streaming filter, returning last N matching entries.
fn read_log_file_filtered(
    path: &Path,
    component: &str,
    max_lines: usize,
    filter: &LogFilter,
) -> Vec<LogEntry> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let reader = BufReader::new(file);
    // Use a ring buffer to keep only the last max_lines matching entries
    let mut ring: Vec<LogEntry> = Vec::with_capacity(max_lines + 1);

    for line in reader.lines().filter_map(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(entry) = LogEntry::from_json_line(&line, component) {
            if filter.matches(&entry) {
                ring.push(entry);
                if ring.len() > max_lines {
                    ring.remove(0);
                }
            }
        }
    }

    ring
}

/// Read from multiple log files (current + rotated) respecting filters.
fn read_log_files_filtered(
    log_dir: &Path,
    base_name: &str,
    component: &str,
    max_lines: usize,
    filter: &LogFilter,
) -> Vec<LogEntry> {
    let files = discover_log_files(log_dir, base_name);

    if files.is_empty() {
        return Vec::new();
    }

    // If no --since filter, only read current file for performance
    if filter.since.is_none() {
        if let Some(current) = files.first() {
            return read_log_file_filtered(current, component, max_lines, filter);
        }
        return Vec::new();
    }

    // With --since, we may need to read rotated files too (oldest first)
    let mut all: Vec<LogEntry> = Vec::new();
    // Read files in reverse order (oldest first) so we can accumulate chronologically
    for file in files.iter().rev() {
        let entries = read_log_file_filtered(file, component, max_lines, filter);
        all.extend(entries);
    }

    // Take last N
    if all.len() > max_lines {
        all.drain(..all.len() - max_lines);
    }

    all
}

async fn logs(
    lines: usize,
    follow: bool,
    component: LogComponent,
    session: Option<String>,
    json_output: bool,
    errors_only: bool,
    since: Option<String>,
) -> Result<()> {
    let log_dir = get_canonical_log_dir();
    let daemon_log = log_dir.join("daemon.jsonl");
    let mcp_log = log_dir.join("mcp-server.jsonl");

    // Parse --since into a cutoff timestamp
    let since_cutoff = match &since {
        Some(s) => Some(parse_since(s)?),
        None => None,
    };

    if !json_output {
        output::section("Logs");
        output::kv("Log directory", &log_dir.display().to_string());
        output::kv(
            "Component",
            match component {
                LogComponent::All => "all (merged)",
                LogComponent::Daemon => "daemon",
                LogComponent::McpServer => "mcp-server",
            },
        );
        if let Some(ref sid) = session {
            output::kv("Session filter", sid);
        }
        if errors_only {
            output::kv("Level filter", "ERROR/WARN only");
        }
        if let Some(ref cutoff) = since_cutoff {
            output::kv("Since", &timestamps::format_utc(cutoff));
        }
        output::separator();
    }

    // Handle follow mode
    if follow {
        return follow_logs(component, &daemon_log, &mcp_log, session, errors_only).await;
    }

    let filter = LogFilter {
        errors_only,
        since: since_cutoff,
        session,
    };

    // Collect log entries based on component filter
    let mut all_entries: Vec<LogEntry> = Vec::new();

    match component {
        LogComponent::All => {
            all_entries.extend(read_log_files_filtered(
                &log_dir, "daemon", "daemon", lines, &filter,
            ));
            all_entries.extend(read_log_files_filtered(
                &log_dir,
                "mcp-server",
                "mcp-server",
                lines,
                &filter,
            ));
        }
        LogComponent::Daemon => {
            all_entries.extend(read_log_files_filtered(
                &log_dir, "daemon", "daemon", lines, &filter,
            ));
        }
        LogComponent::McpServer => {
            all_entries.extend(read_log_files_filtered(
                &log_dir,
                "mcp-server",
                "mcp-server",
                lines,
                &filter,
            ));
        }
    }

    // Sort by timestamp if merging
    if matches!(component, LogComponent::All) {
        all_entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    // Take last N entries
    let start = all_entries.len().saturating_sub(lines);
    let display_entries = &all_entries[start..];

    if display_entries.is_empty() {
        if !json_output {
            output::warning("No log entries found");
            show_log_locations(&daemon_log, &mcp_log);
        }
        return Ok(());
    }

    // Output entries
    for entry in display_entries {
        if json_output {
            println!("{}", entry.raw_line);
        } else {
            // Format nicely for terminal: [component] line
            if matches!(component, LogComponent::All) {
                println!("[{}] {}", entry.component, entry.raw_line);
            } else {
                println!("{}", entry.raw_line);
            }
        }
    }

    if !json_output {
        output::separator();
        output::info(format!("Showing {} entries", display_entries.len()));
    }

    Ok(())
}

/// Follow logs in real-time
async fn follow_logs(
    component: LogComponent,
    daemon_log: &PathBuf,
    mcp_log: &PathBuf,
    session: Option<String>,
    errors_only: bool,
) -> Result<()> {
    output::info("Following logs... Press Ctrl+C to stop");
    output::separator();

    // Build tail command based on component
    let mut files_to_follow: Vec<&PathBuf> = Vec::new();

    match component {
        LogComponent::All => {
            if daemon_log.exists() {
                files_to_follow.push(daemon_log);
            }
            if mcp_log.exists() {
                files_to_follow.push(mcp_log);
            }
        }
        LogComponent::Daemon => {
            if daemon_log.exists() {
                files_to_follow.push(daemon_log);
            }
        }
        LogComponent::McpServer => {
            if mcp_log.exists() {
                files_to_follow.push(mcp_log);
            }
        }
    }

    if files_to_follow.is_empty() {
        output::warning("No log files found to follow");
        show_log_locations(daemon_log, mcp_log);
        return Ok(());
    }

    let file_args: Vec<String> = files_to_follow
        .iter()
        .map(|p| p.display().to_string())
        .collect();

    // Build grep filter patterns
    let mut grep_patterns: Vec<String> = Vec::new();
    if let Some(sid) = session {
        grep_patterns.push(sid);
    }
    if errors_only {
        // Match JSON level fields: "level":"ERROR", "level":"WARN", "level":50, "level":40
        grep_patterns
            .push(r#""level"\s*:\s*\("ERROR"\|"WARN"\|"error"\|"warn"\|50\|40\)"#.to_string());
    }

    if grep_patterns.is_empty() {
        // No filtering - direct tail
        let mut cmd = Command::new("tail");
        cmd.arg("-f").arg("-n").arg("10");
        for path in &files_to_follow {
            cmd.arg(path);
        }
        let _ = cmd.status();
    } else {
        // Pipe through grep for filtering
        let grep_expr = grep_patterns.join(r"\|");
        let tail_cmd = format!(
            "tail -f -n 50 {} | grep --line-buffered '{}'",
            file_args.join(" "),
            grep_expr
        );
        let _ = Command::new("sh").arg("-c").arg(&tail_cmd).status();
    }

    Ok(())
}

/// Show expected log file locations
fn show_log_locations(daemon_log: &PathBuf, mcp_log: &PathBuf) {
    output::info("Expected log locations:");
    output::info(format!("  Daemon: {}", daemon_log.display()));
    output::info(format!("  MCP Server: {}", mcp_log.display()));

    #[cfg(target_os = "linux")]
    {
        output::info("  Also check: journalctl --user -u memexd");
    }

    #[cfg(target_os = "macos")]
    {
        output::info("  Also check: log show --predicate 'process == \"memexd\"' --last 1h");
    }
}

async fn errors(count: usize, component: Option<String>) -> Result<()> {
    output::section("Recent Errors");

    if let Some(comp) = &component {
        output::kv("Component filter", comp);
    }
    output::kv("Max errors", &count.to_string());
    output::separator();

    // Try to get errors from daemon health endpoint
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    let mut error_count = 0;

                    for comp_health in &health.components {
                        // Filter by component if specified
                        if let Some(filter) = &component {
                            if !comp_health
                                .component_name
                                .to_lowercase()
                                .contains(&filter.to_lowercase())
                            {
                                continue;
                            }
                        }

                        // Only show components with issues
                        if comp_health.status != 0 && !comp_health.message.is_empty() {
                            error_count += 1;
                            if error_count > count {
                                break;
                            }

                            let status = ServiceStatus::from_proto(comp_health.status);
                            output::status_line(&comp_health.component_name, status);
                            output::kv("  Message", &comp_health.message);
                        }
                    }

                    if error_count == 0 {
                        output::success("No errors found in daemon components");
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get health: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::warning("Cannot check errors without daemon connection");
        }
    }

    // Also check error log file
    output::separator();
    output::info("Checking error log file...");

    let err_log_path = "/tmp/memexd.err.log";
    let path = std::path::Path::new(err_log_path);

    if path.exists() {
        let output_result = Command::new("tail")
            .args(["-n", &count.to_string(), err_log_path])
            .output()?;

        if output_result.status.success() {
            let content = String::from_utf8_lossy(&output_result.stdout);
            if !content.trim().is_empty() {
                output::separator();
                output::info(format!("Last {} lines from {}:", count, err_log_path));
                println!("{}", content);
            } else {
                output::success("Error log is empty");
            }
        }
    } else {
        output::info("No error log file found at /tmp/memexd.err.log");
    }

    Ok(())
}

async fn queue_errors(count: usize, operation: Option<String>) -> Result<()> {
    output::section("Queue Processing Errors");

    if let Some(op) = &operation {
        output::kv("Operation filter", op);
    }
    output::kv("Max errors", &count.to_string());
    output::separator();

    // Query unified queue for failed items
    let db_path = dirs::home_dir()
        .map(|h| h.join(".workspace-qdrant/state.db"))
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;

    if !db_path.exists() {
        output::warning("Database not found - daemon may not have been started");
        return Ok(());
    }

    let conn = rusqlite::Connection::open(&db_path)?;

    // Build query with optional operation filter
    let mut query = String::from(
        "SELECT queue_id, item_type, op, tenant_id, last_error, retry_count, updated_at
         FROM unified_queue
         WHERE status = 'failed'",
    );

    if let Some(op) = &operation {
        query.push_str(&format!(" AND op = '{}'", op));
    }

    query.push_str(&format!(" ORDER BY updated_at DESC LIMIT {}", count));

    let mut stmt = conn.prepare(&query)?;
    let mut rows = stmt.query([])?;

    let mut error_count = 0;
    while let Some(row) = rows.next()? {
        error_count += 1;

        let queue_id: String = row.get(0)?;
        let item_type: String = row.get(1)?;
        let op: String = row.get(2)?;
        let tenant_id: String = row.get(3)?;
        let last_error: Option<String> = row.get(4)?;
        let retry_count: i32 = row.get(5)?;
        let updated_at: String = row.get(6)?;

        output::separator();
        output::kv("Queue ID", &queue_id[..8.min(queue_id.len())]);
        output::kv("Type", &item_type);
        output::kv("Operation", &op);
        output::kv("Tenant", &tenant_id);
        output::kv("Retries", &retry_count.to_string());
        output::kv("Updated", &updated_at);
        if let Some(err) = last_error {
            output::kv("Error", &err);
        }
    }

    if error_count == 0 {
        output::success("No failed queue items found");
    } else {
        output::separator();
        output::info(format!("Found {} failed queue items", error_count));
        output::info("Use 'wqm queue show <id>' for details");
    }

    Ok(())
}

async fn diagnose_language(language: &str, verbose: bool) -> Result<()> {
    output::section(format!("Language Diagnostics: {}", language));

    output::info("Running diagnostic checks...");
    output::separator();

    // 1. Check LSP server
    output::info("1. LSP Server Check");

    // Common LSP binary names for popular languages
    let lang_lower = language.to_lowercase();
    let fallback_lsp = format!("{}-language-server", lang_lower);
    let lsp_binaries: Vec<&str> = match lang_lower.as_str() {
        "python" => vec!["pyright", "pylsp", "pyright-langserver"],
        "javascript" | "typescript" | "js" | "ts" => {
            vec!["typescript-language-server", "tsserver"]
        }
        "rust" => vec!["rust-analyzer"],
        "go" | "golang" => vec!["gopls"],
        "java" => vec!["jdtls", "java-language-server"],
        "c" | "cpp" | "c++" => vec!["clangd", "ccls"],
        "ruby" => vec!["solargraph", "ruby-lsp"],
        "php" => vec!["phpactor", "psalm-language-server", "intelephense"],
        _ => vec![fallback_lsp.as_str()],
    };

    let mut lsp_found = false;
    for binary in &lsp_binaries {
        match which::which(binary) {
            Ok(path) => {
                output::success(format!("  Found: {} at {}", binary, path.display()));
                lsp_found = true;

                if verbose {
                    // Try to get version
                    if let Ok(output) = Command::new(binary).arg("--version").output() {
                        if output.status.success() {
                            let version = String::from_utf8_lossy(&output.stdout);
                            output::kv("    Version", version.trim());
                        }
                    }
                }
                break;
            }
            Err(_) => {
                if verbose {
                    output::info(format!("  Not found: {}", binary));
                }
            }
        }
    }

    if !lsp_found {
        output::warning(format!("  No LSP server found for {}", language));
        output::info("  Install suggestions:");
        match language.to_lowercase().as_str() {
            "python" => output::info("    npm install -g pyright"),
            "javascript" | "typescript" => {
                output::info("    npm install -g typescript-language-server typescript")
            }
            "rust" => output::info("    rustup component add rust-analyzer"),
            "go" => output::info("    go install golang.org/x/tools/gopls@latest"),
            _ => output::info(&format!(
                "    Search for {}-language-server",
                language
            )),
        }
    }

    output::separator();

    // 2. Check Tree-sitter grammar
    output::info("2. Tree-sitter Grammar Check");

    let grammar_paths = vec![
        dirs::data_local_dir()
            .map(|d| d.join("tree-sitter/lib"))
            .unwrap_or_default(),
        dirs::home_dir()
            .map(|h| h.join(".local/share/tree-sitter/lib"))
            .unwrap_or_default(),
        std::path::PathBuf::from("/usr/local/lib/tree-sitter"),
    ];

    let grammar_name = format!("{}.so", language.to_lowercase());
    let alt_grammar_name = format!("tree-sitter-{}.so", language.to_lowercase());
    let dylib_name = format!("{}.dylib", language.to_lowercase());
    let alt_dylib_name = format!("tree-sitter-{}.dylib", language.to_lowercase());

    let mut grammar_found = false;
    for base_path in &grammar_paths {
        if !base_path.exists() {
            continue;
        }

        for name in [&grammar_name, &alt_grammar_name, &dylib_name, &alt_dylib_name] {
            let grammar_path = base_path.join(name);
            if grammar_path.exists() {
                output::success(format!("  Found: {}", grammar_path.display()));
                grammar_found = true;
                break;
            }
        }
        if grammar_found {
            break;
        }
    }

    if !grammar_found {
        output::warning(format!("  No tree-sitter grammar found for {}", language));
        output::info("  Grammar search paths:");
        for path in &grammar_paths {
            output::info(&format!("    - {}", path.display()));
        }
        output::info("  Install with: wqm language install <language>");
    }

    output::separator();

    // 3. Check daemon language support status
    output::info("3. Daemon Language Support");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();

                    for comp in &health.components {
                        if comp.component_name.to_lowercase().contains(&lang_lower)
                            || comp.component_name.contains("lsp")
                            || comp.component_name.contains("grammar")
                        {
                            let status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&format!("  {}", comp.component_name), status);
                            if !comp.message.is_empty() && verbose {
                                output::kv("    Details", &comp.message);
                            }
                        }
                    }
                }
                Err(e) => {
                    output::warning(format!("  Could not get daemon status: {}", e));
                }
            }
        }
        Err(_) => {
            output::warning("  Daemon not running - cannot check language support status");
        }
    }

    output::separator();

    // 4. File extension mapping
    output::info("4. File Extension Mapping");

    let extensions = match language.to_lowercase().as_str() {
        "python" => vec![".py", ".pyi", ".pyw"],
        "javascript" => vec![".js", ".mjs", ".cjs"],
        "typescript" => vec![".ts", ".mts", ".cts", ".tsx"],
        "rust" => vec![".rs"],
        "go" => vec![".go"],
        "java" => vec![".java"],
        "c" => vec![".c", ".h"],
        "cpp" | "c++" => vec![".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"],
        "ruby" => vec![".rb", ".rake", ".gemspec"],
        "php" => vec![".php", ".phtml"],
        _ => vec![],
    };

    if !extensions.is_empty() {
        output::kv("  Extensions", &extensions.join(", "));
    } else {
        output::info(&format!(
            "  Unknown file extensions for {}",
            language
        ));
    }

    output::separator();
    output::info("Diagnostic Summary:");

    if lsp_found && grammar_found {
        output::success(format!(
            "  {} support appears fully configured",
            language
        ));
    } else if lsp_found {
        output::warning(format!(
            "  {} has LSP but missing grammar",
            language
        ));
    } else if grammar_found {
        output::warning(format!(
            "  {} has grammar but missing LSP",
            language
        ));
    } else {
        output::error(format!(
            "  {} support not configured",
            language
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike};
    use std::io::Write;
    use tempfile::TempDir;

    // ── parse_relative_duration ──────────────────────────────────────

    #[test]
    fn test_parse_relative_duration_seconds() {
        let d = parse_relative_duration("30s").unwrap();
        assert_eq!(d, Duration::seconds(30));
    }

    #[test]
    fn test_parse_relative_duration_minutes() {
        let d = parse_relative_duration("5m").unwrap();
        assert_eq!(d, Duration::minutes(5));
    }

    #[test]
    fn test_parse_relative_duration_hours() {
        let d = parse_relative_duration("2h").unwrap();
        assert_eq!(d, Duration::hours(2));
    }

    #[test]
    fn test_parse_relative_duration_days() {
        let d = parse_relative_duration("7d").unwrap();
        assert_eq!(d, Duration::days(7));
    }

    #[test]
    fn test_parse_relative_duration_weeks() {
        let d = parse_relative_duration("1w").unwrap();
        assert_eq!(d, Duration::weeks(1));
    }

    #[test]
    fn test_parse_relative_duration_milliseconds() {
        let d = parse_relative_duration("500ms").unwrap();
        assert_eq!(d, Duration::milliseconds(500));
    }

    #[test]
    fn test_parse_relative_duration_no_unit_defaults_to_seconds() {
        let d = parse_relative_duration("60").unwrap();
        assert_eq!(d, Duration::seconds(60));
    }

    #[test]
    fn test_parse_relative_duration_invalid_unit() {
        assert!(parse_relative_duration("5x").is_err());
    }

    #[test]
    fn test_parse_relative_duration_empty() {
        assert!(parse_relative_duration("").is_err());
    }

    // ── parse_since ─────────────────────────────────────────────────

    #[test]
    fn test_parse_since_relative() {
        let before = Utc::now();
        let cutoff = parse_since("1h").unwrap();
        let expected_approx = before - Duration::hours(1);
        // Within 2 seconds tolerance
        assert!((cutoff - expected_approx).num_seconds().abs() < 2);
    }

    #[test]
    fn test_parse_since_rfc3339() {
        let cutoff = parse_since("2025-06-15T10:30:00Z").unwrap();
        assert_eq!(cutoff.year(), 2025);
        assert_eq!(cutoff.month(), 6);
        assert_eq!(cutoff.hour(), 10);
    }

    #[test]
    fn test_parse_since_naive_datetime() {
        let cutoff = parse_since("2025-06-15T10:30:00").unwrap();
        assert_eq!(cutoff.year(), 2025);
    }

    #[test]
    fn test_parse_since_space_separated() {
        let cutoff = parse_since("2025-06-15 10:30:00").unwrap();
        assert_eq!(cutoff.year(), 2025);
    }

    #[test]
    fn test_parse_since_invalid() {
        assert!(parse_since("not-a-time").is_err());
    }

    // ── LogLevel ────────────────────────────────────────────────────

    #[test]
    fn test_log_level_from_string() {
        assert_eq!(LogLevel::from_str("ERROR"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("WARN"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("DEBUG"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("TRACE"), Some(LogLevel::Trace));
    }

    #[test]
    fn test_log_level_from_string_case_insensitive() {
        assert_eq!(LogLevel::from_str("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warn));
    }

    #[test]
    fn test_log_level_from_pino_numeric() {
        // pino uses numeric levels: 50=error, 40=warn, 30=info, 20=debug, 10=trace
        let error = serde_json::json!(50);
        assert_eq!(LogLevel::from_json(&error), Some(LogLevel::Error));

        let warn = serde_json::json!(40);
        assert_eq!(LogLevel::from_json(&warn), Some(LogLevel::Warn));

        let info = serde_json::json!(30);
        assert_eq!(LogLevel::from_json(&info), Some(LogLevel::Info));
    }

    #[test]
    fn test_log_level_is_error_or_warn() {
        assert!(LogLevel::Error.is_error_or_warn());
        assert!(LogLevel::Warn.is_error_or_warn());
        assert!(!LogLevel::Info.is_error_or_warn());
        assert!(!LogLevel::Debug.is_error_or_warn());
    }

    // ── LogEntry parsing ────────────────────────────────────────────

    #[test]
    fn test_log_entry_from_json_with_level() {
        let line = r#"{"timestamp":"2025-06-15T10:30:00Z","level":"ERROR","msg":"test"}"#;
        let entry = LogEntry::from_json_line(line, "daemon").unwrap();
        assert_eq!(entry.level, Some(LogLevel::Error));
        assert!(entry.parsed_time.is_some());
        assert_eq!(entry.component, "daemon");
    }

    #[test]
    fn test_log_entry_from_pino_json() {
        let line = r#"{"time":"2025-06-15T10:30:00Z","level":30,"msg":"hello"}"#;
        let entry = LogEntry::from_json_line(line, "mcp-server").unwrap();
        assert_eq!(entry.level, Some(LogLevel::Info));
    }

    #[test]
    fn test_log_entry_from_non_json() {
        let line = "plain text log line";
        let entry = LogEntry::from_json_line(line, "daemon").unwrap();
        assert!(entry.level.is_none());
        assert!(entry.parsed_time.is_none());
    }

    // ── LogFilter ───────────────────────────────────────────────────

    #[test]
    fn test_log_filter_errors_only() {
        let filter = LogFilter {
            errors_only: true,
            since: None,
            session: None,
        };

        let error_entry = LogEntry {
            timestamp: String::new(),
            parsed_time: None,
            level: Some(LogLevel::Error),
            component: "daemon".to_string(),
            session_id: None,
            raw_line: "error".to_string(),
        };
        assert!(filter.matches(&error_entry));

        let info_entry = LogEntry {
            level: Some(LogLevel::Info),
            ..error_entry
        };
        assert!(!filter.matches(&info_entry));
    }

    #[test]
    fn test_log_filter_since() {
        let cutoff = Utc::now() - Duration::hours(1);
        let filter = LogFilter {
            errors_only: false,
            since: Some(cutoff),
            session: None,
        };

        let recent = LogEntry {
            timestamp: String::new(),
            parsed_time: Some(Utc::now()),
            level: Some(LogLevel::Info),
            component: "daemon".to_string(),
            session_id: None,
            raw_line: "recent".to_string(),
        };
        assert!(filter.matches(&recent));

        let old = LogEntry {
            parsed_time: Some(Utc::now() - Duration::hours(2)),
            ..recent
        };
        assert!(!filter.matches(&old));
    }

    #[test]
    fn test_log_filter_session() {
        let filter = LogFilter {
            errors_only: false,
            since: None,
            session: Some("abc123".to_string()),
        };

        let matching = LogEntry {
            timestamp: String::new(),
            parsed_time: None,
            level: None,
            component: "mcp".to_string(),
            session_id: Some("session-abc123-xyz".to_string()),
            raw_line: "test".to_string(),
        };
        assert!(filter.matches(&matching));

        let no_session = LogEntry {
            session_id: None,
            ..matching
        };
        assert!(!filter.matches(&no_session));
    }

    // ── discover_log_files ──────────────────────────────────────────

    #[test]
    fn test_discover_log_files_current_only() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("daemon.jsonl");
        File::create(&log_path).unwrap();

        let files = discover_log_files(dir.path(), "daemon");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], log_path);
    }

    #[test]
    fn test_discover_log_files_with_rotated() {
        let dir = TempDir::new().unwrap();

        // Current + 2 rotated
        File::create(dir.path().join("daemon.jsonl")).unwrap();
        File::create(dir.path().join("daemon.jsonl.1")).unwrap();
        File::create(dir.path().join("daemon.jsonl.2")).unwrap();
        // .gz files should be ignored (not yet supported)
        File::create(dir.path().join("daemon.jsonl.3.gz")).unwrap();

        let files = discover_log_files(dir.path(), "daemon");
        assert_eq!(files.len(), 3);
        // Current first, then rotated in order
        assert!(files[0].ends_with("daemon.jsonl"));
        assert!(files[1].ends_with("daemon.jsonl.1"));
        assert!(files[2].ends_with("daemon.jsonl.2"));
    }

    #[test]
    fn test_discover_log_files_empty_dir() {
        let dir = TempDir::new().unwrap();
        let files = discover_log_files(dir.path(), "daemon");
        assert!(files.is_empty());
    }

    // ── read_log_file_filtered ──────────────────────────────────────

    #[test]
    fn test_read_log_file_filtered_errors_only() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            writeln!(f, r#"{{"timestamp":"2025-06-15T10:00:00Z","level":"INFO","msg":"info line"}}"#).unwrap();
            writeln!(f, r#"{{"timestamp":"2025-06-15T10:01:00Z","level":"ERROR","msg":"error line"}}"#).unwrap();
            writeln!(f, r#"{{"timestamp":"2025-06-15T10:02:00Z","level":"WARN","msg":"warn line"}}"#).unwrap();
            writeln!(f, r#"{{"timestamp":"2025-06-15T10:03:00Z","level":"DEBUG","msg":"debug line"}}"#).unwrap();
        }

        let filter = LogFilter {
            errors_only: true,
            since: None,
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 100, &filter);
        assert_eq!(entries.len(), 2);
        assert!(entries[0].raw_line.contains("error line"));
        assert!(entries[1].raw_line.contains("warn line"));
    }

    #[test]
    fn test_read_log_file_filtered_since() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            // Old entry (2020)
            writeln!(f, r#"{{"timestamp":"2020-01-01T00:00:00Z","level":"INFO","msg":"old"}}"#).unwrap();
            // Recent entry
            let recent = Utc::now().to_rfc3339();
            writeln!(f, r#"{{"timestamp":"{}","level":"INFO","msg":"recent"}}"#, recent).unwrap();
        }

        let filter = LogFilter {
            errors_only: false,
            since: Some(Utc::now() - Duration::hours(1)),
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 100, &filter);
        assert_eq!(entries.len(), 1);
        assert!(entries[0].raw_line.contains("recent"));
    }

    #[test]
    fn test_read_log_file_filtered_max_lines() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            for i in 0..20 {
                writeln!(f, r#"{{"timestamp":"2025-06-15T10:{:02}:00Z","level":"INFO","msg":"line {}"}}"#, i, i).unwrap();
            }
        }

        let filter = LogFilter {
            errors_only: false,
            since: None,
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 5, &filter);
        assert_eq!(entries.len(), 5);
        // Should be the last 5 lines (15-19)
        assert!(entries[0].raw_line.contains("line 15"));
        assert!(entries[4].raw_line.contains("line 19"));
    }

    #[test]
    fn test_read_log_files_filtered_with_rotated() {
        let dir = TempDir::new().unwrap();

        // Write old entries to rotated file
        {
            let mut f = File::create(dir.path().join("test.jsonl.1")).unwrap();
            writeln!(f, r#"{{"timestamp":"2025-06-15T08:00:00Z","level":"ERROR","msg":"old error"}}"#).unwrap();
        }
        // Write recent entries to current file
        {
            let mut f = File::create(dir.path().join("test.jsonl")).unwrap();
            let recent = Utc::now().to_rfc3339();
            writeln!(f, r#"{{"timestamp":"{}","level":"ERROR","msg":"new error"}}"#, recent).unwrap();
        }

        // With --since, should read from both files
        let filter = LogFilter {
            errors_only: true,
            since: Some(Utc::now() - Duration::days(365)),
            session: None,
        };
        let entries = read_log_files_filtered(dir.path(), "test", "test", 100, &filter);
        assert_eq!(entries.len(), 2);
    }

    // ── Combined filter ─────────────────────────────────────────────

    #[test]
    fn test_combined_errors_only_and_since() {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("test.jsonl");
        {
            let mut f = File::create(&log_path).unwrap();
            // Old error
            writeln!(f, r#"{{"timestamp":"2020-01-01T00:00:00Z","level":"ERROR","msg":"old error"}}"#).unwrap();
            // Recent info
            let recent = Utc::now().to_rfc3339();
            writeln!(f, r#"{{"timestamp":"{}","level":"INFO","msg":"recent info"}}"#, recent).unwrap();
            // Recent error
            let recent2 = Utc::now().to_rfc3339();
            writeln!(f, r#"{{"timestamp":"{}","level":"ERROR","msg":"recent error"}}"#, recent2).unwrap();
        }

        let filter = LogFilter {
            errors_only: true,
            since: Some(Utc::now() - Duration::hours(1)),
            session: None,
        };
        let entries = read_log_file_filtered(&log_path, "test", 100, &filter);
        assert_eq!(entries.len(), 1);
        assert!(entries[0].raw_line.contains("recent error"));
    }
}
