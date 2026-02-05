//! Debug command - consolidated diagnostics
//!
//! Provides diagnostic tools for troubleshooting the daemon and services.
//! Subcommands: logs, errors, queue-errors, language

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
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
        } => logs(lines, follow, component, session, json).await,
        DebugCommand::Errors { count, component } => errors(count, component).await,
        DebugCommand::QueueErrors { count, operation } => queue_errors(count, operation).await,
        DebugCommand::Language { language, verbose } => diagnose_language(&language, verbose).await,
    }
}

/// Returns the canonical OS-specific log directory for workspace-qdrant logs.
fn get_canonical_log_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        env::var("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| env::temp_dir())
                    .join(".local")
                    .join("state")
            })
            .join("workspace-qdrant")
            .join("logs")
    }

    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join("Library")
            .join("Logs")
            .join("workspace-qdrant")
    }

    #[cfg(target_os = "windows")]
    {
        dirs::data_local_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join("workspace-qdrant")
            .join("logs")
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        dirs::home_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join(".workspace-qdrant")
            .join("logs")
    }
}

/// Represents a parsed log entry with timestamp and component info
#[derive(Debug)]
struct LogEntry {
    timestamp: String,
    component: String,
    session_id: Option<String>,
    raw_line: String,
}

impl LogEntry {
    /// Try to parse a JSON log line
    fn from_json_line(line: &str, component: &str) -> Option<Self> {
        // Parse JSON to extract timestamp and session_id
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            let timestamp = json
                .get("timestamp")
                .or_else(|| json.get("time"))
                .or_else(|| json.get("ts"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let session_id = json
                .get("session_id")
                .and_then(|v| v.as_str())
                .map(String::from);

            Some(LogEntry {
                timestamp,
                component: component.to_string(),
                session_id,
                raw_line: line.to_string(),
            })
        } else {
            // Non-JSON line - use current time as fallback
            Some(LogEntry {
                timestamp: String::new(),
                component: component.to_string(),
                session_id: None,
                raw_line: line.to_string(),
            })
        }
    }
}

/// Read log entries from a file (last N lines)
fn read_log_file(path: &PathBuf, component: &str, max_lines: usize) -> Vec<LogEntry> {
    let mut entries = Vec::new();

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();

        // Take last N lines
        let start = lines.len().saturating_sub(max_lines);
        for line in &lines[start..] {
            if let Some(entry) = LogEntry::from_json_line(line, component) {
                entries.push(entry);
            }
        }
    }

    entries
}

async fn logs(
    lines: usize,
    follow: bool,
    component: LogComponent,
    session: Option<String>,
    json_output: bool,
) -> Result<()> {
    let log_dir = get_canonical_log_dir();
    let daemon_log = log_dir.join("daemon.jsonl");
    let mcp_log = log_dir.join("mcp-server.jsonl");

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
        output::separator();
    }

    // Handle follow mode
    if follow {
        return follow_logs(component, &daemon_log, &mcp_log, session).await;
    }

    // Collect log entries based on component filter
    let mut all_entries: Vec<LogEntry> = Vec::new();

    match component {
        LogComponent::All => {
            if daemon_log.exists() {
                all_entries.extend(read_log_file(&daemon_log, "daemon", lines));
            }
            if mcp_log.exists() {
                all_entries.extend(read_log_file(&mcp_log, "mcp-server", lines));
            }
        }
        LogComponent::Daemon => {
            if daemon_log.exists() {
                all_entries.extend(read_log_file(&daemon_log, "daemon", lines));
            }
        }
        LogComponent::McpServer => {
            if mcp_log.exists() {
                all_entries.extend(read_log_file(&mcp_log, "mcp-server", lines));
            }
        }
    }

    // Filter by session if specified
    if let Some(ref session_filter) = session {
        all_entries.retain(|e| {
            e.session_id
                .as_ref()
                .map(|s| s.contains(session_filter))
                .unwrap_or(false)
        });
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

    // Use tail -f with multiple files
    let mut cmd = Command::new("tail");
    cmd.arg("-f");
    cmd.arg("-n");
    cmd.arg("10"); // Show last 10 lines initially

    for path in &files_to_follow {
        cmd.arg(path);
    }

    // If session filter specified, pipe through grep
    if let Some(sid) = session {
        // Use shell to pipe tail through grep
        let file_args: Vec<String> = files_to_follow
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        let tail_cmd = format!(
            "tail -f -n 10 {} | grep --line-buffered '{}'",
            file_args.join(" "),
            sid
        );

        let _ = Command::new("sh").arg("-c").arg(&tail_cmd).status();
    } else {
        let _ = cmd.status();
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
