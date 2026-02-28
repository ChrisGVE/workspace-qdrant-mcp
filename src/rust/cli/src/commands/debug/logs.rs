//! Logs subcommand for viewing daemon and MCP server logs
//!
//! Supports merged multi-component log viewing, filtering by level/session/time,
//! JSON output, and real-time follow mode via `tail -f`.

use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;
use wqm_common::timestamps;

use super::log_parsing::{
    get_canonical_log_dir, parse_since, read_log_files_filtered, LogFilter,
};
use super::LogComponent;
use crate::output;

/// View daemon and MCP server logs (merged from canonical paths).
pub async fn logs(
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
    let mut all_entries = collect_entries(&log_dir, component, lines, &filter);

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

/// Collect log entries from the appropriate component files.
fn collect_entries(
    log_dir: &std::path::Path,
    component: LogComponent,
    lines: usize,
    filter: &LogFilter,
) -> Vec<super::log_parsing::LogEntry> {
    let mut all_entries = Vec::new();

    match component {
        LogComponent::All => {
            all_entries.extend(read_log_files_filtered(
                log_dir, "daemon", "daemon", lines, filter,
            ));
            all_entries.extend(read_log_files_filtered(
                log_dir,
                "mcp-server",
                "mcp-server",
                lines,
                filter,
            ));
        }
        LogComponent::Daemon => {
            all_entries.extend(read_log_files_filtered(
                log_dir, "daemon", "daemon", lines, filter,
            ));
        }
        LogComponent::McpServer => {
            all_entries.extend(read_log_files_filtered(
                log_dir,
                "mcp-server",
                "mcp-server",
                lines,
                filter,
            ));
        }
    }

    all_entries
}

/// Follow logs in real-time.
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

/// Show expected log file locations.
pub fn show_log_locations(daemon_log: &PathBuf, mcp_log: &PathBuf) {
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
