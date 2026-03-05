//! Service logs subcommand

use anyhow::{bail, Context, Result};

use crate::output;

use super::platform::log_directory;

/// View daemon logs
pub async fn execute(lines: usize, follow: bool, errors_only: bool) -> Result<()> {
    let log_file = resolve_log_file()?;

    output::info(format!("Log file: {}", log_file.display()));

    if follow {
        follow_logs(&log_file, lines, errors_only)
    } else {
        tail_logs(&log_file, lines, errors_only)
    }
}

/// Find the log file to display, checking multiple locations
fn resolve_log_file() -> Result<std::path::PathBuf> {
    let log_dir = log_directory()?;
    let log_file = log_dir.join("daemon.jsonl");

    if log_file.exists() {
        return Ok(log_file);
    }

    let alt = log_dir.join("daemon.log");
    if alt.exists() {
        return Ok(alt);
    }

    // Check launchd stdout/stderr log locations as last resort
    #[cfg(target_os = "macos")]
    {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        let launchd_log = home.join("Library/Logs/workspace-qdrant/daemon.out.log");
        if launchd_log.exists() {
            return Ok(launchd_log);
        }
    }

    bail!(
        "No log files found in {}\nHint: use 'wqm debug logs' for comprehensive log discovery",
        log_dir.display()
    );
}

/// Follow log output in real-time (tail -f)
fn follow_logs(log_file: &std::path::Path, lines: usize, errors_only: bool) -> Result<()> {
    if errors_only {
        follow_errors_only(log_file)
    } else {
        follow_all(log_file, lines)
    }
}

fn follow_errors_only(log_file: &std::path::Path) -> Result<()> {
    let mut child = std::process::Command::new("tail")
        .args(["-f", log_file.to_str().unwrap_or("")])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .context("Failed to start tail command")?;

    let stdout = child
        .stdout
        .take()
        .context("Failed to capture tail output")?;
    let reader = std::io::BufReader::new(stdout);
    use std::io::BufRead;
    for line in reader.lines() {
        let line = line?;
        let upper = line.to_uppercase();
        if upper.contains("ERROR") || upper.contains("WARN") {
            println!("{}", line);
        }
    }
    child.wait()?;
    Ok(())
}

fn follow_all(log_file: &std::path::Path, lines: usize) -> Result<()> {
    let args = vec!["-f".to_string(), "-n".to_string(), lines.to_string()];
    let status = std::process::Command::new("tail")
        .args(&args)
        .arg(log_file)
        .status()
        .context("Failed to run tail command")?;
    if !status.success() {
        bail!("tail command failed");
    }
    Ok(())
}

/// Read last N lines from the log file
fn tail_logs(log_file: &std::path::Path, lines: usize, errors_only: bool) -> Result<()> {
    let content = std::fs::read_to_string(log_file)
        .with_context(|| format!("Failed to read log file: {}", log_file.display()))?;

    let all_lines: Vec<&str> = content.lines().collect();
    let start = all_lines.len().saturating_sub(lines);
    let tail = &all_lines[start..];

    if errors_only {
        for line in tail {
            let upper = line.to_uppercase();
            if upper.contains("ERROR") || upper.contains("WARN") {
                println!("{}", line);
            }
        }
    } else {
        for line in tail {
            println!("{}", line);
        }
    }

    Ok(())
}
