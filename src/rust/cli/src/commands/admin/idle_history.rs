//! Idle-history subcommand handler

use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::BufRead;
use tabled::Tabled;

use crate::output;
use crate::output::ColumnHints;

#[derive(Deserialize)]
struct Entry {
    timestamp: String,
    from_mode: String,
    to_mode: String,
    idle_seconds: f64,
    duration_in_previous_secs: f64,
}

#[derive(Tabled)]
struct IdleHistoryRow {
    #[tabled(rename = "Timestamp")]
    timestamp: String,
    #[tabled(rename = "From")]
    from_mode: String,
    #[tabled(rename = "To")]
    to_mode: String,
    #[tabled(rename = "Idle")]
    idle: String,
    #[tabled(rename = "Duration")]
    duration: String,
}

impl ColumnHints for IdleHistoryRow {
    fn content_columns() -> &'static [usize] {
        &[]
    }
}

/// Show idle state transition history and flip-flop analysis
pub fn execute(hours: f64, script: bool, no_headers: bool) -> Result<()> {
    let config_dir = wqm_common::paths::get_config_dir().map_err(|e| anyhow::anyhow!("{}", e))?;
    let history_path = config_dir.join("idle_history.jsonl");

    if !history_path.exists() {
        output::info("No idle history file found. The daemon records transitions automatically.");
        output::kv("Expected path", history_path.display().to_string());
        return Ok(());
    }

    let file = std::fs::File::open(&history_path).context("Failed to open idle history file")?;
    let reader = std::io::BufReader::new(file);

    // Compute cutoff timestamp using wqm_common (avoids chrono dependency)
    let cutoff_str = wqm_common::timestamps::hours_ago(hours);

    let entries: Vec<Entry> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| serde_json::from_str::<Entry>(&line).ok())
        .filter(|e| e.timestamp >= cutoff_str)
        .collect();

    output::section(format!("Idle History (last {:.0}h)", hours));
    output::kv("File", history_path.display().to_string());
    output::kv("Transitions", entries.len().to_string());

    if entries.is_empty() {
        output::info("No transitions in the specified window.");
        return Ok(());
    }

    // Analysis
    let transitions_per_hour = entries.len() as f64 / hours;
    let avg_duration = entries
        .iter()
        .map(|e| e.duration_in_previous_secs)
        .sum::<f64>()
        / entries.len() as f64;
    let short_count = entries
        .iter()
        .filter(|e| e.duration_in_previous_secs < 30.0)
        .count();
    let is_flip_flopping = transitions_per_hour > 10.0;

    output::separator();
    output::kv(
        "Rate",
        format!("{:.1} transitions/hr", transitions_per_hour),
    );
    output::kv(
        "Avg mode duration",
        wqm_common::duration_fmt::format_duration(avg_duration, 0),
    );
    output::kv("Short (<30s)", short_count.to_string());

    if is_flip_flopping {
        output::separator();
        output::warning("Flip-flop detected! Consider increasing idle_cooloff_polls in config.");
        let recommended = ((transitions_per_hour / 10.0).ceil() as u32).saturating_sub(1);
        output::kv("Recommended +cooloff", format!("+{} polls", recommended));
    }

    // Show last 20 transitions in a table
    let tail: Vec<&Entry> = entries
        .iter()
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let idle_secs: Vec<f64> = tail.iter().map(|e| e.idle_seconds).collect();
    let dur_secs: Vec<f64> = tail.iter().map(|e| e.duration_in_previous_secs).collect();
    let idle_fmt = wqm_common::duration_fmt::format_duration_column(&idle_secs);
    let dur_fmt = wqm_common::duration_fmt::format_duration_column(&dur_secs);

    let rows: Vec<IdleHistoryRow> = tail
        .iter()
        .enumerate()
        .map(|(i, entry)| IdleHistoryRow {
            timestamp: wqm_common::timestamp_fmt::format_local(&entry.timestamp),
            from_mode: entry.from_mode.clone(),
            to_mode: entry.to_mode.clone(),
            idle: idle_fmt[i].clone(),
            duration: dur_fmt[i].clone(),
        })
        .collect();

    output::separator();
    if script {
        output::print_script(&rows, !no_headers);
    } else {
        output::print_table_auto(&rows);
    }

    Ok(())
}
