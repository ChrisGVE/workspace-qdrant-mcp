//! Output formatting module
//!
//! Provides JSON and table output formatting with colored terminal support.
//! Used across all CLI commands for consistent output.

use colored::Colorize;
use serde::Serialize;
use std::fmt::Display;
use tabled::{settings::Style, Table, Tabled};

use crate::config::OutputFormat;

/// Format and print a success message
pub fn success(message: impl Display) {
    println!("{} {}", "✓".green(), message);
}

/// Format and print an error message
pub fn error(message: impl Display) {
    eprintln!("{} {}", "✗".red(), message);
}

/// Format and print a warning message
pub fn warning(message: impl Display) {
    eprintln!("{} {}", "!".yellow(), message);
}

/// Format and print an info message
pub fn info(message: impl Display) {
    println!("{} {}", "ℹ".blue(), message);
}

/// Format and print data based on output format
pub fn print_data<T: Serialize + Tabled>(data: &[T], format: OutputFormat) {
    match format {
        OutputFormat::Json => print_json(data),
        OutputFormat::Table => print_table(data),
        OutputFormat::Plain => print_plain(data),
    }
}

/// Print data as JSON
pub fn print_json<T: Serialize + ?Sized>(data: &T) {
    match serde_json::to_string_pretty(data) {
        Ok(json) => println!("{}", json),
        Err(e) => error(format!("Failed to serialize to JSON: {}", e)),
    }
}

/// Print data as a formatted table
pub fn print_table<T: Tabled>(data: &[T]) {
    if data.is_empty() {
        info("No data to display");
        return;
    }

    let table = Table::new(data).with(Style::rounded()).to_string();
    println!("{}", table);
}

/// Print data as plain text (one item per line)
pub fn print_plain<T: Tabled>(data: &[T]) {
    if data.is_empty() {
        return;
    }

    // Use table but with minimal style
    let table = Table::new(data).with(Style::blank()).to_string();
    println!("{}", table);
}

/// Print a single key-value pair
pub fn kv(key: impl Display, value: impl Display) {
    println!("{}: {}", key.to_string().bold(), value);
}

/// Print a status line with colored status indicator
pub fn status_line(label: impl Display, status: ServiceStatus) {
    let (indicator, color_status) = match status {
        ServiceStatus::Healthy => ("●".green(), "healthy".green()),
        ServiceStatus::Degraded => ("●".yellow(), "degraded".yellow()),
        ServiceStatus::Unhealthy => ("●".red(), "unhealthy".red()),
        ServiceStatus::Unknown => ("○".dimmed(), "unknown".dimmed()),
    };
    println!("{} {}: {}", indicator, label, color_status);
}

/// Service health status for colored output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl ServiceStatus {
    /// Create from proto ServiceStatus enum value
    pub fn from_proto(value: i32) -> Self {
        match value {
            1 => Self::Healthy,
            2 => Self::Degraded,
            3 => Self::Unhealthy,
            _ => Self::Unknown,
        }
    }
}

/// Print a horizontal separator
pub fn separator() {
    println!("{}", "─".repeat(60).dimmed());
}

/// Print a section header
pub fn section(title: impl Display) {
    println!("\n{}", title.to_string().bold().underline());
}

/// Format bytes as human-readable size
pub fn format_bytes(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable string
pub fn format_duration(seconds: u64) -> String {
    if seconds >= 86400 {
        format!("{}d {}h", seconds / 86400, (seconds % 86400) / 3600)
    } else if seconds >= 3600 {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}s", seconds)
    }
}

/// Progress spinner for long-running operations
pub struct Spinner {
    message: String,
    frames: Vec<&'static str>,
    current: usize,
}

impl Spinner {
    /// Create a new spinner with the given message
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            current: 0,
        }
    }

    /// Advance the spinner to the next frame
    pub fn tick(&mut self) {
        self.current = (self.current + 1) % self.frames.len();
        print!("\r{} {} ", self.frames[self.current].cyan(), self.message);
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    /// Finish the spinner with a success message
    pub fn success(self, message: impl Display) {
        println!("\r{} {}", "✓".green(), message);
    }

    /// Finish the spinner with a failure message
    pub fn fail(self, message: impl Display) {
        println!("\r{} {}", "✗".red(), message);
    }

    /// Clear the spinner line
    pub fn clear(&self) {
        print!("\r{}\r", " ".repeat(self.message.len() + 4));
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Print a waiting message (simpler than spinner for quick operations)
pub fn waiting(message: impl Display) {
    print!("{} {}...", "⏳".yellow(), message);
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// Complete a waiting message with success
pub fn waiting_done() {
    println!(" {}", "done".green());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3700), "1h 1m");
        assert_eq!(format_duration(90000), "1d 1h");
    }

    #[test]
    fn test_service_status_from_proto() {
        assert_eq!(ServiceStatus::from_proto(1), ServiceStatus::Healthy);
        assert_eq!(ServiceStatus::from_proto(2), ServiceStatus::Degraded);
        assert_eq!(ServiceStatus::from_proto(3), ServiceStatus::Unhealthy);
        assert_eq!(ServiceStatus::from_proto(0), ServiceStatus::Unknown);
        assert_eq!(ServiceStatus::from_proto(99), ServiceStatus::Unknown);
    }
}
