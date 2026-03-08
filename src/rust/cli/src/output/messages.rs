//! Terminal message helpers.
//!
//! Provides simple coloured output helpers (`success`, `error`, `warning`,
//! `info`), structural helpers (`separator`, `section`, `kv`),
//! `ServiceStatus` with its coloured `status_line`, and the `confirm`
//! interactive prompt.

use std::fmt::Display;

use colored::Colorize;

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

/// Print a single key-value pair
pub fn kv(key: impl Display, value: impl Display) {
    println!("{}: {}", key.to_string().bold(), value);
}

/// Print a horizontal separator
pub fn separator() {
    println!("{}", "─".repeat(60).dimmed());
}

/// Print a section header
pub fn section(title: impl Display) {
    println!("\n{}", title.to_string().bold().underline());
}

/// Print a status line with colored status indicator
pub fn status_line(label: impl Display, status: ServiceStatus) {
    let (indicator, color_status) = match status {
        ServiceStatus::Healthy => ("●".green(), "healthy".green()),
        ServiceStatus::Degraded => ("●".yellow(), "degraded".yellow()),
        ServiceStatus::Unhealthy => ("●".red(), "unhealthy".red()),
        ServiceStatus::Inactive => ("○".dimmed(), "inactive".dimmed()),
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
    Inactive,
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

/// Prompt user for yes/no confirmation, defaulting to No.
///
/// Returns true only if the user types "y" or "yes" (case-insensitive).
pub fn confirm(message: &str) -> bool {
    use std::io::{self, Write};
    print!("{} [y/N] ", message);
    io::stdout().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_status_from_proto() {
        assert_eq!(ServiceStatus::from_proto(1), ServiceStatus::Healthy);
        assert_eq!(ServiceStatus::from_proto(2), ServiceStatus::Degraded);
        assert_eq!(ServiceStatus::from_proto(3), ServiceStatus::Unhealthy);
        assert_eq!(ServiceStatus::from_proto(0), ServiceStatus::Unknown);
        assert_eq!(ServiceStatus::from_proto(99), ServiceStatus::Unknown);
    }
}
