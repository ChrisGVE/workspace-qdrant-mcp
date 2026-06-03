//! Messages and errors subcommands.
//!
//! Columnar template per cli-feedback.md.

use anyhow::Result;
use clap::Subcommand;

use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;

/// Message subcommands
#[derive(Subcommand)]
pub enum MessageAction {
    /// List all messages
    List,
    /// Clear all messages
    Clear,
}

/// Show or manage system messages.
pub async fn messages(action: Option<MessageAction>) -> Result<()> {
    match action {
        None | Some(MessageAction::List) => {
            canvas::print_title("System Messages");
            canvas::print_blank();

            ColumnarBuilder::new()
                .kv(
                    "Log Location (macOS)",
                    "/tmp/memexd.out.log, /tmp/memexd.err.log",
                )
                .kv("Log Location (Linux)", "journalctl --user -u memexd")
                .kv("View Recent", "wqm service logs")
                .render();
        }
        Some(MessageAction::Clear) => {
            canvas::print_title("Clear Messages");
            canvas::print_blank();

            ColumnarBuilder::new()
                .kv("Status", "not supported — logs are managed by the system")
                .render();
        }
    }
    Ok(())
}

/// Show recent error metrics from the daemon.
pub async fn errors(limit: usize) -> Result<()> {
    let error_metrics = fetch_error_metrics().await;

    canvas::print_title(&format!("Recent Errors (Last {})", limit));
    canvas::print_blank();

    let mut builder = ColumnarBuilder::new()
        .kv("View Errors", format!("wqm service logs -n {}", limit))
        .kv(
            "Grep Errors",
            format!("grep -i error /tmp/memexd.err.log | tail -n {}", limit),
        );

    if let Some(metrics) = error_metrics {
        if !metrics.is_empty() {
            builder = builder.section(Some("Error Metrics"));
            for (name, value) in &metrics {
                builder = builder.kv(name, format!("{:.0}", value));
            }
        }
    }

    builder.render();

    Ok(())
}

/// Fetch error-related metrics from daemon, returning None if unreachable.
async fn fetch_error_metrics() -> Option<Vec<(String, f64)>> {
    let mut client = crate::grpc::connect_default().await.ok()?;
    let response = client.system().get_metrics(()).await.ok()?;
    let metrics_resp = response.into_inner();

    let error_metrics: Vec<(String, f64)> = metrics_resp
        .metrics
        .iter()
        .filter(|m| m.name.contains("error") || m.name.contains("failed"))
        .map(|m| (m.name.clone(), m.value))
        .collect();

    if error_metrics.is_empty() {
        None
    } else {
        Some(error_metrics)
    }
}
