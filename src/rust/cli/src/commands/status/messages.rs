//! Messages and errors subcommands.

use anyhow::Result;
use clap::Subcommand;

use crate::grpc::client::DaemonClient;
use crate::output;

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
            output::section("System Messages");
            output::info("System messages available in daemon logs:");
            output::info("  macOS: /tmp/memexd.out.log, /tmp/memexd.err.log");
            output::info("  Linux: journalctl --user -u memexd");
            output::separator();
            output::info("Use 'wqm service logs' to view recent messages");
        }
        Some(MessageAction::Clear) => {
            output::info(
                "Message clearing not supported - logs are managed by the system",
            );
        }
    }
    Ok(())
}

/// Show recent error metrics from the daemon.
pub async fn errors(limit: usize) -> Result<()> {
    output::section(format!("Recent Errors (last {})", limit));

    output::info("Error tracking available via daemon logs:");
    output::info(&format!("  Use: wqm service logs -n {}", limit));
    output::info("  Or: grep -i error /tmp/memexd.err.log | tail -n {}", );

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().get_metrics(()).await {
                Ok(response) => {
                    let metrics_resp = response.into_inner();

                    for metric in &metrics_resp.metrics {
                        if metric.name.contains("error")
                            || metric.name.contains("failed")
                        {
                            output::kv(&metric.name, &format!("{:.0}", metric.value));
                        }
                    }
                }
                Err(_) => {}
            }
        }
        Err(_) => {
            output::warning("Cannot connect to daemon for error metrics");
        }
    }

    Ok(())
}
