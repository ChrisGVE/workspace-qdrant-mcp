//! Service restart subcommand

use anyhow::Result;

use crate::output;

/// Restart the daemon service (stop then start)
pub async fn execute() -> Result<()> {
    output::info("Restarting daemon...");
    super::stop::execute().await?;
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    super::start::execute().await?;
    Ok(())
}
