//! Service restart subcommand

use anyhow::Result;

use crate::output;

/// Restart the daemon service (stop then start)
///
/// Stop waits for confirmed shutdown before starting the new instance.
/// This prevents port conflicts and ensures clean state transitions.
pub async fn execute() -> Result<()> {
    output::info("Restarting daemon...");
    super::stop::execute().await?;
    // Small grace period after confirmed shutdown to release OS resources
    // (file locks, TCP port TIME_WAIT, etc.)
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    super::start::execute().await?;
    Ok(())
}
