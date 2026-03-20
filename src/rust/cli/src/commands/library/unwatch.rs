//! Library unwatch subcommand

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::UnwatchLibraryRequest;
use crate::output;

/// Stop watching a library (preserves indexed content)
pub async fn execute(tag: &str) -> Result<()> {
    output::section(format!("Unwatch Library: {}", tag));

    let mut client = ensure_daemon_available().await?;

    let response = client
        .library_write()
        .unwatch_library(UnwatchLibraryRequest {
            tag: tag.to_string(),
        })
        .await?
        .into_inner();

    if response.affected_count > 0 {
        output::success(format!("Library '{}' watching disabled", tag));
        output::info("Existing indexed content is preserved.");
        output::info("To re-enable: wqm library watch <tag> <path>");
        output::info("To remove completely: wqm library remove <tag>");
    } else {
        output::error(format!("Library '{}' not found or already unwatched", tag));
    }

    Ok(())
}
