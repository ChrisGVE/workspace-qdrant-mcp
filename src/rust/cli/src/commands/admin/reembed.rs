//! `wqm admin reembed` — trigger the daemon's full re-embed pipeline.
//!
//! Calls `AdminWriteService.TriggerReembed` over gRPC. Refuses to run
//! without `--confirm` since the operation is destructive (drops and
//! recreates the four canonical Qdrant collections).

use anyhow::{Context, Result};

use crate::output;

/// Execute the `wqm admin reembed` subcommand.
pub async fn execute(confirm: bool) -> Result<()> {
    if !confirm {
        output::error(
            "wqm admin reembed will drop and recreate all four canonical \
             Qdrant collections (projects, libraries, rules, scratchpad) at \
             the configured embedding output_dim, then re-enqueue all indexed \
             sources for re-embedding.\n\nRe-run with --confirm to proceed.",
        );
        anyhow::bail!("--confirm flag required for wqm admin reembed");
    }

    use crate::grpc::ensure_daemon_available;
    use crate::grpc::proto::TriggerReembedRequest;

    let mut grpc_client = ensure_daemon_available().await?;
    let response = grpc_client
        .admin_write()
        .trigger_reembed(TriggerReembedRequest { confirm: true })
        .await
        .context("TriggerReembed RPC failed")?
        .into_inner();

    output::print_title("Re-embed pipeline triggered");
    println!("  files re-enqueued       : {}", response.files_enqueued);
    println!("  rules re-enqueued       : {}", response.rules_enqueued);
    println!(
        "  scratchpad re-enqueued  : {}",
        response.scratchpad_enqueued
    );
    if !response.message.is_empty() {
        println!();
        println!("  {}", response.message);
    }
    println!();
    println!(
        "{}",
        output::dim_style(
            "Monitor progress with `wqm queue stats` and `wqm status health`. \
             Re-embedding completes when the queue drains."
        )
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn confirm_flag_required() {
        let err = execute(false).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("--confirm"),
            "error must mention --confirm flag, got: {msg}"
        );
    }
}
