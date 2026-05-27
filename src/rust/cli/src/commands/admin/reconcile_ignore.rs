//! `wqm admin reconcile-ignore` — re-apply current ignore rules without restart.
//!
//! Calls `AdminWriteService.ReapplyIgnoreRules` over gRPC. Iterates active
//! projects, compares `tracked_files` against the current global + per-project
//! ignore rules, and enqueues `file/delete` for newly-excluded paths and
//! `file/add` for newly-included paths. Use this after editing
//! `global.wqmignore` to avoid a daemon restart.

use anyhow::{Context, Result};

use crate::grpc::ensure_daemon_available;
use crate::output;

/// Execute the `wqm admin reconcile-ignore` subcommand.
pub async fn execute() -> Result<()> {
    let mut grpc_client = ensure_daemon_available().await?;
    let response = grpc_client
        .admin_write()
        .reapply_ignore_rules(())
        .await
        .context("ReapplyIgnoreRules RPC failed")?
        .into_inner();

    output::print_title("Ignore-rule reconciliation complete");
    println!("  projects processed     : {}", response.projects_processed);
    println!("  stale deletes enqueued : {}", response.stale_deleted);
    println!("  missing adds enqueued  : {}", response.missing_added);

    if response.stale_deleted == 0 && response.missing_added == 0 {
        println!();
        println!(
            "{}",
            output::dim_style("No changes — tracked_files already matches the current ignore set.")
        );
    } else {
        println!();
        println!(
            "{}",
            output::dim_style(
                "Monitor progress with `wqm queue stats`. The enqueued items \
                 drain through the normal queue processor."
            )
        );
    }

    Ok(())
}
