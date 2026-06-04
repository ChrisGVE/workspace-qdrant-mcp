//! Rebalance-IDF subcommand: correct IDF drift in stored sparse vectors.
//!
//! Thin client wrapper (WI-f1, #82). The whole engine — reading
//! `corpus_statistics`/`sparse_vocabulary`, computing IDF with daemon BM25,
//! performing the Qdrant sparse-vector writes, and persisting `last_corrected_n`
//! — lives daemon-side in `workspace_qdrant_core::idf_rebalance` behind
//! `AdminWriteService::RebalanceIdf`. The CLI no longer opens state.db
//! read-write nor links the Qdrant write client; it issues one gRPC call.

use anyhow::Result;

use crate::output;

/// Run IDF drift correction for one or all collections via the daemon.
pub async fn execute(collection: Option<String>, dry_run: bool, min_growth_pct: f64) -> Result<()> {
    output::section("IDF Drift Correction");

    if dry_run {
        output::info("Dry-run mode — no vectors will be updated");
    }

    let mut client = crate::grpc::ensure_daemon_available().await?;
    let resp = client
        .rebalance_idf(collection, dry_run, min_growth_pct)
        .await
        .map_err(|e| anyhow::anyhow!("Rebalance failed: {}", e.message()))?;

    if resp.per_collection.is_empty() {
        output::info("No collections with corpus statistics found — nothing to do.");
        return Ok(());
    }

    for c in &resp.per_collection {
        match &c.skipped_reason {
            Some(reason) => {
                output::info(format!("  '{}': skipped ({})", c.collection, reason));
            }
            None => output::info(format!(
                "  '{}' (N={}): {} point(s) {}",
                c.collection,
                c.current_n,
                c.updated,
                if resp.dry_run {
                    "would be updated"
                } else {
                    "updated"
                },
            )),
        }
    }

    output::separator();
    if resp.dry_run {
        output::info(format!(
            "Dry-run complete. Would update {} point(s).",
            resp.total_updated
        ));
    } else {
        output::success(format!("Updated {} point(s) total.", resp.total_updated));
    }

    Ok(())
}
