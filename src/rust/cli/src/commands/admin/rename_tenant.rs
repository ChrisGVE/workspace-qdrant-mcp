//! Rename-tenant subcommand handler

use std::io::Write as _;

use anyhow::{Context, Result};

use super::VALID_COLLECTIONS;
use crate::grpc::client::DaemonClient;
use crate::grpc::proto::RenameTenantRequest;
use crate::output;

/// Rename a tenant_id across SQLite (via daemon gRPC) and verify
pub async fn execute(old_id: String, new_id: String, yes: bool) -> Result<()> {
    output::section("Tenant Rename");
    output::kv("From", &old_id);
    output::kv("To", &new_id);
    output::separator();
    output::warning("This will rename the tenant_id in SQLite tables (watch_folders, unified_queue, tracked_files).");
    output::info("Qdrant payloads (project_id field) will NOT be updated automatically.");
    output::info("To update Qdrant, reset the affected collections after rename: wqm collections reset projects");
    println!();

    if !yes && !confirm_rename(&old_id)? {
        return Ok(());
    }

    match DaemonClient::connect_default().await {
        Ok(mut client) => handle_rename_via_grpc(&mut client, &old_id, &new_id).await,
        Err(_) => {
            output::warning("Daemon not running, using direct SQLite access");
            handle_rename_fallback(&old_id, &new_id).await;
        }
    }

    Ok(())
}

/// Prompt the user to type the old tenant ID to confirm. Returns `false` on abort.
fn confirm_rename(old_id: &str) -> Result<bool> {
    print!("  To confirm, type the old tenant_id '{}': ", old_id);
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input != old_id {
        output::error(format!("Expected '{}', got '{}'. Aborting.", old_id, input));
        return Ok(false);
    }
    println!();
    Ok(true)
}

/// Attempt rename through the daemon gRPC service, falling back to direct SQLite on error.
async fn handle_rename_via_grpc(client: &mut DaemonClient, old_id: &str, new_id: &str) {
    let request = RenameTenantRequest {
        old_tenant_id: old_id.to_string(),
        new_tenant_id: new_id.to_string(),
        collections: VALID_COLLECTIONS.iter().map(|s| s.to_string()).collect(),
    };

    match client.project().rename_tenant(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            if resp.success {
                output::success(format!(
                    "Renamed '{}' -> '{}': {} SQLite rows updated",
                    old_id, new_id, resp.sqlite_rows_updated
                ));
            } else {
                output::error(format!("Rename failed: {}", resp.message));
            }
        }
        Err(e) => {
            output::error(format!("gRPC call failed: {}", e));
            output::info("Falling back to direct SQLite rename...");
            handle_rename_fallback(old_id, new_id).await;
        }
    }
}

/// Perform the rename directly against SQLite (used when daemon is unavailable).
async fn handle_rename_fallback(old_id: &str, new_id: &str) {
    match rename_direct(old_id, new_id).await {
        Ok(count) => output::success(format!(
            "Direct rename '{}' -> '{}': {} SQLite rows updated",
            old_id, new_id, count
        )),
        Err(e) => output::error(format!("Direct rename failed: {}", e)),
    }
}

/// Direct SQLite rename (fallback when daemon is not running)
async fn rename_direct(old_id: &str, new_id: &str) -> Result<usize> {
    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    let conn = rusqlite::Connection::open(&db_path).context("Failed to open state database")?;

    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;")
        .context("Failed to set SQLite pragmas")?;

    let tx = conn
        .unchecked_transaction()
        .context("Failed to begin transaction")?;

    let mut total = 0usize;

    // Update watch_folders
    let count = tx
        .execute(
            "UPDATE watch_folders SET tenant_id = ?1 WHERE tenant_id = ?2",
            rusqlite::params![new_id, old_id],
        )
        .context("Failed to update watch_folders")?;
    total += count;

    // Update unified_queue
    let count = tx
        .execute(
            "UPDATE unified_queue SET tenant_id = ?1 WHERE tenant_id = ?2",
            rusqlite::params![new_id, old_id],
        )
        .context("Failed to update unified_queue")?;
    total += count;

    // Update tracked_files (may not exist)
    match tx.execute(
        "UPDATE tracked_files SET tenant_id = ?1 WHERE tenant_id = ?2",
        rusqlite::params![new_id, old_id],
    ) {
        Ok(count) => total += count,
        Err(_) => {} // Table may not exist, ignore
    }

    tx.commit().context("Failed to commit rename")?;

    Ok(total)
}
