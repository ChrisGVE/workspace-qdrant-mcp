//! Rename-tenant subcommand handler

use std::io::Write as _;

use anyhow::Result;

use super::VALID_COLLECTIONS;
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::RenameTenantRequest;
use crate::output;

/// Rename a tenant_id across SQLite (via daemon gRPC)
pub async fn execute(old_id: String, new_id: String, yes: bool) -> Result<()> {
    // The old id may be given as a project name / partial input; the
    // confirmation prompt below always shows the RESOLVED tenant id.
    let old_id = crate::data::tenants::resolve_tenant(&old_id)?;
    output::section("Tenant Rename");
    output::kv("From", &old_id);
    output::kv("To", &new_id);
    output::separator();
    output::warning(
        "This will rename the tenant_id in SQLite tables \
         (watch_folders, unified_queue, tracked_files).",
    );
    output::info("Qdrant payloads (project_id field) will NOT be updated automatically.");
    output::info(
        "To update Qdrant, reset the affected collections after rename: \
         wqm collections reset projects",
    );
    println!();

    if !yes && !confirm_rename(&old_id)? {
        return Ok(());
    }

    let mut client = ensure_daemon_available().await?;

    let request = RenameTenantRequest {
        old_tenant_id: old_id.to_string(),
        new_tenant_id: new_id.to_string(),
        collections: VALID_COLLECTIONS.iter().map(|s| s.to_string()).collect(),
    };

    let response = client.project().rename_tenant(request).await?;
    let resp = response.into_inner();

    if resp.success {
        output::success(format!(
            "Renamed '{}' -> '{}': {} SQLite rows updated",
            old_id, new_id, resp.sqlite_rows_updated
        ));
    } else {
        output::error(format!("Rename failed: {}", resp.message));
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
