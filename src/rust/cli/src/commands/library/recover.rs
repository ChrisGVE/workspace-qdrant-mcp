//! Re-point a library to a new source path (#140).
//!
//! `wqm library recover <tag> [--new-path <dir>] [--dry-run]`. The daemon's
//! `RecoverLibrary` RPC swaps the stored source path and rewrites every stored
//! file path old->new across SQLite and Qdrant. This command resolves the new
//! path to an absolute canonical form, ships the request, and renders the
//! result.

use std::path::PathBuf;

use anyhow::Result;

use super::helpers::canonical_from_cli_path;
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{RecoverLibraryRequest, RecoverLibraryResponse};
use crate::output;
use crate::output::style::home_to_tilde;

pub async fn execute(tag: &str, new_path: Option<PathBuf>, dry_run: bool) -> Result<()> {
    output::section(if dry_run {
        format!("Recover Library: {tag} (dry run)")
    } else {
        format!("Recover Library: {tag}")
    });

    // Resolve --new-path to an absolute canonical path (client-side), matching
    // the convention `wqm library add` uses for its path argument.
    let new_path_str = match new_path.as_ref() {
        Some(p) => {
            if !p.exists() {
                output::warning(format!(
                    "New path does not exist yet: {}",
                    home_to_tilde(&p.display().to_string())
                ));
            }
            Some(canonical_from_cli_path(p)?.into_string())
        }
        None => None,
    };

    if let Some(p) = new_path_str.as_ref() {
        output::kv("New path", &home_to_tilde(p));
        output::separator();
    }

    let mut client = ensure_daemon_available().await?;

    let response = client
        .library_write()
        .recover_library(RecoverLibraryRequest {
            tag: tag.to_string(),
            new_path: new_path_str,
            dry_run,
        })
        .await?
        .into_inner();

    render(response);
    Ok(())
}

fn render(r: RecoverLibraryResponse) {
    if !r.success {
        output::error(r.message);
        return;
    }
    if !r.changed {
        output::info(r.message);
        return;
    }

    output::kv(
        "Path",
        &format!(
            "{} -> {}",
            home_to_tilde(&r.old_path),
            home_to_tilde(&r.new_path)
        ),
    );
    output::kv("SQLite rows", &r.sqlite_rows_updated.to_string());
    output::kv("Qdrant points", &r.qdrant_points_updated.to_string());
    output::separator();

    if r.dry_run {
        output::info(r.message);
    } else {
        output::success(r.message);
    }
}
