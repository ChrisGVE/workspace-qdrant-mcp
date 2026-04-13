//! Library config subcommand

use anyhow::Result;

use super::helpers::LibraryMode;
use crate::data::db::connect_readonly;
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::ConfigureLibraryRequest;
use crate::output;

/// Configure library settings
pub async fn execute(
    tag: &str,
    mode: Option<LibraryMode>,
    _patterns: Option<String>,
    enable: bool,
    disable: bool,
    show: bool,
) -> Result<()> {
    output::section(format!("Library Configuration: {}", tag));

    // Show current configuration (reads directly — still allowed)
    if show || (mode.is_none() && !enable && !disable) {
        if let Ok(conn) = connect_readonly() {
            show_current_config(&conn, tag, &format!("lib-{}", tag))?;
            if mode.is_some() || enable || disable {
                output::separator();
            }
        }
    }

    // Apply changes via daemon if any mutation requested
    if mode.is_some() || enable || disable {
        let mut client = ensure_daemon_available().await?;

        let response = client
            .library_write()
            .configure_library(ConfigureLibraryRequest {
                tag: tag.to_string(),
                mode: mode.map(|m| m.to_string()),
                enable: if enable { Some(true) } else { None },
                disable: if disable { Some(true) } else { None },
            })
            .await?
            .into_inner();

        if response.affected_count > 0 {
            output::success("Configuration updated");
        }
    }

    Ok(())
}

/// Display the current library configuration (read-only)
fn show_current_config(conn: &rusqlite::Connection, tag: &str, watch_id: &str) -> Result<()> {
    output::info("Current configuration:");
    output::separator();

    let result: Result<(String, Option<String>, i32, bool), _> = conn.query_row(
        "SELECT path, library_mode, enabled, follow_symlinks \
         FROM watch_folders WHERE watch_id = ?",
        [watch_id],
        |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get::<_, i32>(3)? != 0,
            ))
        },
    );

    match result {
        Ok((path, lib_mode, enabled, follow_symlinks)) => {
            output::kv("Tag", tag);
            output::kv("Watch ID", watch_id);
            output::kv("Path", &path);
            output::kv("Mode", lib_mode.as_deref().unwrap_or("incremental"));
            output::kv("Enabled", if enabled == 1 { "yes" } else { "no" });
            output::kv(
                "Follow Symlinks",
                if follow_symlinks { "yes" } else { "no" },
            );
        }
        Err(e) => {
            output::error(format!("Failed to read configuration: {}", e));
        }
    }

    Ok(())
}
