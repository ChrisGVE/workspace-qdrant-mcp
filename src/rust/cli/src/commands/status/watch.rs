//! Watch status subcommand.
//!
//! Columnar template per cli-feedback.md.

use anyhow::Result;

use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;

/// Show file watcher status.
pub async fn watch() -> Result<()> {
    let active_projects = fetch_active_projects().await;

    canvas::print_title("Watch Status");
    canvas::print_blank();

    let builder = match active_projects {
        Some(ref projects) if !projects.is_empty() => {
            let mut b = ColumnarBuilder::new().kv("Active Projects", projects.len().to_string());
            let mut nested = ColumnarBuilder::new();
            for project in projects {
                nested = nested.kv("", project);
            }
            b = b.nested("", nested);
            b
        }
        Some(_) => ColumnarBuilder::new().kv("Active Projects", "none"),
        None => ColumnarBuilder::new().kv("Daemon", "not reachable"),
    };

    builder.render();

    Ok(())
}

/// Fetch active project names from daemon, returning None if unreachable.
async fn fetch_active_projects() -> Option<Vec<String>> {
    let mut client = crate::grpc::connect_default().await.ok()?;
    let response = client.system().get_status(()).await.ok()?;
    Some(response.into_inner().active_projects)
}
