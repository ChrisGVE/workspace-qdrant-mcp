//! Graph stats subcommand — node/edge counts
//!
//! Columnar template per cli-feedback.md.

use anyhow::{Context, Result};

use crate::grpc::client::workspace_daemon::GraphStatsRequest;
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};

pub async fn graph_stats(tenant_id: Option<String>) -> Result<()> {
    let mut client = crate::grpc::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .get_graph_stats(GraphStatsRequest {
            tenant_id: tenant_id.clone(),
            branch: None,
        })
        .await
        .context("GetGraphStats RPC failed")?
        .into_inner();

    canvas::print_title("Graph Statistics");
    canvas::print_blank();

    let locale = NumberLocale::default();

    let mut builder = ColumnarBuilder::new();

    if let Some(ref t) = tenant_id {
        builder = builder.kv("Tenant", t);
    } else {
        builder = builder.kv("Scope", "all tenants");
    }

    builder = builder
        .kv(
            "Total Nodes",
            format_usize(resp.total_nodes as usize, &locale),
        )
        .kv(
            "Total Edges",
            format_usize(resp.total_edges as usize, &locale),
        );

    if !resp.nodes_by_type.is_empty() {
        builder = builder.section(Some("Nodes by Type"));
        let mut sorted: Vec<_> = resp.nodes_by_type.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        let entries: Vec<(&str, String, Gutter)> = sorted
            .iter()
            .map(|(t, count)| {
                (
                    t.as_str(),
                    format_usize(**count as usize, &locale),
                    Gutter::None,
                )
            })
            .collect();
        builder = builder.aligned_group(entries);
    }

    if !resp.edges_by_type.is_empty() {
        builder = builder.section(Some("Edges by Type"));
        let mut sorted: Vec<_> = resp.edges_by_type.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        let entries: Vec<(&str, String, Gutter)> = sorted
            .iter()
            .map(|(t, count)| {
                (
                    t.as_str(),
                    format_usize(**count as usize, &locale),
                    Gutter::None,
                )
            })
            .collect();
        builder = builder.aligned_group(entries);
    }

    builder.render();

    Ok(())
}
