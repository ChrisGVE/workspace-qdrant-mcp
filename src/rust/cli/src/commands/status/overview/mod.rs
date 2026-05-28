//! Default status overview subcommand.
//!
//! Columnar template per cli-feedback.md. Uses SQLite for metrics,
//! gRPC only for daemon health and resource mode.

mod data;
mod entity_query;
mod format;
mod render_entity;

use anyhow::Result;

use crate::grpc::client::workspace_daemon::SystemStatusResponse;
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::{self, ServiceStatus};

use super::types::{status_label, SystemStatusJson};
use data::StatusData;
use format::{format_duration_short, format_health};

/// Show the default system status overview.
pub async fn default_status(
    show_queue: bool,
    show_watch: bool,
    show_performance: bool,
    verbose: bool,
    json: bool,
) -> Result<()> {
    let d = data::collect().await;

    if json {
        return render_json(&d);
    }

    canvas::print_title("System Status");
    canvas::print_blank();

    let locale = NumberLocale::default();
    let mut builder = build_columnar(&d, verbose, &locale);

    if !verbose {
        if let Some(ref status) = d.daemon_status {
            if status.resource_mode.is_some() {
                builder = add_resource_mode(builder, status);
            }
        }
    }

    builder.render();

    if verbose {
        if let Some(per_entity) = entity_query::load_entity_queue_data() {
            render_entity::render_entity_two_column(&per_entity, &locale, &d.daemon_status);
        }
    }

    emit_warnings(&d);

    if show_queue {
        println!();
        super::queue::queue(false).await?;
    }
    if show_watch {
        println!();
        super::watch::watch().await?;
    }
    if show_performance {
        println!();
        super::performance::performance().await?;
    }

    Ok(())
}

/// Render JSON output and return.
fn render_json(d: &StatusData) -> Result<()> {
    let json_out = SystemStatusJson {
        connected: d.daemon_up,
        status: if d.daemon_up {
            d.daemon_status
                .as_ref()
                .map(|s| status_label(ServiceStatus::from_proto(s.status)).to_string())
                .unwrap_or_else(|| "unknown".to_string())
        } else {
            "unhealthy".to_string()
        },
        collections: d.collection_count as i32,
        documents: d.document_count as i32,
        active_projects: d.project_names.clone(),
        pending_operations: Some(d.queue_stats.pending as i32),
        resource_mode: d
            .daemon_status
            .as_ref()
            .and_then(|s| s.resource_mode.clone()),
        idle_seconds: d.daemon_status.as_ref().and_then(|s| s.idle_seconds),
        current_max_embeddings: d
            .daemon_status
            .as_ref()
            .and_then(|s| s.current_max_embeddings),
    };
    output::print_json(&json_out);
    Ok(())
}

/// Build the main columnar display block.
fn build_columnar(d: &StatusData, verbose: bool, locale: &NumberLocale) -> ColumnarBuilder {
    let total_queue = d.queue_stats.pending + d.queue_stats.in_progress + d.queue_stats.failed;
    let queue_reason = d.queue_stats.health_reason();

    let mut builder = ColumnarBuilder::new();
    if verbose {
        builder = builder.full_width();
    }

    builder = builder
        .kv("Overall", format_health(d.overall))
        .section(Some("Services"))
        .kv_annotated(
            "Workspace-Qdrant Worker",
            format_health(d.worker_health),
            format!("{}ms", d.grpc_ms),
        )
        .kv_annotated(
            "Workspace-Qdrant Queue",
            format_health(d.queue_health),
            queue_reason.as_deref().unwrap_or(""),
        )
        .kv_annotated(
            "Qdrant Server",
            format_health(d.qdrant_level),
            format!("{}ms", d.qdrant_ms),
        )
        .section(Some("Metrics"))
        .kv("Collections", format_usize(d.collection_count, locale))
        .kv("Documents", format_usize(d.document_count, locale))
        .kv(
            "Active Projects",
            format_usize(d.active_project_count, locale),
        );

    if !d.project_names.is_empty() {
        builder = builder.section(Some("Active Projects"));
        for name in &d.project_names {
            builder = builder.raw(name, Gutter::Sync);
        }
    }

    builder = add_queue_section(builder, d, total_queue, locale);

    builder
}

/// Append the queue section (total + decomposition) to the builder.
fn add_queue_section(
    mut builder: ColumnarBuilder,
    d: &StatusData,
    total_queue: usize,
    locale: &NumberLocale,
) -> ColumnarBuilder {
    builder = builder.section(Some("Queue"));

    if let Some(avg) = d.avg_processing_ms {
        builder = builder.kv_underline_annotated(
            "Total",
            format_usize(total_queue, locale),
            format!("avg {}/item", format_duration_short(avg as u64)),
        );
    } else {
        builder = builder.kv_underline("Total", format_usize(total_queue, locale));
    }

    let eta_annotation = d.avg_processing_ms.map(|avg| {
        let eta_ms = d.queue_stats.pending as f64 * avg;
        format!("est. {}", format_duration_short(eta_ms as u64))
    });

    let decomp: Vec<(&str, String, Gutter, Option<String>)> = vec![
        (
            "Pending",
            format_usize(d.queue_stats.pending, locale),
            Gutter::Add,
            eta_annotation,
        ),
        (
            "In Progress",
            format_usize(d.queue_stats.in_progress, locale),
            Gutter::Update,
            None,
        ),
        (
            "Failed",
            format_usize(d.queue_stats.failed, locale),
            Gutter::Remove,
            None,
        ),
    ];
    let inner = ColumnarBuilder::new().aligned_group_annotated(decomp);
    builder = builder.nested("", inner);

    builder
}

/// Emit warning lines after the columnar block.
fn emit_warnings(d: &StatusData) {
    if !d.daemon_up {
        output::warning("Worker not running. Start with: wqm service start");
    }
    if !d.qdrant_reachable {
        if let Some(ref err) = d.qdrant_error {
            output::warning(err);
        }
    }
}

/// Add resource mode section to builder (non-verbose path only).
fn add_resource_mode(
    mut builder: ColumnarBuilder,
    status: &SystemStatusResponse,
) -> ColumnarBuilder {
    builder = builder.section(Some("Resource Mode"));

    if let Some(ref mode) = status.resource_mode {
        builder = builder.kv("Mode", mode);
    }
    if let Some(idle) = status.idle_seconds {
        builder = builder.kv(
            "Idle Time",
            wqm_common::duration_fmt::format_duration(idle, 0),
        );
    }
    if let Some(max_emb) = status.current_max_embeddings {
        builder = builder.kv("Max Embeddings", max_emb.to_string());
    }
    builder
}
