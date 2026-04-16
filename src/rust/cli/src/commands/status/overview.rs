//! Default status overview subcommand.
//!
//! Columnar template per cli-feedback.md. Uses SQLite for metrics,
//! gRPC only for daemon health and resource mode.

use anyhow::Result;
use colored::Colorize;

use crate::commands::watch::helpers::{build_full_tenant_name_map, prefixed_display_name};
use crate::data::db::connect_readonly;
use crate::data::health;
use crate::data::queries::{self, HealthLevel};
use crate::grpc::client::{workspace_daemon::SystemStatusResponse, DaemonClient};
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::table::terminal_width;
use crate::output::{self, ServiceStatus};

use super::types::{status_label, SystemStatusJson};

/// Show the default system status overview.
pub async fn default_status(
    show_queue: bool,
    show_watch: bool,
    show_performance: bool,
    verbose: bool,
    json: bool,
) -> Result<()> {
    // Check daemon connectivity (gRPC) — measure response time
    let grpc_start = std::time::Instant::now();
    let (daemon_up, daemon_status) = match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let status = client
                .system()
                .get_status(())
                .await
                .ok()
                .map(|r| r.into_inner());
            (true, status)
        }
        Err(_) => (false, None),
    };
    let grpc_ms = grpc_start.elapsed().as_millis() as u64;

    // Check Qdrant connectivity — measure response time
    let qdrant_start = std::time::Instant::now();
    let qdrant_health = health::check_qdrant().await;
    let qdrant_ms = qdrant_start.elapsed().as_millis() as u64;

    // Get metrics from SQLite
    let (collection_count, document_count, active_project_count, project_names, queue_stats) =
        match connect_readonly() {
            Ok(conn) => {
                let collections = queries::get_active_collection_count(&conn).unwrap_or(0);
                let documents = queries::get_total_document_count(&conn, "projects").unwrap_or(0);
                let active = queries::get_active_project_count(&conn).unwrap_or(0);
                let projects = queries::get_projects(&conn).unwrap_or_default();
                let names: Vec<String> = projects
                    .iter()
                    .filter(|p| p.is_active)
                    .map(|p| {
                        p.path
                            .rsplit('/')
                            .find(|s| !s.is_empty())
                            .unwrap_or(&p.tenant_id)
                            .to_string()
                    })
                    .collect();
                let q = queries::get_queue_stats(&conn).unwrap_or_default();
                (collections, documents, active, names, q)
            }
            Err(_) => (0, 0, 0, Vec::new(), queries::QueueStats::default()),
        };

    let total_collections = if qdrant_health.reachable {
        qdrant_health.collection_count.max(collection_count)
    } else {
        collection_count
    };

    // Compute health levels
    let worker_health = if daemon_up {
        HealthLevel::Healthy
    } else {
        HealthLevel::Unhealthy
    };
    let qdrant_level = if qdrant_health.reachable {
        HealthLevel::Healthy
    } else {
        HealthLevel::Unhealthy
    };
    let queue_health = queue_stats.health();
    let overall = worker_health.worst(qdrant_level).worst(queue_health);

    // JSON mode
    if json {
        let json_out = SystemStatusJson {
            connected: daemon_up,
            status: if daemon_up {
                daemon_status
                    .as_ref()
                    .map(|s| status_label(ServiceStatus::from_proto(s.status)).to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            } else {
                "unhealthy".to_string()
            },
            collections: total_collections as i32,
            documents: document_count as i32,
            active_projects: project_names,
            pending_operations: Some(queue_stats.pending as i32),
            resource_mode: daemon_status.as_ref().and_then(|s| s.resource_mode.clone()),
            idle_seconds: daemon_status.as_ref().and_then(|s| s.idle_seconds),
            current_max_embeddings: daemon_status
                .as_ref()
                .and_then(|s| s.current_max_embeddings),
            current_inter_item_delay_ms: daemon_status
                .as_ref()
                .and_then(|s| s.current_inter_item_delay_ms),
        };
        output::print_json(&json_out);
        return Ok(());
    }

    canvas::print_title("System Status");
    canvas::print_blank();

    let locale = NumberLocale::default();

    // Build columnar display
    let total_queue = queue_stats.pending + queue_stats.in_progress + queue_stats.failed;

    let mut builder = ColumnarBuilder::new();

    // In verbose mode, use full terminal width for hybrid layout compliance.
    if verbose {
        builder = builder.full_width();
    }

    // Service lines with response times as right-aligned annotations
    let queue_reason = queue_stats.health_reason();

    builder = builder
        .kv("Overall", format_health(overall))
        .section(Some("Services"))
        .kv_annotated(
            "Workspace-Qdrant Worker",
            format_health(worker_health),
            format!("{grpc_ms}ms"),
        )
        .kv_annotated(
            "Workspace-Qdrant Queue",
            format_health(queue_health),
            queue_reason.as_deref().unwrap_or(""),
        )
        .kv_annotated(
            "Qdrant Server",
            format_health(qdrant_level),
            format!("{qdrant_ms}ms"),
        )
        .section(Some("Metrics"))
        .kv("Collections", format_usize(total_collections, &locale))
        .kv("Documents", format_usize(document_count, &locale))
        .kv(
            "Active Projects",
            format_usize(active_project_count, &locale),
        );

    // List active project names
    if !project_names.is_empty() {
        builder = builder.section(Some("Active Projects"));
        for name in &project_names {
            builder = builder.raw(name, Gutter::Sync);
        }
    }

    // Queue section with decomposition + avg processing time + ETA as annotations
    let avg_ms = connect_readonly()
        .ok()
        .and_then(|conn| queries::get_avg_processing_ms(&conn));

    builder = builder.section(Some("Queue"));

    // Total line, with avg processing time as annotation
    if let Some(avg) = avg_ms {
        builder = builder.kv_underline_annotated(
            "Total",
            format_usize(total_queue, &locale),
            format!("avg {}/item", format_duration_short(avg as u64)),
        );
    } else {
        builder = builder.kv_underline("Total", format_usize(total_queue, &locale));
    }

    {
        // ETA as annotation on Pending line
        let eta_annotation = avg_ms.map(|avg| {
            let eta_ms = queue_stats.pending as f64 * avg;
            format!("est. {}", format_duration_short(eta_ms as u64))
        });

        let decomp: Vec<(&str, String, Gutter, Option<String>)> = vec![
            (
                "Pending",
                format_usize(queue_stats.pending, &locale),
                Gutter::Add,
                eta_annotation,
            ),
            (
                "In Progress",
                format_usize(queue_stats.in_progress, &locale),
                Gutter::Update,
                None,
            ),
            (
                "Failed",
                format_usize(queue_stats.failed, &locale),
                Gutter::Remove,
                None,
            ),
        ];
        let inner = ColumnarBuilder::new().aligned_group_annotated(decomp);
        builder = builder.nested("", inner);
    }

    // Resource mode from daemon (only via gRPC) — still part of main columnar
    if let Some(ref status) = daemon_status {
        if !verbose && status.resource_mode.is_some() {
            builder = add_resource_mode(builder, status);
        }
    }

    builder.render();

    // Verbose: per-entity queue breakdown as two-column layout
    // Rendered AFTER the main columnar block per hybrid template rules:
    // the single-columnar section uses its own separator width, then the
    // two-column section uses full terminal width.
    let verbose_entity_data = if verbose {
        connect_readonly().ok().and_then(|conn| {
            let tenant_names = build_full_tenant_name_map(&conn);
            let per_entity = get_per_entity_queue(&conn, &tenant_names);
            if per_entity.is_empty() {
                None
            } else {
                Some(per_entity)
            }
        })
    } else {
        None
    };

    if let Some(per_entity) = verbose_entity_data {
        render_entity_two_column(&per_entity, &locale, &daemon_status);
    }

    // Warnings after columnar block
    if !daemon_up {
        output::warning("Worker not running. Start with: wqm service start");
    }
    if !qdrant_health.reachable {
        if let Some(ref err) = qdrant_health.error {
            output::warning(err);
        }
    }

    // Optional additional sections
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

/// Format a health level with color (no gutter — inline colored text).
fn format_health(level: HealthLevel) -> String {
    match level {
        HealthLevel::Healthy => "healthy".green().to_string(),
        HealthLevel::Degraded => "degraded".yellow().to_string(),
        HealthLevel::Unhealthy => "unhealthy".red().to_string(),
    }
}

/// Get per-entity queue breakdown with collection-aware display names.
///
/// Returns `(display_name, pending, in_progress, failed)` tuples sorted
/// alphabetically by display name. Display names use collection prefixes
/// (`prj:`, `lib:`, `rls:`, `scp:`) when the queue contains items from
/// multiple collection types.
fn get_per_entity_queue(
    conn: &rusqlite::Connection,
    tenant_names: &std::collections::HashMap<String, String>,
) -> Vec<(String, usize, usize, usize)> {
    let mut result: std::collections::HashMap<(String, String), (usize, usize, usize)> =
        std::collections::HashMap::new();

    let Ok(mut stmt) = conn.prepare(
        "SELECT collection, tenant_id, status, COUNT(*) FROM unified_queue \
         WHERE status IN ('pending', 'in_progress', 'failed') \
         GROUP BY collection, tenant_id, status",
    ) else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, usize>(3)?,
        ))
    }) else {
        return Vec::new();
    };

    // Track which collections appear to decide if prefixes are needed
    let mut collections_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for row in rows.flatten() {
        let (collection, tenant_id, status, count) = row;
        collections_seen.insert(collection.clone());
        let entry = result.entry((collection, tenant_id)).or_insert((0, 0, 0));
        match status.as_str() {
            "pending" => entry.0 += count,
            "in_progress" => entry.1 += count,
            "failed" => entry.2 += count,
            _ => {}
        }
    }

    let use_prefixes = collections_seen.len() > 1;

    let mut sorted: Vec<(String, usize, usize, usize)> = result
        .into_iter()
        .map(|((collection, tenant_id), (p, i, f))| {
            let display = if use_prefixes {
                prefixed_display_name(&collection, &tenant_id, tenant_names)
            } else {
                tenant_names.get(&tenant_id).cloned().unwrap_or(tenant_id)
            };
            (display, p, i, f)
        })
        .collect();
    sorted.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
    sorted
}

/// Right-pad a formatted number to a target width.
fn pad_number(formatted: &str, target_width: usize) -> String {
    let width = formatted.chars().count();
    let padding = target_width.saturating_sub(width);
    format!("{}{}", " ".repeat(padding), formatted)
}

/// Format milliseconds into a compact human-readable duration.
/// Examples: "1.2s", "45s", "12m", "3h 25m", "2d 5h"
fn format_duration_short(ms: u64) -> String {
    let secs = ms / 1000;
    if secs < 60 {
        if ms < 10_000 {
            format!("{:.1}s", ms as f64 / 1000.0)
        } else {
            format!("{secs}s")
        }
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        if m > 0 {
            format!("{h}h {m}m")
        } else {
            format!("{h}h")
        }
    } else {
        let d = secs / 86400;
        let h = (secs % 86400) / 3600;
        if h > 0 {
            format!("{d}d {h}h")
        } else {
            format!("{d}d")
        }
    }
}

/// Pad a string (which may contain ANSI codes) to a target visible width.
fn pad_to(s: &str, target: usize) -> String {
    let visible = output::strip_ansi(s).chars().count();
    let padding = target.saturating_sub(visible);
    format!("{s}{}", " ".repeat(padding))
}

/// Render the per-entity queue breakdown in a two-column layout.
///
/// Per the hybrid template spec (cli-feedback.md): homogeneous list split
/// across columns is allowed, separators span full terminal width, and
/// the section uses the entire terminal width.
fn render_entity_two_column(
    entities: &[(String, usize, usize, usize)],
    locale: &NumberLocale,
    daemon_status: &Option<SystemStatusResponse>,
) {
    let term_w = terminal_width();

    // Compute metrics for consistent alignment across both columns.
    // "In Progress" is the widest decomposition key at 11 chars + ":"
    let decomp_key_w = 12; // "In Progress:" = 12 chars

    let num_w = entities
        .iter()
        .flat_map(|(_, p, i, f)| [*p, *i, *f, p + i + f])
        .map(|n| format_usize(n, locale).chars().count())
        .max()
        .unwrap_or(1);

    let name_w = entities
        .iter()
        .map(|(name, ..)| name.chars().count() + 1) // +1 for colon
        .max()
        .unwrap_or(1);

    // Each column needs: gutter(2) + key_w + 1(space) + num_w
    // For the header line: gutter(2) + name_w + 1(space) + num_w
    // Take the wider of decomp and header key widths.
    let key_col_w = name_w.max(decomp_key_w);
    let col_inner = key_col_w + 1 + num_w; // key + space + number

    // Minimum width for two columns: both column contents + 4 char gap
    let min_two_col = 2 * (Gutter::WIDTH + col_inner) + 4;
    let use_two_col = term_w >= min_two_col;

    // Each column occupies half the terminal. Content is left-aligned
    // within its half-column, producing an even spread per the spec:
    // "Columns should be spread evenly across the available screen."
    let half_w = term_w / 2;

    // Section header — no opening separator needed because the main
    // columnar block's closing separator serves as the section break.
    println!("  {}", "Queue By Entity".bold());

    // Chunk size: 2 for two-column, 1 for narrow fallback
    let chunk_size = if use_two_col { 2 } else { 1 };

    for pair in entities.chunks(chunk_size) {
        let left = &pair[0];
        let right = if use_two_col { pair.get(1) } else { None };

        // Blank line between groups
        println!();

        // Header line: entity name + total
        let left_hdr =
            format_entity_header(&left.0, left.1 + left.2 + left.3, key_col_w, num_w, locale);
        if let Some(r) = right {
            let left_hdr_padded = pad_to(&format!("  {left_hdr}"), half_w);
            let right_hdr = format_entity_header(&r.0, r.1 + r.2 + r.3, key_col_w, num_w, locale);
            println!("{left_hdr_padded}  {right_hdr}");
        } else {
            println!("  {left_hdr}");
        }

        // Underline under each header
        let ul = "─".repeat(col_inner);
        if let Some(_) = right {
            let left_ul = pad_to(&format!("  {ul}"), half_w);
            println!(
                "{}{}",
                format!("{left_ul}").dimmed(),
                format!("  {ul}").dimmed(),
            );
        } else {
            println!("{}", format!("  {ul}").dimmed());
        }

        // Decomposition rows: Pending, In Progress, Failed
        let decomp_labels = [
            ("Pending", Gutter::Add),
            ("In Progress", Gutter::Update),
            ("Failed", Gutter::Remove),
        ];
        let left_nums = [left.1, left.2, left.3];

        for (i, (label, gutter)) in decomp_labels.iter().enumerate() {
            let left_line =
                format_decomp_line(*gutter, label, left_nums[i], key_col_w, num_w, locale);
            if let Some(r) = right {
                let right_nums = [r.1, r.2, r.3];
                let right_line =
                    format_decomp_line(*gutter, label, right_nums[i], key_col_w, num_w, locale);
                let left_padded = pad_to(&left_line, half_w);
                println!("{left_padded}{right_line}");
            } else {
                println!("{left_line}");
            }
        }
    }

    // Resource mode section (if available) — rendered within the same
    // full-width block per hybrid template rules.
    if let Some(ref status) = daemon_status {
        if status.resource_mode.is_some() {
            println!();
            println!("{}", "─".repeat(term_w).dimmed());
            // "Inter-Item Delay:" = 17 chars — widest key with colon
            const RM_KEY_W: usize = 17;
            println!("  {}", "Resource Mode".bold());
            if let Some(ref mode) = status.resource_mode {
                let k = "Mode:";
                println!(
                    "      {}{} {mode}",
                    k.bold(),
                    " ".repeat(RM_KEY_W - k.len())
                );
            }
            if let Some(idle) = status.idle_seconds {
                let k = "Idle Time:";
                println!(
                    "      {}{} {}",
                    k.bold(),
                    " ".repeat(RM_KEY_W - k.len()),
                    wqm_common::duration_fmt::format_duration(idle, 0)
                );
            }
            if let Some(max_emb) = status.current_max_embeddings {
                let k = "Max Embeddings:";
                println!(
                    "      {}{} {max_emb}",
                    k.bold(),
                    " ".repeat(RM_KEY_W - k.len())
                );
            }
            if let Some(delay) = status.current_inter_item_delay_ms {
                let k = "Inter-Item Delay:";
                println!(
                    "      {}{} {delay}ms",
                    k.bold(),
                    " ".repeat(RM_KEY_W - k.len())
                );
            }
        }
    }

    // Closing separator (full terminal width)
    println!("{}", "─".repeat(term_w));
}

/// Format an entity header line: "name:" right-padded, then right-aligned total.
fn format_entity_header(
    name: &str,
    total: usize,
    key_w: usize,
    num_w: usize,
    locale: &NumberLocale,
) -> String {
    let label = format!("{name}:");
    let label_w = label.chars().count();
    let key_pad = key_w.saturating_sub(label_w);
    let total_str = format_usize(total, locale);
    let num_pad = num_w.saturating_sub(total_str.chars().count());
    format!(
        "{}{}{}{}{}",
        label.bold(),
        " ".repeat(key_pad),
        " ", // separator between key and value
        " ".repeat(num_pad),
        total_str,
    )
}

/// Format a decomposition line with gutter, label, and right-aligned number.
fn format_decomp_line(
    gutter: Gutter,
    label: &str,
    value: usize,
    key_w: usize,
    num_w: usize,
    locale: &NumberLocale,
) -> String {
    let key_str = format!("{label}:");
    let key_pad = key_w.saturating_sub(key_str.chars().count());
    let val_str = format_usize(value, locale);
    let num_pad = num_w.saturating_sub(val_str.chars().count());
    format!(
        "{} {}{} {}{}",
        gutter.colored(),
        key_str,
        " ".repeat(key_pad),
        " ".repeat(num_pad),
        val_str,
    )
}

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
    if let Some(delay) = status.current_inter_item_delay_ms {
        builder = builder.kv("Inter-Item Delay", format!("{}ms", delay));
    }
    builder
}
