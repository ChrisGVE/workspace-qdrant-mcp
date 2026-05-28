//! Two-column per-entity queue breakdown renderer.

use colored::Colorize;

use crate::grpc::client::workspace_daemon::SystemStatusResponse;
use crate::output::gutter::Gutter;
use crate::output::number::NumberLocale;
use crate::output::table::terminal_width;

use super::format::{format_decomp_line, format_entity_header, pad_to};

/// Render the per-entity queue breakdown in a two-column layout.
///
/// Per the hybrid template spec (cli-feedback.md): homogeneous list split
/// across columns is allowed, separators span full terminal width, and
/// the section uses the entire terminal width.
pub(super) fn render_entity_two_column(
    entities: &[(String, usize, usize, usize)],
    locale: &NumberLocale,
    daemon_status: &Option<SystemStatusResponse>,
) {
    let term_w = terminal_width();
    let (key_col_w, num_w) = compute_layout_metrics(entities, locale);

    let col_inner = key_col_w + 1 + num_w; // key + space + number

    // Minimum width for two columns: both column contents + 4 char gap
    let min_two_col = 2 * (Gutter::WIDTH + col_inner) + 4;
    let use_two_col = term_w >= min_two_col;

    // Each column occupies half the terminal.
    let half_w = term_w / 2;

    // Section header — no opening separator needed because the main
    // columnar block's closing separator serves as the section break.
    println!("  {}", "Queue By Entity".bold());

    let chunk_size = if use_two_col { 2 } else { 1 };

    for pair in entities.chunks(chunk_size) {
        render_entity_pair(
            pair,
            use_two_col,
            half_w,
            key_col_w,
            num_w,
            col_inner,
            locale,
        );
    }

    render_resource_mode_section(daemon_status, term_w);

    // Closing separator (full terminal width)
    println!("{}", "─".repeat(term_w));
}

/// Compute column layout metrics from the entity data.
///
/// Returns `(key_col_w, num_w)` where `key_col_w` is the width of the key
/// column (max of entity name width and decomp label width) and `num_w` is
/// the width needed to right-align all numbers.
fn compute_layout_metrics(
    entities: &[(String, usize, usize, usize)],
    locale: &NumberLocale,
) -> (usize, usize) {
    use crate::output::number::format_usize;

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

    // Take the wider of decomp and header key widths.
    let key_col_w = name_w.max(decomp_key_w);

    (key_col_w, num_w)
}

/// Render one pair (or single) entity with header, underline, and decomp rows.
fn render_entity_pair(
    pair: &[(String, usize, usize, usize)],
    use_two_col: bool,
    half_w: usize,
    key_col_w: usize,
    num_w: usize,
    col_inner: usize,
    locale: &NumberLocale,
) {
    let left = &pair[0];
    let right = if use_two_col { pair.get(1) } else { None };

    println!();

    render_header_row(left, right, half_w, key_col_w, num_w, locale);
    render_underline_row(right, half_w, col_inner);
    render_decomp_rows(left, right, half_w, key_col_w, num_w, locale);
}

/// Render the entity name + total header line.
fn render_header_row(
    left: &(String, usize, usize, usize),
    right: Option<&(String, usize, usize, usize)>,
    half_w: usize,
    key_col_w: usize,
    num_w: usize,
    locale: &NumberLocale,
) {
    let left_hdr =
        format_entity_header(&left.0, left.1 + left.2 + left.3, key_col_w, num_w, locale);
    if let Some(r) = right {
        let left_hdr_padded = pad_to(&format!("  {left_hdr}"), half_w);
        let right_hdr = format_entity_header(&r.0, r.1 + r.2 + r.3, key_col_w, num_w, locale);
        println!("{left_hdr_padded}  {right_hdr}");
    } else {
        println!("  {left_hdr}");
    }
}

/// Render the underline separator below the header.
fn render_underline_row(
    right: Option<&(String, usize, usize, usize)>,
    half_w: usize,
    col_inner: usize,
) {
    let ul = "─".repeat(col_inner);
    if right.is_some() {
        let left_ul = pad_to(&format!("  {ul}"), half_w);
        println!(
            "{}{}",
            format!("{left_ul}").dimmed(),
            format!("  {ul}").dimmed(),
        );
    } else {
        println!("{}", format!("  {ul}").dimmed());
    }
}

/// Render the three decomposition rows (Pending, In Progress, Failed).
fn render_decomp_rows(
    left: &(String, usize, usize, usize),
    right: Option<&(String, usize, usize, usize)>,
    half_w: usize,
    key_col_w: usize,
    num_w: usize,
    locale: &NumberLocale,
) {
    let decomp_labels = [
        ("Pending", Gutter::Add),
        ("In Progress", Gutter::Update),
        ("Failed", Gutter::Remove),
    ];
    let left_nums = [left.1, left.2, left.3];

    for (i, (label, gutter)) in decomp_labels.iter().enumerate() {
        let left_line = format_decomp_line(*gutter, label, left_nums[i], key_col_w, num_w, locale);
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

/// Render the resource mode section within the two-column block (verbose only).
fn render_resource_mode_section(daemon_status: &Option<SystemStatusResponse>, term_w: usize) {
    let Some(ref status) = daemon_status else {
        return;
    };
    if status.resource_mode.is_none() {
        return;
    }

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
}
