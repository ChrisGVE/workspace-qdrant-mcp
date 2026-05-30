//! Rendering functions for the list tool's three output formats.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/list-files/renderers.ts`.
//!
//! Each renderer returns `(listing_string, rendered_count)`.

use std::collections::BTreeMap;

use crate::sqlite::tracked_files::TrackedFileEntry;

use super::types::FolderNode;

// ---------------------------------------------------------------------------
// Tree renderer
// ---------------------------------------------------------------------------

struct WalkState {
    lines: Vec<String>,
    count: usize,
}

/// Recursively walk the tree for the `tree` format.
///
/// Mirrors `walkTree` in renderers.ts lines 22-55.
/// Returns `false` when the limit has been hit.
fn walk_tree(
    node: &FolderNode,
    indent: usize,
    current_depth: u32,
    max_depth: u32,
    limit: u32,
    state: &mut WalkState,
) -> bool {
    let prefix = "  ".repeat(indent);

    // BTreeMap already iterates in sorted key order — matches `.sort((a,b) =>
    // a[0].localeCompare(b[0]))` in TS.
    for child in node.children.values() {
        if state.count >= limit as usize {
            return false;
        }
        if let Some(sm) = &child.submodule {
            state.lines.push(format!(
                "{prefix}{}/ [submodule: {}]",
                child.name, sm.repo_name
            ));
            state.count += 1;
            continue;
        }
        if current_depth >= max_depth {
            state.lines.push(format!(
                "{prefix}{}/ ({} files)",
                child.name, child.total_files
            ));
            state.count += 1;
            continue;
        }
        state.lines.push(format!("{prefix}{}/", child.name));
        state.count += 1;
        if !walk_tree(
            child,
            indent + 1,
            current_depth + 1,
            max_depth,
            limit,
            state,
        ) {
            return false;
        }
    }

    // Files sorted by name — mirrors `.sort((a,b) => a.name.localeCompare(b.name))`.
    let mut sorted_files = node.files.clone();
    sorted_files.sort_by(|a, b| a.name.cmp(&b.name));
    for file in &sorted_files {
        if state.count >= limit as usize {
            return false;
        }
        let tag = file
            .extension
            .as_deref()
            .map(|e| format!(" [{e}]"))
            .unwrap_or_default();
        state.lines.push(format!("{prefix}{}{tag}", file.name));
        state.count += 1;
    }
    true
}

/// Render the `tree` format.
///
/// Mirrors `renderTree` in renderers.ts lines 57-61.
pub fn render_tree(root: &FolderNode, max_depth: u32, limit: u32) -> (String, usize) {
    let mut state = WalkState {
        lines: Vec::new(),
        count: 0,
    };
    walk_tree(root, 0, 1, max_depth, limit, &mut state);
    (state.lines.join("\n"), state.count)
}

// ---------------------------------------------------------------------------
// Summary renderer
// ---------------------------------------------------------------------------

/// Recursively walk the tree for the `summary` format.
///
/// Mirrors `walkSummary` in renderers.ts lines 65-97.
fn walk_summary(
    node: &FolderNode,
    indent: usize,
    current_depth: u32,
    chain_prefix: &str,
    max_depth: u32,
    limit: u32,
    state: &mut WalkState,
) -> bool {
    for (name, child) in &node.children {
        if state.count >= limit as usize {
            return false;
        }
        let child_path = if chain_prefix.is_empty() {
            name.clone()
        } else {
            format!("{chain_prefix}{name}")
        };

        if let Some(sm) = &child.submodule {
            let prefix = "  ".repeat(indent);
            state.lines.push(format!(
                "{prefix}{child_path}/ [submodule: {}]",
                sm.repo_name
            ));
            state.count += 1;
            continue;
        }

        // Single-child chain collapsing — mirrors TS lines 82-84.
        if child.children.len() == 1 && child.files.is_empty() && current_depth < max_depth {
            if !walk_summary(
                child,
                indent,
                current_depth + 1,
                &format!("{child_path}/"),
                max_depth,
                limit,
                state,
            ) {
                return false;
            }
            continue;
        }

        let prefix = "  ".repeat(indent);
        let summary = format_extension_summary(child.total_files, &aggregate_extensions(child));
        state.lines.push(format!("{prefix}{child_path}/ {summary}"));
        state.count += 1;

        if current_depth < max_depth {
            if !walk_summary(
                child,
                indent + 1,
                current_depth + 1,
                "",
                max_depth,
                limit,
                state,
            ) {
                return false;
            }
        }
    }
    true
}

/// Render the `summary` format.
///
/// Mirrors `renderSummary` in renderers.ts lines 99-103.
pub fn render_summary(root: &FolderNode, max_depth: u32, limit: u32) -> (String, usize) {
    let mut state = WalkState {
        lines: Vec::new(),
        count: 0,
    };
    walk_summary(root, 0, 1, "", max_depth, limit, &mut state);
    (state.lines.join("\n"), state.count)
}

/// Aggregate extension counts across a subtree (excluding submodule children).
///
/// Mirrors `aggregateExtensions` in renderers.ts lines 105-120.
fn aggregate_extensions(node: &FolderNode) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    aggregate_extensions_inner(node, &mut counts);
    counts
}

fn aggregate_extensions_inner(node: &FolderNode, counts: &mut BTreeMap<String, usize>) {
    for file in &node.files {
        let key = file
            .extension
            .clone()
            .unwrap_or_else(|| "other".to_string());
        *counts.entry(key).or_insert(0) += 1;
    }
    for child in node.children.values() {
        if child.submodule.is_none() {
            aggregate_extensions_inner(child, counts);
        }
    }
}

/// Format the extension summary string.
///
/// Mirrors `formatExtensionSummary` in renderers.ts lines 122-139.
fn format_extension_summary(total_files: usize, ext_counts: &BTreeMap<String, usize>) -> String {
    if total_files == 0 {
        return "(empty)".to_string();
    }

    // Sort by count descending, show top 4.
    let mut sorted: Vec<(&String, &usize)> = ext_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

    let shown = &sorted[..sorted.len().min(4)];
    let mut parts: Vec<String> = shown.iter().map(|(ext, n)| format!("{n} {ext}")).collect();

    if sorted.len() > 4 {
        let shown_total: usize = shown.iter().map(|(_, n)| **n).sum();
        let remaining = total_files.saturating_sub(shown_total);
        if remaining > 0 {
            parts.push(format!("{remaining} other"));
        }
    }

    format!("({total_files} files: {})", parts.join(", "))
}

// ---------------------------------------------------------------------------
// Flat renderer
// ---------------------------------------------------------------------------

/// Render the `flat` format.
///
/// Mirrors `renderFlat` in renderers.ts lines 143-157.
pub fn render_flat(files: &[TrackedFileEntry], limit: u32) -> (String, usize) {
    let mut lines = Vec::new();
    let mut count = 0usize;

    for file in files {
        if count >= limit as usize {
            break;
        }
        lines.push(file.relative_path.clone());
        count += 1;
    }

    (lines.join("\n"), count)
}
