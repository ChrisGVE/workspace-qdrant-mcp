//! Rendering functions for the list tool's three output formats.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/list-files/renderers.ts`.
//!
//! Each renderer returns `(listing_string, rendered_count)`.

use crate::sqlite::tracked_files::{SubmoduleEntry, TrackedFileEntry};

use super::tree::build_tree;
use super::types::FolderNode;

// ---------------------------------------------------------------------------
// Ordering helper
// ---------------------------------------------------------------------------

/// Case-insensitive name comparator for children and files.
///
/// Primary key: `name.to_lowercase()` (approximates `localeCompare` for ASCII
/// mixed-case — interleaves uppercase/lowercase like `apple`, `Apple`, `Zebra`).
/// Stable tiebreak: original name bytes (handles identical-case-fold pairs).
///
/// Residual divergence from JS `localeCompare`: non-ASCII accent collation
/// (e.g. `é` vs `e`) is NOT implemented — full ICU ordering is out of scope.
/// The golden-suite task 33 normalises any remaining divergence.
fn name_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.to_lowercase()
        .cmp(&b.to_lowercase())
        .then_with(|| a.cmp(b))
}

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

    // Sort children case-insensitively to approximate `localeCompare` in TS.
    // (Primary: lowercase; stable tiebreak: original bytes.  Non-ASCII accent
    //  order may still diverge from JS — golden-suite task 33 normalises.)
    let mut child_keys: Vec<&String> = node.children.keys().collect();
    child_keys.sort_by(|a, b| name_cmp(a, b));
    for key in &child_keys {
        let child = &node.children[*key];
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

    // Files sorted case-insensitively to approximate `localeCompare` in TS.
    let mut sorted_files = node.files.clone();
    sorted_files.sort_by(|a, b| name_cmp(&a.name, &b.name));
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
    // Sort children case-insensitively to approximate `localeCompare` in TS.
    let mut child_keys: Vec<&String> = node.children.keys().collect();
    child_keys.sort_by(|a, b| name_cmp(a, b));
    for name in &child_keys {
        let child = &node.children[*name];
        if state.count >= limit as usize {
            return false;
        }
        let child_path = if chain_prefix.is_empty() {
            (*name).clone()
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
/// Returns an insertion-ordered list of `(extension, count)` pairs — mirrors
/// the `Map` (insertion-ordered) built by `aggregateExtensions` in
/// renderers.ts lines 105-120.
fn aggregate_extensions(node: &FolderNode) -> Vec<(String, usize)> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    aggregate_extensions_inner(node, &mut counts);
    counts
}

fn aggregate_extensions_inner(node: &FolderNode, counts: &mut Vec<(String, usize)>) {
    for file in &node.files {
        let key = file
            .extension
            .clone()
            .unwrap_or_else(|| "other".to_string());
        if let Some(entry) = counts.iter_mut().find(|(k, _)| *k == key) {
            entry.1 += 1;
        } else {
            counts.push((key, 1));
        }
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
/// Sorts by count descending only (stable) — equal-count extensions keep
/// first-seen (insertion/traversal) order, matching the TS `Map` + stable sort.
fn format_extension_summary(total_files: usize, ext_counts: &[(String, usize)]) -> String {
    if total_files == 0 {
        return "(empty)".to_string();
    }

    // Sort by count descending only; stable sort preserves first-seen order for equal counts.
    let mut sorted: Vec<(&String, usize)> = ext_counts.iter().map(|(k, v)| (k, *v)).collect();
    sorted.sort_by_key(|&(_, n)| std::cmp::Reverse(n));

    let shown = &sorted[..sorted.len().min(4)];
    let mut parts: Vec<String> = shown.iter().map(|(ext, n)| format!("{n} {ext}")).collect();

    if sorted.len() > 4 {
        let shown_total: usize = shown.iter().map(|(_, n)| n).sum();
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

// ---------------------------------------------------------------------------
// Dispatch helper (used by list_tool in mod.rs)
// ---------------------------------------------------------------------------

/// Dispatch to the correct renderer based on the `format` string.
///
/// Returns `(listing_string, rendered_count)`.
pub(super) fn render_files(
    files: &[TrackedFileEntry],
    submodules: &[SubmoduleEntry],
    base_path: &str,
    format: &str,
    depth: u32,
    limit: u32,
) -> (String, usize) {
    let root = build_tree(files, submodules, base_path);
    match format {
        "summary" => render_summary(&root, depth, limit),
        "flat" => render_flat(files, limit),
        _ => render_tree(&root, depth, limit),
    }
}
