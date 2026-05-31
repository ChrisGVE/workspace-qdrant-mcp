//! Directory tree construction from flat file path lists.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/list-files/tree-builder.ts`.

use std::collections::BTreeMap;

use crate::sqlite::tracked_files::{SubmoduleEntry, TrackedFileEntry};

use super::types::{FileLeaf, FolderNode, SubmoduleMarker};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a submodule lookup map keyed by path.
///
/// Mirrors `buildSubmoduleMap` in tree-builder.ts line 12.
fn build_submodule_map(
    submodules: &[SubmoduleEntry],
) -> std::collections::HashMap<String, &SubmoduleEntry> {
    let mut map = std::collections::HashMap::new();
    for sm in submodules {
        map.insert(sm.submodule_path.clone(), sm);
    }
    map
}

/// Insert a single file entry into the folder tree.
///
/// Mirrors `insertFile` in tree-builder.ts lines 21-59.
fn insert_file(
    root: &mut FolderNode,
    file: &TrackedFileEntry,
    base_path: &str,
    submodule_map: &std::collections::HashMap<String, &SubmoduleEntry>,
) {
    let mut rel_path = file.relative_path.as_str();
    if !base_path.is_empty() {
        let prefix = format!("{base_path}/");
        if let Some(stripped) = rel_path.strip_prefix(prefix.as_str()) {
            rel_path = stripped;
        }
    }

    let segments: Vec<&str> = rel_path.split('/').collect();
    let (dir_segments, file_name) = match segments.split_last() {
        Some((last, rest)) => (rest, *last),
        None => return,
    };

    let mut current = root;
    let mut path_so_far = base_path.to_string();

    for segment in dir_segments {
        if path_so_far.is_empty() {
            path_so_far = segment.to_string();
        } else {
            path_so_far = format!("{path_so_far}/{segment}");
        }

        if !current.children.contains_key(*segment) {
            let mut node = FolderNode {
                name: segment.to_string(),
                children: BTreeMap::new(),
                files: Vec::new(),
                submodule: None,
                total_files: 0,
            };
            if let Some(sm) = submodule_map.get(&path_so_far) {
                node.submodule = Some(SubmoduleMarker {
                    repo_name: sm.repo_name.clone(),
                });
            }
            current.children.insert(segment.to_string(), node);
        }

        current = current.children.get_mut(*segment).unwrap();
        // Stop descending into submodule nodes (mirrors TS line 48).
        if current.submodule.is_some() {
            return;
        }
    }

    if current.submodule.is_none() {
        current.files.push(FileLeaf {
            name: file_name.to_string(),
            extension: file.extension.clone(),
            language: file.language.clone(),
            is_test: file.is_test,
        });
    }
}

/// Recursively compute `total_files` for every node.
///
/// Mirrors `computeTotalFiles` in tree-builder.ts lines 82-89.
pub fn compute_total_files(node: &mut FolderNode) -> usize {
    let mut total = node.files.len();
    for child in node.children.values_mut() {
        total += compute_total_files(child);
    }
    node.total_files = total;
    total
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a folder tree from a flat list of tracked files.
///
/// Mirrors `buildTree` in tree-builder.ts lines 66-80.
pub fn build_tree(
    files: &[TrackedFileEntry],
    submodules: &[SubmoduleEntry],
    base_path: &str,
) -> FolderNode {
    let mut root = FolderNode {
        name: if base_path.is_empty() {
            ".".to_string()
        } else {
            base_path.to_string()
        },
        children: BTreeMap::new(),
        files: Vec::new(),
        submodule: None,
        total_files: 0,
    };

    let submodule_map = build_submodule_map(submodules);
    for file in files {
        insert_file(&mut root, file, base_path, &submodule_map);
    }
    compute_total_files(&mut root);
    root
}

/// Count all folder nodes in a tree (recursive).
///
/// Mirrors `countFolders` in filters.ts lines 12-17.
pub fn count_folders(node: &FolderNode) -> usize {
    let mut count = 0;
    for child in node.children.values() {
        count += 1 + count_folders(child);
    }
    count
}
