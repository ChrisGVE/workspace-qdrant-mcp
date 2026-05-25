//! Tests for branch discovery module.

use std::collections::HashMap;

use super::db::KnownFile;
use super::scanner::{classify_files, infer_parent_branch};

fn make_known(file_id: i64, branches: &[&str], base_point: Option<&str>) -> KnownFile {
    KnownFile {
        file_id,
        branches: branches.iter().map(|s| s.to_string()).collect(),
        base_point: base_point.map(|s| s.to_string()),
    }
}

#[test]
fn test_classify_all_shared() {
    let mut fs_files = HashMap::new();
    fs_files.insert("src/a.rs".to_string(), "hash_a".to_string());
    fs_files.insert("src/b.rs".to_string(), "hash_b".to_string());

    let mut known = HashMap::new();
    known.insert(
        ("src/a.rs".to_string(), "hash_a".to_string()),
        make_known(1, &["main"], Some("bp1")),
    );
    known.insert(
        ("src/b.rs".to_string(), "hash_b".to_string()),
        make_known(2, &["main"], Some("bp2")),
    );

    let (shared, novel) = classify_files(&fs_files, &known, "feature");
    assert_eq!(shared.len(), 2);
    assert!(novel.is_empty());
}

#[test]
fn test_classify_all_novel() {
    let mut fs_files = HashMap::new();
    fs_files.insert("src/new.rs".to_string(), "hash_new".to_string());

    let known = HashMap::new();

    let (shared, novel) = classify_files(&fs_files, &known, "feature");
    assert!(shared.is_empty());
    assert_eq!(novel.len(), 1);
    assert_eq!(novel[0], "src/new.rs");
}

#[test]
fn test_classify_mixed() {
    let mut fs_files = HashMap::new();
    fs_files.insert("src/a.rs".to_string(), "hash_a".to_string());
    fs_files.insert("src/b.rs".to_string(), "hash_b_changed".to_string());
    fs_files.insert("src/c.rs".to_string(), "hash_c".to_string());

    let mut known = HashMap::new();
    known.insert(
        ("src/a.rs".to_string(), "hash_a".to_string()),
        make_known(1, &["main"], Some("bp1")),
    );
    // b.rs has different hash — no match
    known.insert(
        ("src/b.rs".to_string(), "hash_b_original".to_string()),
        make_known(2, &["main"], Some("bp2")),
    );

    let (shared, novel) = classify_files(&fs_files, &known, "feature");
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].file_id, 1);
    // b.rs changed hash and c.rs is entirely new
    assert_eq!(novel.len(), 2);
    assert!(novel.contains(&"src/b.rs".to_string()));
    assert!(novel.contains(&"src/c.rs".to_string()));
}

#[test]
fn test_classify_skips_already_present_branch() {
    let mut fs_files = HashMap::new();
    fs_files.insert("src/a.rs".to_string(), "hash_a".to_string());

    let mut known = HashMap::new();
    known.insert(
        ("src/a.rs".to_string(), "hash_a".to_string()),
        make_known(1, &["main", "feature"], Some("bp1")),
    );

    // "feature" already in branches — should not appear in shared
    let (shared, novel) = classify_files(&fs_files, &known, "feature");
    assert!(shared.is_empty());
    assert!(novel.is_empty());
}

#[test]
fn test_infer_parent_most_matches() {
    let mut fs_files = HashMap::new();
    fs_files.insert("src/a.rs".to_string(), "h1".to_string());
    fs_files.insert("src/b.rs".to_string(), "h2".to_string());
    fs_files.insert("src/c.rs".to_string(), "h3".to_string());

    let mut known = HashMap::new();
    // main has all 3 files
    known.insert(
        ("src/a.rs".to_string(), "h1".to_string()),
        make_known(1, &["main", "dev"], None),
    );
    known.insert(
        ("src/b.rs".to_string(), "h2".to_string()),
        make_known(2, &["main"], None),
    );
    known.insert(
        ("src/c.rs".to_string(), "h3".to_string()),
        make_known(3, &["main"], None),
    );

    // main appears 3 times, dev appears 1 time → parent = main
    let parent = infer_parent_branch(&fs_files, &known);
    assert_eq!(parent, Some("main".to_string()));
}

#[test]
fn test_infer_parent_no_matches() {
    let mut fs_files = HashMap::new();
    fs_files.insert("src/new.rs".to_string(), "new_hash".to_string());

    let known = HashMap::new();

    let parent = infer_parent_branch(&fs_files, &known);
    assert_eq!(parent, None);
}

#[test]
fn test_discovery_result_default() {
    let result = super::scanner::BranchDiscoveryResult::default();
    assert_eq!(result.shared_count, 0);
    assert!(result.novel_paths.is_empty());
    assert!(result.parent_branch.is_none());
    assert_eq!(result.errors, 0);
}
