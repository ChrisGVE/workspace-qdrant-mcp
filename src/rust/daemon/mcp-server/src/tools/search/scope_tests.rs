//! Tests for scope resolution: decay-map building, relevance decay, base points.

use std::collections::HashMap;

use serde_json::Value;

use super::*;
use crate::proto::{ResolveSearchScopeResponse, TenantDecay};
use crate::qdrant::fusion::{SearchType, TaggedResult};
use crate::tools::search::types::SearchScope;

fn tagged(id: &str, score: f64, tenant: Option<&str>) -> TaggedResult {
    let mut payload: HashMap<String, Value> = HashMap::new();
    if let Some(t) = tenant {
        payload.insert("tenant_id".to_string(), Value::String(t.to_string()));
    }
    TaggedResult {
        id: id.to_string(),
        score,
        collection: "projects".to_string(),
        payload,
        search_type: SearchType::Semantic,
    }
}

// ── scope_filter_from_response ──────────────────────────────────────────────

#[test]
fn scope_filter_group_returns_tenant_ids_and_decay() {
    let resp = ResolveSearchScopeResponse {
        tenant_ids: vec!["a".into(), "b".into()],
        filter_by_tenant: true,
        decay_map: vec![
            TenantDecay {
                tenant_id: "a".into(),
                multiplier: 1.0,
            },
            TenantDecay {
                tenant_id: "b".into(),
                multiplier: 0.7,
            },
        ],
    };
    let (ids, decay) = scope_filter_from_response(&resp);
    assert_eq!(ids, Some(vec!["a".to_string(), "b".to_string()]));
    let decay = decay.unwrap();
    assert_eq!(decay.get("a"), Some(&1.0));
    // 0.7 originates as an f32 proto field widened to f64, so compare with tolerance.
    assert!((decay.get("b").copied().unwrap() - 0.7).abs() < 1e-6);
}

#[test]
fn scope_filter_all_returns_no_tenant_filter_but_decay() {
    // scope=all: filter_by_tenant=false → no group filter, decay still present.
    let resp = ResolveSearchScopeResponse {
        tenant_ids: vec!["a".into()],
        filter_by_tenant: false,
        decay_map: vec![TenantDecay {
            tenant_id: "a".into(),
            multiplier: 1.0,
        }],
    };
    let (ids, decay) = scope_filter_from_response(&resp);
    assert_eq!(ids, None);
    assert!(decay.is_some());
}

#[test]
fn scope_filter_empty_decay_is_none() {
    let resp = ResolveSearchScopeResponse {
        tenant_ids: vec![],
        filter_by_tenant: true,
        decay_map: vec![],
    };
    let (ids, decay) = scope_filter_from_response(&resp);
    assert_eq!(ids, Some(vec![]));
    assert!(decay.is_none());
}

// ── apply_relevance_decay ───────────────────────────────────────────────────

#[test]
fn relevance_decay_multiplies_and_resorts() {
    let mut results = vec![
        tagged("other", 0.9, Some("other_tenant")), // *0.4 = 0.36
        tagged("current", 0.8, Some("cur")),        // *1.0 = 0.80
        tagged("untenanted", 0.5, None),            // unchanged 0.50
    ];
    let mut decay = HashMap::new();
    decay.insert("cur".to_string(), 1.0);
    // "other_tenant" absent → default 0.4.
    apply_relevance_decay(&mut results, &decay);

    // Re-sorted by decayed score desc: current(0.80) > untenanted(0.50) > other(0.36).
    assert_eq!(results[0].id, "current");
    assert!((results[0].score - 0.80).abs() < 1e-9);
    assert_eq!(results[1].id, "untenanted");
    assert!((results[1].score - 0.50).abs() < 1e-9);
    assert_eq!(results[2].id, "other");
    assert!((results[2].score - 0.36).abs() < 1e-9);
}

// ── format_base_points_degraded_reason ──────────────────────────────────────

#[test]
fn degraded_reason_includes_count_and_cap() {
    let r = format_base_points_degraded_reason(Some(742));
    assert!(r.contains("742 active base points"));
    assert!(r.contains("500-filter cap"));
    let r2 = format_base_points_degraded_reason(None);
    assert!(r2.contains("too many active base points"));
}

// ── resolve_base_points ─────────────────────────────────────────────────────

/// Build a temp state.db with a watch folder + the given base points.
fn make_base_points_db(
    dir: &tempfile::TempDir,
    tenant: &str,
    watch_id: &str,
    base_points: &[&str],
) -> std::path::PathBuf {
    let db_path = dir.path().join("state.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE watch_folders (watch_id TEXT, tenant_id TEXT, collection TEXT, parent_watch_id TEXT);
         CREATE TABLE tracked_files (base_point TEXT, watch_folder_id TEXT);",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (watch_id, tenant_id, collection, parent_watch_id) VALUES (?1, ?2, 'projects', NULL)",
        rusqlite::params![watch_id, tenant],
    )
    .unwrap();
    for bp in base_points {
        conn.execute(
            "INSERT INTO tracked_files (base_point, watch_folder_id) VALUES (?1, ?2)",
            rusqlite::params![bp, watch_id],
        )
        .unwrap();
    }
    drop(conn);
    db_path
}

#[test]
fn base_points_under_cap_returned() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = make_base_points_db(&dir, "T", "w1", &["/a", "/b"]);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, count) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/a/x"),
    );
    let mut got = bp.unwrap();
    got.sort();
    assert_eq!(got, vec!["/a".to_string(), "/b".to_string()]);
    assert!(!degraded);
    assert_eq!(count, None);
}

#[test]
fn base_points_non_project_scope_is_none() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = make_base_points_db(&dir, "T", "w1", &["/a"]);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, _, _) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::All,
        std::path::Path::new("/a"),
    );
    assert_eq!(bp, None);
}

#[test]
fn base_points_over_cap_narrows_to_primary() {
    let dir = tempfile::TempDir::new().unwrap();
    // 501 distinct base points; one is a prefix of cwd.
    let owned: Vec<String> = (0..501).map(|i| format!("/bp/{i}")).collect();
    let mut refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    refs.push("/primary/here");
    let db = make_base_points_db(&dir, "T", "w1", &refs);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, _) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/primary/here/sub"),
    );
    assert_eq!(bp, Some(vec!["/primary/here".to_string()]));
    assert!(!degraded);
}

#[test]
fn base_points_over_cap_no_primary_degrades() {
    let dir = tempfile::TempDir::new().unwrap();
    let owned: Vec<String> = (0..520).map(|i| format!("/bp/{i}")).collect();
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    let db = make_base_points_db(&dir, "T", "w1", &refs);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, count) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/unrelated/cwd"),
    );
    assert_eq!(bp, None);
    assert!(degraded);
    assert_eq!(count, Some(520));
}

#[test]
fn base_points_over_cap_sibling_prefix_does_not_false_match() {
    // cwd `/repo-a/...` must NOT match the sibling base point `/repo` (raw
    // string prefix would). With no real prefix match → degrade (Finding #3).
    let dir = tempfile::TempDir::new().unwrap();
    let owned: Vec<String> = (0..520).map(|i| format!("/bp/{i}")).collect();
    let mut refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    refs.push("/repo");
    let db = make_base_points_db(&dir, "T", "w1", &refs);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, _) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/repo-a/src"),
    );
    assert_eq!(bp, None);
    assert!(degraded);
}

// ── cwd_under_base_point ─────────────────────────────────────────────────────

#[test]
fn cwd_under_base_point_segment_boundaries() {
    let sep = std::path::MAIN_SEPARATOR;
    let join = |a: &str, b: &str| format!("{a}{sep}{b}");
    let repo = join("", "repo"); // "/repo" on unix
                                 // Exact match and true descendants match.
    assert!(cwd_under_base_point(&repo, &repo));
    assert!(cwd_under_base_point(&join(&repo, "src"), &repo));
    // Sibling sharing a string prefix must NOT match.
    assert!(!cwd_under_base_point(&format!("{repo}-a"), &repo));
    // Trailing separator on the base point is tolerated.
    assert!(cwd_under_base_point(
        &join(&repo, "src"),
        &format!("{repo}{sep}")
    ));
}
