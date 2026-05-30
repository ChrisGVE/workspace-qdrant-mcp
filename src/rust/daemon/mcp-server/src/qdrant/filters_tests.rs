//! Unit tests for `filters.rs` — hermetic, no live Qdrant required.
//!
//! Every test asserts on the *shape* of the built `Filter` struct:
//! `must` length, `must_not` length, `should` nesting, and specific
//! field names/values where deterministic.

use super::*;
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS};

// ── extract_glob_prefix ───────────────────────────────────────────────────────

#[test]
fn glob_prefix_no_meta_returns_full_string() {
    assert_eq!(extract_glob_prefix("src/main.rs"), "src/main.rs");
}

#[test]
fn glob_prefix_star_star_slash_returns_empty() {
    assert_eq!(extract_glob_prefix("**/*.rs"), "");
}

#[test]
fn glob_prefix_src_star_star_returns_src_slash() {
    assert_eq!(extract_glob_prefix("src/**/*.rs"), "src/");
}

#[test]
fn glob_prefix_question_mark_no_slash_returns_empty() {
    assert_eq!(extract_glob_prefix("foo?.rs"), "");
}

#[test]
fn glob_prefix_question_mark_with_slash() {
    assert_eq!(extract_glob_prefix("src/foo?.rs"), "src/");
}

#[test]
fn glob_prefix_empty_string_returns_empty() {
    assert_eq!(extract_glob_prefix(""), "");
}

#[test]
fn glob_prefix_brace_meta() {
    assert_eq!(extract_glob_prefix("src/{a,b}.rs"), "src/");
}

#[test]
fn glob_prefix_bracket_meta() {
    assert_eq!(extract_glob_prefix("src/[abc].rs"), "src/");
}

// ── build_filter — no-op cases ────────────────────────────────────────────────

#[test]
fn no_conditions_returns_none() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "all".to_string(),
        ..Default::default()
    };
    assert!(build_filter(&params).is_none());
}

// ── project scope ─────────────────────────────────────────────────────────────

#[test]
fn project_scope_with_project_id_adds_tenant_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("proj-abc".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).expect("filter must be Some");
    assert_eq!(f.must.len(), 1, "exactly one must condition for tenant_id");
    assert_eq!(f.must_not.len(), 0);
}

#[test]
fn project_scope_without_project_id_returns_none() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: None,
        ..Default::default()
    };
    assert!(build_filter(&params).is_none());
}

// ── group scope ───────────────────────────────────────────────────────────────

#[test]
fn group_scope_with_ids_adds_match_any_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "group".to_string(),
        group_tenant_ids: Some(vec!["t1".to_string(), "t2".to_string()]),
        ..Default::default()
    };
    let f = build_filter(&params).expect("filter must be Some");
    assert_eq!(f.must.len(), 1);
    assert_eq!(f.must_not.len(), 0);
}

#[test]
fn group_scope_empty_ids_returns_none_gracefully() {
    // PANIC FIX: previously panicked with "Group scope requires non-empty tenant ID set".
    // Now returns None gracefully instead of aborting — a panic is never acceptable at runtime.
    // Full group-scope resolution via resolveSearchScope is DEFERRED (task 30 follow-up).
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "group".to_string(),
        group_tenant_ids: Some(vec![]),
        ..Default::default()
    };
    // Must NOT panic; returns None (no tenant filter added).
    assert!(build_filter(&params).is_none());
}

#[test]
fn group_scope_no_ids_returns_none_gracefully() {
    // PANIC FIX: group scope with group_tenant_ids=None also returns None gracefully.
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "group".to_string(),
        group_tenant_ids: None,
        ..Default::default()
    };
    assert!(build_filter(&params).is_none());
}

// ── branch filter ─────────────────────────────────────────────────────────────

#[test]
fn branch_wildcard_is_skipped() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("proj".to_string()),
        branch: Some("*".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).expect("filter for tenant only");
    assert_eq!(
        f.must.len(),
        1,
        "branch=* must not add a condition; only tenant_id"
    );
}

#[test]
fn branch_specific_adds_one_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("proj".to_string()),
        branch: Some("main".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).expect("filter");
    assert_eq!(f.must.len(), 2, "tenant_id + branches condition");
}

// ── libraries collection ──────────────────────────────────────────────────────

#[test]
fn libraries_collection_adds_deleted_must_not() {
    let params = FilterParams {
        collection: COLLECTION_LIBRARIES.to_string(),
        scope: "all".to_string(),
        ..Default::default()
    };
    let f = build_filter(&params).expect("must_not forces Some");
    assert_eq!(f.must_not.len(), 1, "deleted=true must_not");
}

#[test]
fn projects_collection_no_deleted_must_not() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    assert_eq!(f.must_not.len(), 0);
}

#[test]
fn library_name_condition_only_in_libraries() {
    let params = FilterParams {
        collection: COLLECTION_LIBRARIES.to_string(),
        scope: "all".to_string(),
        library_name: Some("tokio".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    // must has 1 (library_name), must_not has 1 (deleted)
    assert_eq!(f.must.len(), 1);
    assert_eq!(f.must_not.len(), 1);
}

#[test]
fn library_name_condition_ignored_in_projects() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        library_name: Some("tokio".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    assert_eq!(f.must.len(), 1, "only tenant_id; library_name skipped");
}

#[test]
fn library_path_condition_only_in_libraries() {
    let params = FilterParams {
        collection: COLLECTION_LIBRARIES.to_string(),
        scope: "all".to_string(),
        library_path: Some("std/".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    assert_eq!(f.must.len(), 1);
}

// ── file_type ─────────────────────────────────────────────────────────────────

#[test]
fn file_type_adds_one_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        file_type: Some("rs".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    // tenant_id + file_type = 2
    assert_eq!(f.must.len(), 2);
}

// ── tag / tags ────────────────────────────────────────────────────────────────

#[test]
fn single_tag_adds_should_wrapped_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        tag: Some("rust".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    // tenant_id (1) + tag should-group (1) = 2
    assert_eq!(f.must.len(), 2);
}

#[test]
fn multiple_tags_adds_one_should_wrapped_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        tags: Some(vec!["rust".to_string(), "async".to_string()]),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    // tenant_id (1) + tags should-group (1) = 2
    assert_eq!(f.must.len(), 2);
}

#[test]
fn empty_tags_vec_does_not_add_condition() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        tags: Some(vec![]),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    assert_eq!(f.must.len(), 1, "empty tags vec must not add condition");
}

// ── component ─────────────────────────────────────────────────────────────────

#[test]
fn component_adds_should_wrapped_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        component: Some("daemon".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    // tenant_id (1) + component should-group (1) = 2
    assert_eq!(f.must.len(), 2);
}

// ── path_glob ─────────────────────────────────────────────────────────────────

#[test]
fn path_glob_with_prefix_adds_file_path_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        path_glob: Some("src/**/*.rs".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    // tenant_id (1) + file_path prefix (1) = 2
    assert_eq!(f.must.len(), 2);
}

#[test]
fn path_glob_no_prefix_does_not_add_condition() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        path_glob: Some("**/*.rs".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    assert_eq!(f.must.len(), 1, "glob with no prefix must be skipped");
}

// ── determine_collections ─────────────────────────────────────────────────────

#[test]
fn determine_collections_explicit_overrides_scope() {
    let cols = determine_collections(Some("custom"), "project", true);
    assert_eq!(cols, vec!["custom"]);
}

#[test]
fn determine_collections_project_scope_no_libs() {
    let cols = determine_collections(None, "project", false);
    assert_eq!(cols, vec![COLLECTION_PROJECTS]);
}

#[test]
fn determine_collections_project_scope_with_libs() {
    let cols = determine_collections(None, "project", true);
    assert_eq!(cols, vec![COLLECTION_PROJECTS, COLLECTION_LIBRARIES]);
}

#[test]
fn determine_collections_group_scope_with_libs() {
    let cols = determine_collections(None, "group", true);
    assert_eq!(cols, vec![COLLECTION_PROJECTS, COLLECTION_LIBRARIES]);
}

#[test]
fn determine_collections_all_scope() {
    let cols = determine_collections(None, "all", false);
    assert_eq!(
        cols,
        vec![
            COLLECTION_PROJECTS,
            COLLECTION_LIBRARIES,
            COLLECTION_SCRATCHPAD
        ]
    );
}

#[test]
fn determine_collections_unknown_scope_defaults_to_projects() {
    let cols = determine_collections(None, "unknown", false);
    assert_eq!(cols, vec![COLLECTION_PROJECTS]);
}

// ── base_points ───────────────────────────────────────────────────────────────

#[test]
fn base_points_adds_one_must() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        base_points: Some(vec!["bp1".to_string(), "bp2".to_string()]),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    // tenant_id (1) + base_point (1) = 2
    assert_eq!(f.must.len(), 2);
}

#[test]
fn empty_base_points_does_not_add_condition() {
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("p".to_string()),
        base_points: Some(vec![]),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    assert_eq!(f.must.len(), 1);
}

// ── combined conditions ───────────────────────────────────────────────────────

#[test]
fn all_conditions_combined_correct_count() {
    // project_id + branch + file_type + tag + component + path_glob = 6 must
    let params = FilterParams {
        collection: COLLECTION_PROJECTS.to_string(),
        scope: "project".to_string(),
        project_id: Some("proj".to_string()),
        branch: Some("main".to_string()),
        file_type: Some("rs".to_string()),
        tag: Some("async".to_string()),
        component: Some("daemon".to_string()),
        path_glob: Some("src/**/*.rs".to_string()),
        ..Default::default()
    };
    let f = build_filter(&params).unwrap();
    assert_eq!(
        f.must.len(),
        6,
        "tenant_id + branches + file_type + tag-group + component-group + file_path"
    );
    assert_eq!(f.must_not.len(), 0);
}
