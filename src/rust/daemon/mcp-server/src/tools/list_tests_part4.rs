//! List tool hermetic tests, part 4: stats/components/basePath/submodule/branch (§16–§23).
//!
//! Included from `list_tests.rs` via `#[cfg(test)] #[path = "list_tests_part4.rs"] mod part4;`.

use crate::tools::list::ListInput;

use super::{call_list, session_with_branch, session_with_project, TestDb};

// ---------------------------------------------------------------------------
// § 16  Stats: languages sorted, folders counted
// ---------------------------------------------------------------------------

#[test]
fn stats_languages_sorted() {
    let db = TestDb::new();
    db.insert_project("wid-lang2", "t-lang2", "/p");
    db.insert_file(
        "f1",
        "wid-lang2",
        "a.rs",
        Some("rust"),
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f2",
        "wid-lang2",
        "b.py",
        Some("python"),
        Some("py"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f3",
        "wid-lang2",
        "c.ts",
        Some("typescript"),
        Some("ts"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-lang2");
    let v = call_list(&db, ListInput::default(), &session);
    let langs = v["stats"]["languages"].as_array().unwrap();
    let lang_strs: Vec<&str> = langs.iter().map(|l| l.as_str().unwrap()).collect();
    let mut sorted = lang_strs.clone();
    sorted.sort();
    assert_eq!(lang_strs, sorted, "languages must be sorted alphabetically");
}

#[test]
fn stats_folders_counted() {
    let db = TestDb::new();
    db.insert_project("wid-fld2", "t-fld2", "/p");
    db.insert_file(
        "f1",
        "wid-fld2",
        "src/a.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f2",
        "wid-fld2",
        "src/sub/b.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f3",
        "wid-fld2",
        "docs/c.md",
        None,
        Some("md"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-fld2");
    let v = call_list(&db, ListInput::default(), &session);
    // Folders: src/, src/sub/, docs/ = 3
    assert_eq!(v["stats"]["folders"], 3);
}

// ---------------------------------------------------------------------------
// § 17  Component detection in stats
// ---------------------------------------------------------------------------

#[test]
fn stats_components_present_when_db_has_components() {
    let db = TestDb::new();
    db.insert_project("wid-cstat", "t-cstat", "/p");
    db.insert_component("c1", "wid-cstat", "daemon", "src/rust", "cargo");
    db.insert_component("c2", "wid-cstat", "cli", "src/cli", "cargo");
    db.insert_file(
        "f1",
        "wid-cstat",
        "src/rust/a.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-cstat");
    let v = call_list(&db, ListInput::default(), &session);

    let comps = v["stats"]["components"]
        .as_array()
        .expect("components should be present");
    assert_eq!(comps.len(), 2);
    let ids: Vec<&str> = comps.iter().map(|c| c["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"daemon"));
    assert!(ids.contains(&"cli"));
    assert_eq!(comps[0]["basePath"].as_str().unwrap(), "src/cli"); // sorted by component_name
    assert_eq!(comps[0]["source"].as_str().unwrap(), "cargo");
}

#[test]
fn stats_components_absent_when_no_components() {
    let db = TestDb::new();
    db.insert_project("wid-nocmp", "t-nocmp", "/p");
    db.insert_file("f1", "wid-nocmp", "a.rs", None, Some("rs"), 0, "[]", None);

    let session = session_with_project("t-nocmp");
    let v = call_list(&db, ListInput::default(), &session);

    let has_comps = v["stats"]
        .get("components")
        .map(|c| !c.is_null())
        .unwrap_or(false);
    assert!(
        !has_comps,
        "components should be absent when none registered"
    );
}

// ---------------------------------------------------------------------------
// § 18  basePath filter (path input)
// ---------------------------------------------------------------------------

#[test]
fn base_path_filters_to_subtree() {
    let db = TestDb::new();
    db.insert_project("wid-bp", "t-bp", "/p");
    db.insert_file("f1", "wid-bp", "src/a.rs", None, Some("rs"), 0, "[]", None);
    db.insert_file("f2", "wid-bp", "docs/b.md", None, Some("md"), 0, "[]", None);

    let session = session_with_project("t-bp");
    let v = call_list(
        &db,
        ListInput {
            path: Some("src".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["basePath"], "src");
}

// ---------------------------------------------------------------------------
// § 19  projectId override in input
// ---------------------------------------------------------------------------

#[test]
fn project_id_input_override_takes_precedence() {
    let db = TestDb::new();
    db.insert_project("wid-ov", "tenant-override", "/proj-override");
    db.insert_file("f1", "wid-ov", "a.rs", None, Some("rs"), 0, "[]", None);

    // Session has a different (missing) project_id; input specifies the real one.
    let session = session_with_project("wrong-tenant");
    let v = call_list(
        &db,
        ListInput {
            project_id: Some("tenant-override".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );
    assert_eq!(v["success"], true);
    assert_eq!(v["stats"]["files"], 1);
}

// ---------------------------------------------------------------------------
// § 20  Submodule marker in tree rendering
// ---------------------------------------------------------------------------

#[test]
fn submodule_appears_in_tree_listing() {
    let db = TestDb::new();
    db.insert_project("wid-sm", "t-sm", "/p");
    db.insert_submodule(
        "wid-sm-sub",
        "t-sm-sub",
        "wid-sm",
        "vendor/lib",
        Some("https://github.com/user/lib.git"),
    );
    db.insert_file("f1", "wid-sm", "src/a.rs", None, Some("rs"), 0, "[]", None);
    // A tracked file under the submodule path is required to trigger submodule
    // detection in build_tree (both TS and Rust insert folders only when a file
    // traverses the submodule path).
    db.insert_file(
        "f2",
        "wid-sm",
        "vendor/lib/lib.c",
        None,
        Some("c"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-sm");
    let v = call_list(
        &db,
        ListInput {
            format: Some("tree".to_string()),
            ..Default::default()
        },
        &session,
    );

    let listing = v["listing"].as_str().unwrap();
    assert!(
        listing.contains("[submodule: lib]"),
        "expected '[submodule: lib]' in listing: {listing:?}"
    );
}

// ---------------------------------------------------------------------------
// § 21  totalMatching reflects all matching rows (independent of page/limit)
// ---------------------------------------------------------------------------

#[test]
fn total_matching_reflects_all_rows() {
    let db = TestDb::new();
    db.insert_project("wid-tot", "t-tot", "/p");
    for i in 0..10u32 {
        db.insert_file(
            &format!("ft{i}"),
            "wid-tot",
            &format!("{i:02}.rs"),
            None,
            Some("rs"),
            0,
            "[]",
            None,
        );
    }

    let session = session_with_project("t-tot");
    // Fetch with limit=3; totalMatching should still be 10.
    let v = call_list(
        &db,
        ListInput {
            page_size: Some(3),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );
    assert_eq!(v["stats"]["totalMatching"], 10);
}

// ---------------------------------------------------------------------------
// § 22  basePath defaults to "." when empty
// ---------------------------------------------------------------------------

#[test]
fn base_path_defaults_to_dot() {
    let db = TestDb::new();
    db.insert_project("wid-dot", "t-dot", "/p");

    let session = session_with_project("t-dot");
    let v = call_list(&db, ListInput::default(), &session);
    assert_eq!(v["basePath"], ".");
}

// ---------------------------------------------------------------------------
// § 23  Branch filter from session
// ---------------------------------------------------------------------------

#[test]
fn branch_filter_from_session() {
    let db = TestDb::new();
    db.insert_project("wid-br", "t-br", "/p");
    db.insert_file(
        "f1",
        "wid-br",
        "a.rs",
        None,
        Some("rs"),
        0,
        r#"["main"]"#,
        None,
    );
    db.insert_file(
        "f2",
        "wid-br",
        "b.rs",
        None,
        Some("rs"),
        0,
        r#"["feat"]"#,
        None,
    );

    let session = session_with_branch("t-br", "main");
    let v = call_list(
        &db,
        ListInput {
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["listing"].as_str().unwrap(), "a.rs");
}

#[test]
fn branch_star_includes_all_branches() {
    let db = TestDb::new();
    db.insert_project("wid-star", "t-star", "/p");
    db.insert_file(
        "f1",
        "wid-star",
        "a.rs",
        None,
        Some("rs"),
        0,
        r#"["main"]"#,
        None,
    );
    db.insert_file(
        "f2",
        "wid-star",
        "b.rs",
        None,
        Some("rs"),
        0,
        r#"["feat"]"#,
        None,
    );

    let session = session_with_project("t-star");
    // branch="*" disables branch filtering
    let v = call_list(
        &db,
        ListInput {
            branch: Some("*".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );
    assert_eq!(v["stats"]["files"], 2);
}
