//! List tool hermetic tests, part 2: filters §8–§11 (fileType, includeTests, pattern, component).
//!
//! Included from `list_tests.rs` via `#[cfg(test)] #[path = "list_tests_part2.rs"] mod part2;`.

use crate::tools::list::ListInput;

use super::{call_list, session_with_project, TestDb};

// ---------------------------------------------------------------------------
// § 8  Filter: fileType
// ---------------------------------------------------------------------------

#[test]
fn filter_file_type() {
    let db = TestDb::new();
    db.insert_project("wid-ft", "t-ft", "/p");
    db.insert_file(
        "f1",
        "wid-ft",
        "a.rs",
        None,
        Some("rs"),
        0,
        "[]",
        Some("code"),
    );
    db.insert_file(
        "f2",
        "wid-ft",
        "b.md",
        None,
        Some("md"),
        0,
        "[]",
        Some("docs"),
    );

    let session = session_with_project("t-ft");
    let v = call_list(
        &db,
        ListInput {
            file_type: Some("docs".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["listing"].as_str().unwrap(), "b.md");
}

// ---------------------------------------------------------------------------
// § 9  Filter: includeTests=false
// ---------------------------------------------------------------------------

#[test]
fn filter_include_tests_false() {
    let db = TestDb::new();
    db.insert_project("wid-it", "t-it", "/p");
    db.insert_file(
        "f1",
        "wid-it",
        "src/lib.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f2",
        "wid-it",
        "src/lib_test.rs",
        None,
        Some("rs"),
        1,
        "[]",
        None,
    );

    let session = session_with_project("t-it");
    let v = call_list(
        &db,
        ListInput {
            include_tests: Some(false),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["listing"].as_str().unwrap(), "src/lib.rs");
}

#[test]
fn filter_include_tests_true_includes_both() {
    let db = TestDb::new();
    db.insert_project("wid-it2", "t-it2", "/p");
    db.insert_file("f1", "wid-it2", "a.rs", None, Some("rs"), 0, "[]", None);
    db.insert_file(
        "f2",
        "wid-it2",
        "a_test.rs",
        None,
        Some("rs"),
        1,
        "[]",
        None,
    );

    let session = session_with_project("t-it2");
    let v = call_list(
        &db,
        ListInput {
            include_tests: Some(true),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 2);
}

// ---------------------------------------------------------------------------
// § 10  Filter: pattern (glob)
// ---------------------------------------------------------------------------

#[test]
fn filter_pattern_glob() {
    let db = TestDb::new();
    db.insert_project("wid-pat", "t-pat", "/p");
    db.insert_file("f1", "wid-pat", "src/a.rs", None, Some("rs"), 0, "[]", None);
    db.insert_file(
        "f2",
        "wid-pat",
        "src/b.toml",
        None,
        Some("toml"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f3",
        "wid-pat",
        "docs/c.md",
        None,
        Some("md"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-pat");
    let v = call_list(
        &db,
        ListInput {
            pattern: Some("src/*.rs".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["listing"].as_str().unwrap(), "src/a.rs");
}

// ---------------------------------------------------------------------------
// § 11  Filter: component
// ---------------------------------------------------------------------------

#[test]
fn filter_component() {
    let db = TestDb::new();
    db.insert_project("wid-comp", "t-comp", "/p");
    db.insert_component("c1", "wid-comp", "daemon", "src/rust", "cargo");
    db.insert_file(
        "f1",
        "wid-comp",
        "src/rust/a.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f2",
        "wid-comp",
        "src/ts/b.ts",
        None,
        Some("ts"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-comp");
    let v = call_list(
        &db,
        ListInput {
            component: Some("daemon".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["listing"].as_str().unwrap(), "src/rust/a.rs");
}

#[test]
fn filter_component_prefix_matches_sub_components() {
    let db = TestDb::new();
    db.insert_project("wid-cpfx", "t-cpfx", "/p");
    db.insert_component(
        "c1",
        "wid-cpfx",
        "daemon.core",
        "src/rust/daemon/core",
        "cargo",
    );
    db.insert_component(
        "c2",
        "wid-cpfx",
        "daemon.grpc",
        "src/rust/daemon/grpc",
        "cargo",
    );
    db.insert_component("c3", "wid-cpfx", "cli", "src/rust/cli", "cargo");
    db.insert_file(
        "f1",
        "wid-cpfx",
        "src/rust/daemon/core/a.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f2",
        "wid-cpfx",
        "src/rust/daemon/grpc/b.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f3",
        "wid-cpfx",
        "src/rust/cli/c.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-cpfx");
    let v = call_list(
        &db,
        ListInput {
            component: Some("daemon".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    // daemon.core + daemon.grpc match the "daemon" prefix — cli does not.
    assert_eq!(v["stats"]["files"], 2);
}
