use super::super::*;

#[test]
fn test_path_to_component_id() {
    assert_eq!(path_to_component_id("daemon/core"), "daemon.core");
    assert_eq!(path_to_component_id("cli"), "cli");
    assert_eq!(path_to_component_id("src/typescript/mcp"), "src.typescript.mcp");
    assert_eq!(path_to_component_id("trailing/"), "trailing");
    assert_eq!(path_to_component_id("/leading"), "leading");
}

#[test]
fn test_parse_cargo_members_basic() {
    let content = r#"
[workspace]
resolver = "2"
members = [
    "daemon/core",
    "daemon/grpc",
    "cli",
]
"#;
    let members = parse_cargo_members(content);
    assert_eq!(members, vec!["daemon/core", "daemon/grpc", "cli"]);
}

#[test]
fn test_parse_cargo_members_inline() {
    let content = r#"
[workspace]
members = ["a", "b"]
"#;
    let members = parse_cargo_members(content);
    assert_eq!(members, vec!["a", "b"]);
}

#[test]
fn test_parse_cargo_members_with_comments() {
    let content = r#"
[workspace]
members = [
    "a",
    # "commented-out",
    "b",
]
"#;
    let members = parse_cargo_members(content);
    assert_eq!(members, vec!["a", "b"]);
}

#[test]
fn test_parse_cargo_members_no_workspace() {
    let content = r#"
[package]
name = "my-crate"
"#;
    assert!(parse_cargo_members(content).is_empty());
}

#[test]
fn test_file_matches_component() {
    let comp = ComponentInfo {
        id: "daemon.core".into(),
        base_path: "daemon/core".into(),
        patterns: vec!["daemon/core/**".into()],
        source: ComponentSource::Cargo,
    };

    assert!(file_matches_component("daemon/core/src/lib.rs", &comp));
    assert!(file_matches_component("daemon/core", &comp));
    assert!(!file_matches_component("daemon/grpc/src/lib.rs", &comp));
    assert!(!file_matches_component("daemon/core_extra/foo.rs", &comp));
}

#[test]
fn test_component_matches_filter() {
    assert!(component_matches_filter("daemon.core", "daemon.core"));
    assert!(component_matches_filter("daemon.core", "daemon"));
    assert!(!component_matches_filter("daemon.core", "cli"));
    assert!(!component_matches_filter("daemon", "daemon.core"));
}

#[test]
fn test_assign_component_most_specific() {
    let mut components = ComponentMap::new();
    components.insert(
        "daemon".into(),
        ComponentInfo {
            id: "daemon".into(),
            base_path: "daemon".into(),
            patterns: vec!["daemon/**".into()],
            source: ComponentSource::Cargo,
        },
    );
    components.insert(
        "daemon.core".into(),
        ComponentInfo {
            id: "daemon.core".into(),
            base_path: "daemon/core".into(),
            patterns: vec!["daemon/core/**".into()],
            source: ComponentSource::Cargo,
        },
    );

    let result = assign_component("daemon/core/src/lib.rs", &components);
    assert_eq!(result.unwrap().id, "daemon.core");

    let result = assign_component("daemon/grpc/src/lib.rs", &components);
    assert_eq!(result.unwrap().id, "daemon");

    let result = assign_component("cli/src/main.rs", &components);
    assert!(result.is_none());
}
