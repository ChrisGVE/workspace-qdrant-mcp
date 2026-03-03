use super::super::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_detect_cargo_workspace() {
    let dir = TempDir::new().unwrap();
    let cargo_toml = dir.path().join("Cargo.toml");
    fs::write(
        &cargo_toml,
        r#"
[workspace]
resolver = "2"
members = ["crate-a", "crate-b"]
"#,
    )
    .unwrap();

    // Create the member directories (not strictly needed for detection,
    // but validates the path computation)
    fs::create_dir_all(dir.path().join("crate-a")).unwrap();
    fs::create_dir_all(dir.path().join("crate-b")).unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);
    assert!(components.contains_key("crate-a"));
    assert!(components.contains_key("crate-b"));
    assert_eq!(components["crate-a"].source, ComponentSource::Cargo);
}

#[test]
fn test_detect_cargo_workspace_nested() {
    let dir = TempDir::new().unwrap();
    // No root Cargo.toml, but one in src/rust/
    let nested = dir.path().join("src/rust");
    fs::create_dir_all(&nested).unwrap();
    fs::write(
        nested.join("Cargo.toml"),
        r#"
[workspace]
members = ["daemon/core", "cli"]
"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);

    // Component IDs are based on member path, not full path
    assert!(components.contains_key("daemon.core"));
    assert!(components.contains_key("cli"));

    // base_path should include the relative prefix
    assert_eq!(components["daemon.core"].base_path, "src/rust/daemon/core");
    assert_eq!(components["cli"].base_path, "src/rust/cli");
}

#[test]
fn test_detect_npm_workspace() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("package.json"),
        r#"{"workspaces": ["packages/ui", "packages/api"]}"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);
    assert!(components.contains_key("packages.ui"));
    assert!(components.contains_key("packages.api"));
    assert_eq!(components["packages.ui"].source, ComponentSource::Npm);
}

#[test]
fn test_detect_npm_workspace_glob() {
    let dir = TempDir::new().unwrap();
    let pkgs = dir.path().join("packages");
    fs::create_dir_all(pkgs.join("alpha")).unwrap();
    fs::create_dir_all(pkgs.join("beta")).unwrap();
    // Create a file that should be ignored (not a dir)
    fs::write(pkgs.join("README.md"), "").unwrap();

    fs::write(
        dir.path().join("package.json"),
        r#"{"workspaces": ["packages/*"]}"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);
    assert!(components.contains_key("packages.alpha"));
    assert!(components.contains_key("packages.beta"));
}

#[test]
fn test_detect_directory_fallback() {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join("src")).unwrap();
    fs::create_dir_all(dir.path().join("tests")).unwrap();
    fs::create_dir_all(dir.path().join(".git")).unwrap();
    fs::create_dir_all(dir.path().join("node_modules")).unwrap();
    fs::write(dir.path().join("README.md"), "").unwrap();

    let components = detect_components(dir.path());
    assert!(components.contains_key("src"));
    assert!(components.contains_key("tests"));
    assert!(!components.contains_key(".git"));
    assert!(!components.contains_key("node_modules"));
    assert_eq!(components["src"].source, ComponentSource::Directory);
}

#[test]
fn test_cargo_takes_priority_over_npm() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"
[workspace]
members = ["shared"]
"#,
    )
    .unwrap();
    fs::write(
        dir.path().join("package.json"),
        r#"{"workspaces": ["shared", "web"]}"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    // "shared" should be Cargo (takes priority)
    assert_eq!(components["shared"].source, ComponentSource::Cargo);
    // "web" should be npm (no conflict)
    assert!(components.contains_key("web"));
    assert_eq!(components["web"].source, ComponentSource::Npm);
}
