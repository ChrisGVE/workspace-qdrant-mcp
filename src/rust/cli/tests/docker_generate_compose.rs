//! Integration tests for `wqm docker generate-compose`.
//!
//! These tests exercise the built `wqm` binary against synthetic
//! `config.yaml` fixtures and assert on the resulting override file +
//! exit codes. They cover the acceptance criteria for T10 and close out
//! T9 subtasks 9.11 (mandatory port publish) and 9.13 (network-mode
//! rejection).

use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

/// Locate the `wqm` binary built by Cargo for these integration tests.
///
/// Cargo sets `CARGO_BIN_EXE_<name>` for each `[[bin]]` in the crate; we
/// use that instead of probing `target/debug` because the path differs
/// in workspace setups and on the `cargo nextest` runner.
fn wqm_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_wqm"))
}

/// Write `body` to `<dir>/config.yaml` and return its path.
fn write_config(dir: &TempDir, body: &str) -> PathBuf {
    let p = dir.path().join("config.yaml");
    std::fs::write(&p, body).expect("write config.yaml fixture");
    p
}

/// Run `wqm docker generate-compose` with the given args, in `cwd`.
fn run_generate(cwd: &TempDir, args: &[&str]) -> std::process::Output {
    Command::new(wqm_bin())
        .arg("docker")
        .arg("generate-compose")
        .args(args)
        .current_dir(cwd.path())
        .output()
        .expect("spawn wqm")
}

// ────────────────────────────────────────────────────────────────────────
// Default mode: generate
// ────────────────────────────────────────────────────────────────────────

#[test]
fn generate_emits_valid_yaml_with_hash_and_port_publish() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts:
  - host: /Users/username/dev
    container: /Users/username/dev
  - host: /Volumes/External/books
    container: /mnt/books
  - host: ~/reference
    container: /mnt/reference
"#,
    );

    let out = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "generate failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let override_path = tmp.path().join("docker-compose.override.yaml");
    let body = std::fs::read_to_string(&override_path).expect("override file exists");

    // Header present
    assert!(
        body.starts_with("# wqm-config-hash: "),
        "missing hash header. body:\n{}",
        body
    );

    // Mandatory control-port publish — T9 subtask 9.11.
    assert!(
        body.contains("127.0.0.1:7799:7799"),
        "missing mandatory control-port publish. body:\n{}",
        body
    );

    // All three mounts emitted in declaration order.
    assert!(body.contains("/Users/username/dev:/Users/username/dev"));
    assert!(body.contains("/Volumes/External/books:/mnt/books"));
    assert!(body.contains("~/reference:/mnt/reference"));

    // State bind mounts (spec §9.2) unconditional.
    assert!(body.contains("~/.local/share/workspace-qdrant:/var/lib/wqm"));
    assert!(body.contains("~/.local/share/qdrant:/qdrant/storage"));

    // Config file bind mount (read-only).
    let cfg_str = cfg.to_str().unwrap();
    assert!(body.contains(&format!("{}:/etc/wqm/config.yaml:ro", cfg_str)));

    // YAML parseable end-to-end.
    let parsed: serde_yaml_ng::Value =
        serde_yaml_ng::from_str(&body).expect("generated override is valid YAML");
    let volumes = parsed
        .get("services")
        .and_then(|s| s.get("memexd"))
        .and_then(|m| m.get("volumes"))
        .and_then(|v| v.as_sequence())
        .expect("volumes sequence in generated override");
    // 3 mounts + 1 config + 1 override-self-bind + 2 state = 7
    assert_eq!(volumes.len(), 7);
}

#[test]
fn generate_uses_custom_control_port_from_config() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts: []
control_port: 9101
"#,
    );

    let out = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(out.status.success());

    let body = std::fs::read_to_string(tmp.path().join("docker-compose.override.yaml")).unwrap();
    assert!(body.contains("127.0.0.1:9101:9101"));
    assert!(!body.contains("7799"));
}

#[test]
fn generate_with_host_network_mode_skips_ports_section() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts: []
network_mode: host
"#,
    );

    let out = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(out.status.success());

    let body = std::fs::read_to_string(tmp.path().join("docker-compose.override.yaml")).unwrap();
    assert!(body.contains(r#"network_mode: "host""#));
    assert!(!body.contains("ports:"));
}

// ────────────────────────────────────────────────────────────────────────
// --check
// ────────────────────────────────────────────────────────────────────────

#[test]
fn check_exits_zero_when_override_matches_config() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts:
  - host: /a
    container: /a
"#,
    );

    let out = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(out.status.success());

    let check = run_generate(&tmp, &["--check", "--config", cfg.to_str().unwrap()]);
    assert!(
        check.status.success(),
        "expected --check to succeed; stdout={} stderr={}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
}

#[test]
fn check_exits_one_on_mounts_drift() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts:
  - host: /a
    container: /a
"#,
    );
    let out = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(out.status.success());

    // Mutate the config — drift introduced.
    std::fs::write(
        &cfg,
        r#"
mounts:
  - host: /a
    container: /a
  - host: /b
    container: /b
"#,
    )
    .unwrap();

    let check = run_generate(&tmp, &["--check", "--config", cfg.to_str().unwrap()]);
    assert_eq!(check.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&check.stderr);
    let stdout = String::from_utf8_lossy(&check.stdout);
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.contains("stale") && combined.contains("wqm docker generate-compose"),
        "expected drift message; got: {}",
        combined
    );
}

#[test]
fn check_exits_one_when_override_is_missing() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(&tmp, "mounts: []\n");
    let check = run_generate(&tmp, &["--check", "--config", cfg.to_str().unwrap()]);
    assert_eq!(check.status.code(), Some(1));
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(
        combined.contains("missing"),
        "expected missing-file message; got: {}",
        combined
    );
}

// ────────────────────────────────────────────────────────────────────────
// --clean
// ────────────────────────────────────────────────────────────────────────

#[test]
fn clean_deletes_override_then_check_reports_missing() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts:
  - host: /a
    container: /a
"#,
    );

    let gen = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(gen.status.success());
    let override_path = tmp.path().join("docker-compose.override.yaml");
    assert!(override_path.exists());

    let clean = run_generate(&tmp, &["--clean"]);
    assert!(clean.status.success());
    assert!(!override_path.exists());

    // --check on missing override = drift exit 1.
    let check = run_generate(&tmp, &["--check", "--config", cfg.to_str().unwrap()]);
    assert_eq!(check.status.code(), Some(1));
}

#[test]
fn clean_on_missing_file_is_a_noop_success() {
    let tmp = TempDir::new().unwrap();
    let clean = run_generate(&tmp, &["--clean"]);
    assert!(
        clean.status.success(),
        "clean on missing should be ok; stderr={}",
        String::from_utf8_lossy(&clean.stderr)
    );
}

// ────────────────────────────────────────────────────────────────────────
// Network-mode rejection — T9 subtask 9.13
// ────────────────────────────────────────────────────────────────────────

#[test]
fn generate_refuses_network_mode_none() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts: []
network_mode: none
"#,
    );

    let out = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("network_mode: none") && stderr.contains("control-port"),
        "expected rejection message pointing at spec; got: {}",
        stderr
    );
    // No override file written.
    assert!(!tmp.path().join("docker-compose.override.yaml").exists());
}

#[test]
fn generate_refuses_unknown_network_mode() {
    let tmp = TempDir::new().unwrap();
    let cfg = write_config(
        &tmp,
        r#"
mounts: []
network_mode: weird-custom-network
"#,
    );

    let out = run_generate(&tmp, &["--config", cfg.to_str().unwrap()]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not supported") && stderr.contains("16-path-abstraction"));
}

// ────────────────────────────────────────────────────────────────────────
// --check and --clean are mutually exclusive at the clap layer
// ────────────────────────────────────────────────────────────────────────

#[test]
fn check_and_clean_are_mutually_exclusive() {
    let tmp = TempDir::new().unwrap();
    let out = run_generate(&tmp, &["--check", "--clean"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot be used with"),
        "expected clap exclusion error; got: {}",
        stderr
    );
}
