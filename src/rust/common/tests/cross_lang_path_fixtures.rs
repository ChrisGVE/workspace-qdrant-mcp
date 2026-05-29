//! Cross-language path-normalization fixtures runner (Rust side).
//!
//! Loads `tests/path-fixtures/cases.json` from the repository root and
//! asserts that `wqm_common::paths::CanonicalPath::from_user_input` and
//! `wqm_common::paths::PathError` agree with every entry. The same JSON
//! is consumed by the TypeScript and PowerShell runners — keeping all
//! three implementations honest.
//!
//! Run with:
//!   cargo test --manifest-path src/rust/Cargo.toml -p wqm-common --test cross_lang_path_fixtures

use std::path::PathBuf;

use serde_json::Value;
use wqm_common::paths::{CanonicalPath, PathError};

/// Walk up from the test binary's manifest dir to find
/// `tests/path-fixtures/cases.json` at the repository root. `cargo test`
/// sets `CARGO_MANIFEST_DIR` to `src/rust/common`, so we walk up four
/// levels to reach the repo root.
fn fixtures_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // src/rust/common -> repo root is 3 levels up
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("expected repo root reachable from CARGO_MANIFEST_DIR");
    repo_root
        .join("tests")
        .join("path-fixtures")
        .join("cases.json")
}

fn load_cases() -> Value {
    let path = fixtures_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path.display(), e))
}

fn classify_error(err: &PathError) -> &'static str {
    match err {
        PathError::RelativeInput(_) => "relative",
        PathError::ContainsParentDir(_) => "dot-dot",
        PathError::EmptyPath => "empty",
        PathError::NonUtf8 => "non-utf8",
        PathError::InvalidNormalization(msg) if msg.contains("NUL") => "nul-byte",
        PathError::InvalidNormalization(_) => "invalid",
        PathError::MountMapError(_) | PathError::NoMountCoverage { .. } => "other",
    }
}

#[test]
fn normalize_positive_cases() {
    let cases = load_cases();
    let entries = cases
        .get("normalize")
        .and_then(Value::as_array)
        .expect("`normalize` array missing");

    let mut failures = Vec::new();
    for entry in entries {
        let name = entry["name"].as_str().expect("case missing `name`");
        let input = entry["input"].as_str().expect("case missing `input`");
        let expected = entry["expected"].as_str().expect("case missing `expected`");

        match CanonicalPath::from_user_input(input) {
            Ok(canon) if canon.as_str() == expected => {} // pass
            Ok(canon) => failures.push(format!(
                "[{name}] input {:?} normalized to {:?}, expected {:?}",
                input,
                canon.as_str(),
                expected
            )),
            Err(e) => failures.push(format!(
                "[{name}] input {:?} unexpectedly errored: {e:?}",
                input
            )),
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} normalize positive case(s) failed:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}

#[test]
fn normalize_error_cases() {
    let cases = load_cases();
    let entries = cases
        .get("normalize_errors")
        .and_then(Value::as_array)
        .expect("`normalize_errors` array missing");

    let mut failures = Vec::new();
    for entry in entries {
        let name = entry["name"].as_str().expect("case missing `name`");
        let input = entry["input"].as_str().expect("case missing `input`");
        let expected_kind = entry["expectedKind"]
            .as_str()
            .expect("case missing `expectedKind`");

        match CanonicalPath::from_user_input(input) {
            Ok(canon) => failures.push(format!(
                "[{name}] input {:?} unexpectedly succeeded -> {:?}; expected error kind {:?}",
                input,
                canon.as_str(),
                expected_kind
            )),
            Err(e) => {
                let actual = classify_error(&e);
                if actual != expected_kind {
                    failures.push(format!(
                        "[{name}] input {:?} returned error {:?} ({}), expected kind {:?}",
                        input, e, actual, expected_kind
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} normalize error case(s) failed:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}
