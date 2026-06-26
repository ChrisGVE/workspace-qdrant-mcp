//! Tests for the canonical-path abstraction.
//!
//! Covers each of the nine normalization rules from spec §3.1, the
//! corresponding negative cases, [`MountMap`] resolution semantics, and
//! property-based round-trip invariants.

#[cfg(unix)]
use std::ffi::OsStr;
#[cfg(unix)]
use std::path::PathBuf;

use proptest::prelude::*;

use super::{CanonicalPath, LocalPath, MountMap, PathError};

// ---------------------------------------------------------------------------
// §3.1 normalization rules — one test per rule, positive case.
// ---------------------------------------------------------------------------

#[test]
fn rule_1_absolute_required() {
    // Positive: absolute input survives.
    let p = CanonicalPath::from_user_input("/Users/username").unwrap();
    assert_eq!(p.as_str(), "/Users/username");
}

#[test]
fn rule_2_tilde_expansion() {
    // shellexpand::tilde calls dirs::home_dir(), which uses
    // platform APIs (NSHomeDirectory on macOS) rather than $HOME.
    // The test cannot stub HOME deterministically, so it asserts
    // against the real home directory.
    let home = dirs::home_dir().expect("HOME or equivalent must be set for tests");
    let home_str = home.to_str().expect("home path must be UTF-8");

    let p = CanonicalPath::from_user_input("~/project").unwrap();
    assert_eq!(p.as_str(), format!("{home_str}/project"));
}

#[test]
fn rule_3_dot_segments_removed() {
    let p = CanonicalPath::from_user_input("/Users/./username/./dev").unwrap();
    assert_eq!(p.as_str(), "/Users/username/dev");
}

#[test]
fn rule_4_parent_dir_rejected() {
    let err = CanonicalPath::from_user_input("/Users/username/../other").unwrap_err();
    assert!(matches!(err, PathError::ContainsParentDir(_)));
}

#[test]
fn rule_5_duplicate_slash_collapsed() {
    let p = CanonicalPath::from_user_input("/Users//username///dev").unwrap();
    assert_eq!(p.as_str(), "/Users/username/dev");
}

#[test]
fn rule_6_case_preserved() {
    let p = CanonicalPath::from_user_input("/Users/Username/DevTools").unwrap();
    assert_eq!(p.as_str(), "/Users/Username/DevTools");
}

#[test]
fn rule_7_no_symlink_resolution() {
    // Rule 7 is satisfied by not calling canonicalize(). We can't easily
    // test "didn't resolve a symlink" directly, but we CAN assert that
    // the function does not touch the filesystem: feed a path that does
    // not exist, and expect success.
    let p =
        CanonicalPath::from_user_input("/definitely/does/not/exist/anywhere/on/this/machine/abc")
            .unwrap();
    assert_eq!(
        p.as_str(),
        "/definitely/does/not/exist/anywhere/on/this/machine/abc"
    );
}

#[test]
fn rule_8_no_fs_access() {
    // Same as rule 7 — non-existent input must succeed.
    let p = CanonicalPath::from_user_input("/nope/nope/nope").unwrap();
    assert_eq!(p.as_str(), "/nope/nope/nope");
}

#[test]
fn rule_9_utf8_required_on_construction() {
    // `CanonicalPath::from_user_input` accepts `&str`, which is
    // UTF-8 by Rust's type system. There is no `from_user_input`
    // path that can carry non-UTF-8 — the language enforces it.
    // This test asserts that UTF-8 input survives, sanity-checking
    // rule 9's positive side. The negative branch is exercised by
    // `rule_9_local_to_canonical_rejects_non_utf8`.
    let p = CanonicalPath::from_user_input("/utf8/ok").unwrap();
    assert_eq!(p.as_str(), "/utf8/ok");
}

#[cfg(unix)]
#[test]
fn rule_9_local_to_canonical_rejects_non_utf8() {
    use std::os::unix::ffi::OsStrExt;

    // Build a LocalPath wrapping a non-UTF-8 PathBuf via the
    // test-only backdoor constructor. The public API cannot
    // produce one because CanonicalPath is UTF-8 by definition.
    let bad_pathbuf = PathBuf::from(OsStr::from_bytes(b"/tmp/\xff\xfe"));
    let local = LocalPath::from_pathbuf_for_test(bad_pathbuf);

    let err = local.to_canonical(&MountMap::identity()).unwrap_err();
    assert!(matches!(err, PathError::NonUtf8));
}

// ---------------------------------------------------------------------------
// Negative cases.
// ---------------------------------------------------------------------------

#[test]
fn error_relative_input() {
    let err = CanonicalPath::from_user_input("relative/path").unwrap_err();
    assert!(matches!(err, PathError::RelativeInput(_)));
}

#[test]
fn error_relative_input_dot_prefix() {
    let err = CanonicalPath::from_user_input("./relative").unwrap_err();
    assert!(matches!(err, PathError::RelativeInput(_)));
}

#[test]
fn error_empty_path() {
    let err = CanonicalPath::from_user_input("").unwrap_err();
    assert!(matches!(err, PathError::EmptyPath));
}

#[test]
fn error_embedded_nul() {
    let err = CanonicalPath::from_user_input("/Users/username\0/dev").unwrap_err();
    assert!(matches!(err, PathError::InvalidNormalization(_)));
}

#[test]
fn error_parent_dir_at_end() {
    let err = CanonicalPath::from_user_input("/a/b/..").unwrap_err();
    assert!(matches!(err, PathError::ContainsParentDir(_)));
}

#[test]
fn error_parent_dir_at_start_after_root() {
    let err = CanonicalPath::from_user_input("/..").unwrap_err();
    assert!(matches!(err, PathError::ContainsParentDir(_)));
}

// ---------------------------------------------------------------------------
// from_validated: same rules, plus debug-assertion of canonical form.
// ---------------------------------------------------------------------------

#[test]
fn from_validated_accepts_canonical() {
    let p = CanonicalPath::from_validated("/Users/username/dev".to_string()).unwrap();
    assert_eq!(p.as_str(), "/Users/username/dev");
}

#[test]
fn from_validated_rejects_relative() {
    let err = CanonicalPath::from_validated("relative".to_string()).unwrap_err();
    assert!(matches!(err, PathError::RelativeInput(_)));
}

#[test]
fn from_validated_rejects_parent_dir() {
    let err = CanonicalPath::from_validated("/a/../b".to_string()).unwrap_err();
    assert!(matches!(err, PathError::ContainsParentDir(_)));
}

// ---------------------------------------------------------------------------
// Equality, hashing, display.
// ---------------------------------------------------------------------------

#[test]
fn canonical_path_equal_after_normalize() {
    let a = CanonicalPath::from_user_input("/Users/username/./dev").unwrap();
    let b = CanonicalPath::from_user_input("/Users//username/dev").unwrap();
    assert_eq!(a, b);
}

#[test]
fn canonical_path_display_matches_as_str() {
    let p = CanonicalPath::from_user_input("/a/b/c").unwrap();
    assert_eq!(format!("{p}"), "/a/b/c");
    assert_eq!(p.as_ref(), "/a/b/c");
}

#[test]
fn canonical_path_into_string() {
    let p = CanonicalPath::from_user_input("/a/b").unwrap();
    let s: String = p.into_string();
    assert_eq!(s, "/a/b");
}

#[test]
fn canonical_path_serde_roundtrip() {
    let p = CanonicalPath::from_user_input("/Users/username/dev").unwrap();
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(json, "\"/Users/username/dev\"");
    let back: CanonicalPath = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}

// ---------------------------------------------------------------------------
// MountMap construction and resolution.
// ---------------------------------------------------------------------------

#[test]
fn mountmap_identity_is_empty() {
    let m = MountMap::identity();
    assert!(m.is_identity());
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

#[test]
fn mountmap_new_single_entry() {
    let m = MountMap::new(vec![(
        "/Users/username/dev".to_string(),
        "/mnt/dev".to_string(),
    )])
    .unwrap();
    assert!(!m.is_identity());
    assert_eq!(m.len(), 1);
}

#[test]
fn mountmap_duplicate_host_rejected() {
    let err = MountMap::new(vec![
        ("/Users/username".to_string(), "/mnt/a".to_string()),
        ("/Users/username".to_string(), "/mnt/b".to_string()),
    ])
    .unwrap_err();
    assert!(matches!(err, PathError::MountMapError(_)));
}

#[test]
fn mountmap_duplicate_container_rejected() {
    let err = MountMap::new(vec![
        ("/a".to_string(), "/mnt".to_string()),
        ("/b".to_string(), "/mnt".to_string()),
    ])
    .unwrap_err();
    assert!(matches!(err, PathError::MountMapError(_)));
}

#[test]
fn mountmap_tilde_expanded_for_duplicate_check() {
    // Two host entries that resolve to the same canonical path after
    // tilde expansion must collide. Resolve the real HOME because
    // shellexpand::tilde uses dirs::home_dir() (platform API), which
    // is not stubbable via $HOME on macOS.
    let home = dirs::home_dir().expect("HOME or equivalent must be set for tests");
    let home_dev = home.join("dev");
    let home_dev_str = home_dev.to_str().expect("home path must be UTF-8");

    let result = MountMap::new(vec![
        ("~/dev".to_string(), "/mnt/a".to_string()),
        (home_dev_str.to_string(), "/mnt/b".to_string()),
    ]);

    assert!(matches!(result, Err(PathError::MountMapError(_))));
}

#[test]
fn mountmap_longest_prefix_wins() {
    let m = MountMap::new(vec![
        ("/Users/username".to_string(), "/Users/username".to_string()),
        ("/Users/username/dev".to_string(), "/mnt/dev".to_string()),
    ])
    .unwrap();
    let cp = CanonicalPath::from_user_input("/Users/username/dev/project/file.txt").unwrap();
    let local = LocalPath::from_canonical(&cp, &m).unwrap();
    // /Users/username/dev wins over /Users/username because it is longer.
    assert_eq!(
        local.as_std_path().to_str().unwrap(),
        "/mnt/dev/project/file.txt"
    );
}

#[test]
fn mountmap_component_boundary_required() {
    // /Users/username/dev must NOT match /Users/username/development.
    let m = MountMap::new(vec![(
        "/Users/username/dev".to_string(),
        "/mnt/dev".to_string(),
    )])
    .unwrap();
    let cp = CanonicalPath::from_user_input("/Users/username/development/file.txt").unwrap();
    let err = LocalPath::from_canonical(&cp, &m).unwrap_err();
    assert!(matches!(err, PathError::NoMountCoverage { .. }));
}

#[test]
fn mountmap_no_coverage_error() {
    let m = MountMap::new(vec![(
        "/Users/username/dev".to_string(),
        "/mnt/dev".to_string(),
    )])
    .unwrap();
    let cp = CanonicalPath::from_user_input("/etc/config.yaml").unwrap();
    let err = LocalPath::from_canonical(&cp, &m).unwrap_err();
    assert!(matches!(err, PathError::NoMountCoverage { .. }));
}

// ---------------------------------------------------------------------------
// LocalPath translation.
// ---------------------------------------------------------------------------

#[test]
fn local_path_identity_passes_through() {
    let cp = CanonicalPath::from_user_input("/Users/username/dev/project").unwrap();
    let local = LocalPath::from_canonical(&cp, &MountMap::identity()).unwrap();
    assert_eq!(
        local.as_std_path().to_str().unwrap(),
        "/Users/username/dev/project"
    );
}

#[test]
fn local_path_mirror_mount() {
    let m = MountMap::new(vec![(
        "/Users/username/dev".to_string(),
        "/Users/username/dev".to_string(),
    )])
    .unwrap();
    let cp = CanonicalPath::from_user_input("/Users/username/dev/project/file.txt").unwrap();
    let local = LocalPath::from_canonical(&cp, &m).unwrap();
    assert_eq!(
        local.as_std_path().to_str().unwrap(),
        "/Users/username/dev/project/file.txt"
    );
}

#[test]
fn local_path_non_mirror_mount() {
    let m = MountMap::new(vec![(
        "/Volumes/External/books".to_string(),
        "/mnt/books".to_string(),
    )])
    .unwrap();
    let cp = CanonicalPath::from_user_input("/Volumes/External/books/rust.pdf").unwrap();
    let local = LocalPath::from_canonical(&cp, &m).unwrap();
    assert_eq!(local.as_std_path().to_str().unwrap(), "/mnt/books/rust.pdf");
}

#[test]
fn local_path_to_canonical_identity() {
    let m = MountMap::identity();
    let cp = CanonicalPath::from_user_input("/Users/username/dev").unwrap();
    let local = LocalPath::from_canonical(&cp, &m).unwrap();
    let back = local.to_canonical(&m).unwrap();
    assert_eq!(cp, back);
}

#[test]
fn local_path_to_canonical_non_mirror() {
    let m = MountMap::new(vec![("/Users/username".to_string(), "/home/user".to_string())]).unwrap();
    let cp = CanonicalPath::from_user_input("/Users/username/project/file.txt").unwrap();
    let local = LocalPath::from_canonical(&cp, &m).unwrap();
    let back = local.to_canonical(&m).unwrap();
    assert_eq!(cp, back);
}

#[test]
fn local_path_to_canonical_no_coverage() {
    let m = MountMap::new(vec![("/a".to_string(), "/mnt/a".to_string())]).unwrap();
    // Build a LocalPath whose inner PathBuf is outside any container prefix.
    let cp = CanonicalPath::from_user_input("/a/file").unwrap();
    let local = LocalPath::from_canonical(&cp, &m).unwrap();
    // Now drop the entry and reconstruct: simulate "no mount covers".
    let empty_map =
        MountMap::new(vec![("/different".to_string(), "/mnt/diff".to_string())]).unwrap();
    let err = local.to_canonical(&empty_map).unwrap_err();
    assert!(matches!(err, PathError::NoMountCoverage { .. }));
}

#[test]
fn local_path_root_prefix() {
    // Root prefix `/` matches every absolute path.
    let m = MountMap::new(vec![("/".to_string(), "/container-root".to_string())]).unwrap();
    let cp = CanonicalPath::from_user_input("/etc/config").unwrap();
    let local = LocalPath::from_canonical(&cp, &m).unwrap();
    assert_eq!(
        local.as_std_path().to_str().unwrap(),
        "/container-root/etc/config"
    );
}

// ---------------------------------------------------------------------------
// Property-based tests.
// ---------------------------------------------------------------------------

proptest! {
    /// Normalization is idempotent: feeding the canonical form back in
    /// produces the same canonical form.
    #[test]
    fn normalize_is_idempotent(segments in prop::collection::vec("[a-zA-Z0-9_-]{1,8}", 1..6)) {
        let input = format!("/{}", segments.join("/"));
        let c1 = CanonicalPath::from_user_input(&input).unwrap();
        let c2 = CanonicalPath::from_user_input(c1.as_str()).unwrap();
        prop_assert_eq!(c1, c2);
    }

    /// Identity-map round-trip: from_canonical -> to_canonical is the
    /// identity function on canonical-shaped inputs.
    #[test]
    fn identity_roundtrip(segments in prop::collection::vec("[a-zA-Z0-9_-]{1,8}", 1..6)) {
        let input = format!("/{}", segments.join("/"));
        let cp = CanonicalPath::from_user_input(&input).unwrap();
        let m = MountMap::identity();
        let local = LocalPath::from_canonical(&cp, &m).unwrap();
        let back = local.to_canonical(&m).unwrap();
        prop_assert_eq!(cp, back);
    }

    /// Non-trivial mount-map round-trip: also the identity in terms of
    /// canonical form, for any path covered by a single mount entry.
    #[test]
    fn non_mirror_roundtrip(
        suffix in prop::collection::vec("[a-zA-Z0-9_-]{1,8}", 1..5),
    ) {
        let host_root = "/Users/username/dev";
        let container_root = "/mnt/dev";
        let m = MountMap::new(vec![(
            host_root.to_string(),
            container_root.to_string(),
        )])
        .unwrap();
        let canonical_str = format!("{host_root}/{}", suffix.join("/"));
        let cp = CanonicalPath::from_user_input(&canonical_str).unwrap();
        let local = LocalPath::from_canonical(&cp, &m).unwrap();
        let back = local.to_canonical(&m).unwrap();
        prop_assert_eq!(cp, back);
    }

    /// Multiple `.` segments and `//` duplicates always normalize away.
    #[test]
    fn dots_and_slashes_normalize_out(segments in prop::collection::vec("[a-zA-Z0-9_-]{1,8}", 1..6)) {
        let noisy = format!("/./{}", segments.join("//./"));
        let clean = format!("/{}", segments.join("/"));
        let c1 = CanonicalPath::from_user_input(&noisy).unwrap();
        let c2 = CanonicalPath::from_user_input(&clean).unwrap();
        prop_assert_eq!(c1, c2);
    }
}
