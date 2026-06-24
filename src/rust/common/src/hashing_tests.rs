//! Unit tests for `super::hashing` -- core hashing, idempotency keys,
//! base-point/point-id, F1 (lp/content_key/point_id), and F0 (single-home +
//! golden vectors). Split out of `hashing.rs` to satisfy the codesize file limit
//! (pure relocation, zero behaviour change). Part 2 (F5/F4) is in
//! `hashing_tests_f5_f4.rs`.

use super::*;

#[test]
fn test_idempotency_key_generation() {
    let key1 = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/path/to/file.rs"}"#,
    )
    .unwrap();

    assert_eq!(key1.len(), 32);

    let key2 = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/path/to/file.rs"}"#,
    )
    .unwrap();
    assert_eq!(key1, key2);

    let key3 = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/path/to/other.rs"}"#,
    )
    .unwrap();
    assert_ne!(key1, key3);
}

#[test]
fn test_idempotency_key_validation() {
    let result = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "",
        "my-collection",
        "{}",
    );
    assert_eq!(result, Err(IdempotencyKeyError::EmptyTenantId));

    let result = generate_idempotency_key(ItemType::File, QueueOperation::Add, "proj", "", "{}");
    assert_eq!(result, Err(IdempotencyKeyError::EmptyCollection));

    let result = generate_idempotency_key(
        ItemType::Collection,
        QueueOperation::Add,
        "proj",
        "col",
        "{}",
    );
    assert!(matches!(
        result,
        Err(IdempotencyKeyError::InvalidOperationForType { .. })
    ));
}

#[test]
fn test_simple_idempotency_key() {
    let key1 =
        generate_simple_idempotency_key(ItemType::File, "my-collection", "/path/to/file.txt");
    let key2 =
        generate_simple_idempotency_key(ItemType::File, "my-collection", "/path/to/file.txt");
    assert_eq!(key1, key2);

    let key3 =
        generate_simple_idempotency_key(ItemType::File, "my-collection", "/path/to/other.txt");
    assert_ne!(key1, key3);
}

#[test]
fn test_compute_content_hash() {
    let hash1 = compute_content_hash("hello world");
    let hash2 = compute_content_hash("hello world");
    let hash3 = compute_content_hash("different content");

    assert_eq!(hash1, hash2);
    assert_ne!(hash1, hash3);
    assert_eq!(hash1.len(), 64);
}

#[test]
fn test_cross_language_compatibility() {
    // This test vector should produce the same hash in Rust, Python, and TypeScript
    // Input: "file|add|proj_abc123|my-project-code|{}"
    let key = generate_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        "{}",
    )
    .unwrap();

    assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(key.len(), 32);
}

#[test]
fn test_normalize_path_for_id() {
    assert_eq!(normalize_path_for_id("src/main.rs"), "src/main.rs");
    assert_eq!(normalize_path_for_id("src\\main.rs"), "src/main.rs");
    assert_eq!(normalize_path_for_id("src/dir/"), "src/dir");
    assert_eq!(
        normalize_path_for_id("src\\dir\\file.rs"),
        "src/dir/file.rs"
    );
}

#[test]
fn test_compute_base_point_deterministic() {
    let bp1 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
    let bp2 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
    assert_eq!(bp1, bp2);
    assert_eq!(bp1.len(), 32);
    assert!(bp1.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_compute_base_point_different_file_hash() {
    let bp1 = compute_base_point("tenant_abc", "src/main.rs", "hash_v1");
    let bp2 = compute_base_point("tenant_abc", "src/main.rs", "hash_v2");
    assert_ne!(bp1, bp2);
}

#[test]
fn test_compute_base_point_branch_agnostic() {
    let bp = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
    assert_eq!(bp.len(), 32);
    assert!(bp.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_compute_base_point_different_path() {
    let bp1 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
    let bp2 = compute_base_point("tenant_abc", "src/lib.rs", "deadbeef");
    assert_ne!(bp1, bp2);
}

#[test]
fn test_compute_base_point_different_tenant() {
    let bp1 = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
    let bp2 = compute_base_point("tenant_xyz", "src/main.rs", "deadbeef");
    assert_ne!(bp1, bp2);
}

#[test]
fn test_compute_base_point_path_normalization() {
    let bp1 = compute_base_point("t", "src/main.rs", "h");
    let bp2 = compute_base_point("t", "src\\main.rs", "h");
    assert_eq!(bp1, bp2);
}

#[test]
fn test_compute_point_id_deterministic() {
    let bp = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
    let pid1 = compute_point_id(&bp, 0);
    let pid2 = compute_point_id(&bp, 0);
    assert_eq!(pid1, pid2);
    assert_eq!(pid1.len(), 32);
}

#[test]
fn test_compute_point_id_different_chunk_index() {
    let bp = compute_base_point("tenant_abc", "src/main.rs", "deadbeef");
    let pid0 = compute_point_id(&bp, 0);
    let pid1 = compute_point_id(&bp, 1);
    let pid2 = compute_point_id(&bp, 2);
    assert_ne!(pid0, pid1);
    assert_ne!(pid1, pid2);
    assert_ne!(pid0, pid2);
}

#[test]
fn test_compute_point_id_different_base_point() {
    let bp1 = compute_base_point("tenant_abc", "src/main.rs", "hash_v1");
    let bp2 = compute_base_point("tenant_abc", "src/main.rs", "hash_v2");
    let pid1 = compute_point_id(&bp1, 0);
    let pid2 = compute_point_id(&bp2, 0);
    assert_ne!(pid1, pid2);
}

#[test]
fn test_base_point_cross_language_test_vector() {
    let bp = compute_base_point("test_tenant", "src/example.rs", "abc123hash");
    assert_eq!(bp, "d08103c2d8f553544dabeb4737fd32b4");

    let pid = compute_point_id(&bp, 0);
    assert_eq!(pid.len(), 32);
}

#[test]
fn hash_rejects_directory() {
    let dir = tempfile::TempDir::new().unwrap();
    let result = compute_file_hash(dir.path());
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn hash_rejects_dev_null() {
    let path = Path::new("/dev/null");
    if !path.exists() {
        return;
    }
    let result = compute_file_hash(path);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn hash_regular_file_works() {
    let dir = tempfile::TempDir::new().unwrap();
    let fp = dir.path().join("test.txt");
    std::fs::write(&fp, b"hello").unwrap();
    assert!(compute_file_hash(&fp).is_ok());
}

// ---- F1: lp / content_key / point_id (branch-lineage) ----

#[test]
fn test_lp_frames_with_big_endian_length() {
    assert_eq!(lp(b""), vec![0, 0, 0, 0]);
    assert_eq!(lp(b"ab"), vec![0, 0, 0, 2, b'a', b'b']);
    // 256-byte payload → length 0x00000100 prefix.
    let big = vec![0x5au8; 256];
    let framed = lp(&big);
    assert_eq!(&framed[..4], &[0, 0, 1, 0]);
    assert_eq!(&framed[4..], &big[..]);
}

proptest::proptest! {
    /// T-F1-lp-injective (DOM-01): the framed concatenation `lp(a)‖lp(b)` is
    /// injective in `(a, b)` — distinct field pairs never produce the same bytes.
    /// This is the property a bare separator (`a|b`) cannot offer, since data may
    /// contain the separator. We assert the contrapositive: differing pairs differ.
    #[test]
    fn t_f1_lp_injective(
        a in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
        b in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
        c in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
        d in proptest::collection::vec(proptest::num::u8::ANY, 0..40),
    ) {
        let mut left = lp(&a);
        left.extend_from_slice(&lp(&b));
        let mut right = lp(&c);
        right.extend_from_slice(&lp(&d));
        if (a, b) == (c, d) {
            proptest::prop_assert_eq!(left, right);
        } else {
            proptest::prop_assert_ne!(left, right);
        }
    }
}

#[test]
fn t_f1_lp_defeats_separator_ambiguity() {
    // The bare-separator hazard: "a|bc" == "ab|c" collides; lp framing must not.
    let mut x = lp(b"a");
    x.extend_from_slice(&lp(b"bc"));
    let mut y = lp(b"ab");
    y.extend_from_slice(&lp(b"c"));
    assert_ne!(x, y);
}

#[test]
fn t_f1_content_key_is_full_32_bytes_and_lp_framed() {
    let ck = content_key_v3("tenant_abc", "fid-1", &"de".repeat(32));
    assert_eq!(
        ck.len(),
        64,
        "content_key keeps full 32-byte digest (64 hex)"
    );
    assert!(ck.chars().all(|c| c.is_ascii_hexdigit()));

    // Independently recompute via the documented formula to pin the encoding.
    let mut h = Sha256::new();
    h.update(lp(b"tenant_abc"));
    h.update(lp(b"fid-1"));
    h.update(lp("de".repeat(32).as_bytes()));
    let expected = format!("{:x}", h.finalize());
    assert_eq!(ck, expected);
}

#[test]
fn t_f1_content_key_field_boundaries_matter() {
    // Moving a character across the identity/hash boundary must change the key.
    let a = content_key_v3("t", "ab", "c");
    let b = content_key_v3("t", "a", "bc");
    assert_ne!(a, b);
}

#[test]
fn t_f1_encoding_agreement() {
    // N7: the SAME (tenant, file_identity_id, file_hash_hex) yields the SAME
    // content_key regardless of which caller computes it — there is one producer.
    let tenant = "tenant_xyz";
    let fid = "file-identity-7";
    let file_hash_hex = "abc123".repeat(10) + "abcd"; // 64-char hex string
    let tagger_side = content_key_v3(tenant, fid, &file_hash_hex);
    let rekey_side = content_key_v3(tenant, fid, &file_hash_hex);
    assert_eq!(tagger_side, rekey_side);
    // The hex STRING (not raw bytes) is the input: a different rendering differs.
    let raw_bytes_rendering = content_key_v3(tenant, fid, "abc123");
    assert_ne!(tagger_side, raw_bytes_rendering);
}

#[test]
fn t_f1_pointid_stability() {
    let ck = content_key_v3("tenant", "fid", &"00".repeat(32));
    let p1 = point_id(&ck, 0);
    let p2 = point_id(&ck, 0);
    assert_eq!(p1, p2, "same content_key + chunk → same point_id");
    assert_eq!(p1.get_version_num(), 5, "point_id is a UUIDv5");
    assert_ne!(
        point_id(&ck, 0),
        point_id(&ck, 1),
        "chunk index is distinguishing"
    );
    // content_point_id is exactly point_id(content_key_v3(tenant, identity, "")) —
    // one flow, identity-addressed (empty content-hash slot).
    let composed = content_point_id("tenant", "doc-7", 3);
    assert_eq!(
        composed,
        point_id(&content_key_v3("tenant", "doc-7", ""), 3)
    );
}

#[test]
fn t_f1_pointid_chunk_index_lp_framed() {
    // chunk_index is framed as lp(u32_be), so it cannot bleed into content_key.
    // Distinct content_keys with distinct chunk indices stay distinct.
    let ck_a = content_key_v3("t", "a", &"11".repeat(32));
    let ck_b = content_key_v3("t", "b", &"11".repeat(32));
    assert_ne!(point_id(&ck_a, 1), point_id(&ck_b, 1));
    assert_ne!(point_id(&ck_a, 1), point_id(&ck_a, 2));
}

// ---- F0: producers pinned to one canonical home (FP-2) + golden output lock ----

/// T-F0-golden: literal golden values for the `content_key` and `point_id`
/// producers. These pin the EXACT bytes the canonical producers emit. Any change
/// to the framing, field order, hash backend, or [`POINT_NS`] namespace flips
/// these values and the test fails loudly — which is the point: content addressing
/// is a stored contract (point IDs already persisted in every corpus), so a
/// behavioral change must be a deliberate, version-gated migration, never silent
/// drift. The golden values were computed out-of-band from the documented formula
/// (`hex(SHA256(lp..))` and `UUIDv5(POINT_NS, lp..)`).
#[test]
fn t_f0_producer_golden_vectors() {
    let ck = content_key_v3("tenant_golden", "fid-golden", &"ab".repeat(32));
    assert_eq!(
        ck, "829340abef5c0c8c6760f472b0d687a0dd9525f74fe03130f20cb1c8bd893b88",
        "content_key_v3 golden vector changed — content addressing would break"
    );
    let pid = point_id(&ck, 0);
    assert_eq!(
        pid.to_string(),
        "d2736140-73aa-5ae6-b133-71d846462533",
        "point_id golden vector changed — stored Qdrant point IDs would break"
    );
}

/// T-F0-single-home (FP-2): the content-addressing producers have exactly ONE
/// definition tree-wide, here in `wqm-common::hashing`. A second definition
/// anywhere is the drift FP-2 forbids — two producers can silently diverge and
/// split a corpus into incompatible key spaces. The guard walks the workspace
/// Rust source and counts `pub fn` producer signatures, asserting one each.
#[test]
fn t_f0_producers_have_single_definition() {
    use std::path::{Path, PathBuf};

    // Locate the workspace source root (src/rust): walk up from this crate's
    // manifest dir until a directory holding both the `common` and `daemon`
    // members. Outside the in-repo tree (e.g. a packaged crate) the guard no-ops.
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws_root = loop {
        if root.join("common").is_dir() && root.join("daemon").is_dir() {
            break Some(root.clone());
        }
        if !root.pop() {
            break None;
        }
    };
    let Some(ws_root) = ws_root else {
        return;
    };

    let producers = [
        "pub fn content_key_v3(",
        "pub fn content_key_v4(",
        "pub fn point_id(",
        "pub fn content_point_id(",
        "pub fn branch_id(",
    ];
    let mut counts = [0usize; 5];

    fn walk(dir: &Path, f: &mut impl FnMut(&Path)) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                walk(&p, f);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                f(&p);
            }
        }
    }

    walk(&ws_root, &mut |p| {
        let Ok(src) = std::fs::read_to_string(p) else {
            return;
        };
        for line in src.lines() {
            let t = line.trim_start();
            for (i, pat) in producers.iter().enumerate() {
                if t.starts_with(pat) {
                    counts[i] += 1;
                }
            }
        }
    });

    for (i, pat) in producers.iter().enumerate() {
        assert_eq!(
            counts[i], 1,
            "expected exactly one definition of `{}` tree-wide, found {} (FP-2 single-home violated)",
            pat, counts[i]
        );
    }
}
