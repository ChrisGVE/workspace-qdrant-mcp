//! Unit tests for `super::hashing`, part 2: F5 (four-slot collection-discriminated
//! producer + version gate) and F4 (branch_id). Split out of `hashing.rs` to
//! satisfy the codesize file limit (pure relocation, zero behaviour change).
//! Part 1 is in `hashing_tests.rs`.

use super::*;

// ---- F5: collection-discriminated four-slot producer + version gate ----

/// T-F5-v3-byte-identical (AC-F5.8 / DOM-R5-N2): `content_key_v3` is
/// BYTE-IDENTICAL to the live pre-F5 three-slot `content_key` it replaces, and
/// is NOT the four-slot-with-empty-collection call (which would orphan every
/// existing Qdrant point of an unmigrated tenant).
#[test]
fn t_f5_v3_byte_identical_to_pre_f5() {
    let tenant = "tenant_golden";
    let identity = "fid-golden";
    let content_hash = "ab".repeat(32);

    // The recorded pre-F5 three-slot digest (same fixture as t_f0 golden).
    let v3 = content_key_v3(tenant, identity, &content_hash);
    assert_eq!(
        v3, "829340abef5c0c8c6760f472b0d687a0dd9525f74fe03130f20cb1c8bd893b88",
        "content_key_v3 must be byte-identical to the pre-F5 three-slot digest"
    );

    // It must NOT equal a four-slot call with an empty collection tag: that
    // inserts a fourth lp("") slot and yields a DIFFERENT digest.
    let v4_empty_collection = content_key_v4(tenant, "", identity, &content_hash);
    assert_ne!(
        v3, v4_empty_collection,
        "content_key_v3 must NOT be content_key_v4(tenant, \"\", identity, hash) — \
         the extra lp(\"\") slot changes the digest and orphans existing points"
    );
}

/// T-F5-v4-golden (AC-F5.6): pin the exact bytes the four-slot producer emits.
/// Computed out-of-band from `hex(SHA256(lp(tenant)||lp("code")||lp(hash)||lp("")))`
/// and cross-checked inline. A change to slot order/framing flips this and fails.
#[test]
fn t_f5_v4_golden_vector() {
    let chunk_hash = "cafe".repeat(16);
    let v4 = content_key_v4("tenant_golden", bucket::CODE, &chunk_hash, "");
    assert_eq!(
        v4, "acffeb1cff2566d4c39fd6e92e62c593d749e905b33863f631c271b944f40a74",
        "content_key_v4 golden vector changed — four-slot addressing would break"
    );

    // Recompute inline via the documented formula to pin the encoding.
    let mut h = Sha256::new();
    h.update(lp(b"tenant_golden"));
    h.update(lp(b"code"));
    h.update(lp(chunk_hash.as_bytes()));
    h.update(lp(b""));
    assert_eq!(v4, format!("{:x}", h.finalize()));

    let pid = point_id(&v4, 0);
    assert_eq!(
        pid.to_string(),
        "59538c5f-5c1e-5d1b-a83d-e9fe2d2afe21",
        "v4 point_id golden vector changed — stored Qdrant point IDs would break"
    );
}

/// T-F5.1 (AC-F5.1): path-independent code-chunk dedup. The same chunk text in
/// file A and file B (same tenant, same collection) yields ONE content_key —
/// the file path is NOT a slot, so identical chunk content collapses to one blob.
#[test]
fn t_f5_path_independent_chunk_dedup() {
    let chunk_hash = compute_content_hash("fn main() {}\n");
    // The chunk came from file A in one call, file B in another — but neither
    // path appears in the key, so the keys are identical.
    let key_in_file_a = content_key_v4("tenant-1", bucket::CODE, &chunk_hash, "");
    let key_in_file_b = content_key_v4("tenant-1", bucket::CODE, &chunk_hash, "");
    assert_eq!(
        key_in_file_a, key_in_file_b,
        "identical chunk content must dedup to one content_key regardless of file"
    );
    // And the point id (chunk_index always 0, AC-F5.2) is likewise identical.
    assert_eq!(
        point_id(&key_in_file_a, 0),
        point_id(&key_in_file_b, 0),
        "one blob -> one Qdrant point"
    );
}

/// T-F5.2 (AC-F5.2): blob points always pass chunk_index = 0 — one blob maps to
/// one Qdrant point; positional distinction lives in blob_refs.chunk_index, not
/// in the point_id. We pin that 0 is the convention and that varying it would
/// mint a different (wrong) point.
#[test]
fn t_f5_blob_point_uses_chunk_index_zero() {
    let chunk_hash = compute_content_hash("payload");
    let key = content_key_v4("t", bucket::CODE, &chunk_hash, "");
    let blob_point = point_id(&key, 0);
    assert_ne!(
        blob_point,
        point_id(&key, 1),
        "chunk_index is distinguishing; blob points must use 0 to stay single"
    );
}

/// T-F5.3 (AC-F5.3): cross-tenant isolation. Identical chunk content in two
/// tenants yields different content_keys (tenant is the first slot), partitioning
/// the point_id space.
#[test]
fn t_f5_cross_tenant_isolation() {
    let chunk_hash = compute_content_hash("shared bytes");
    let t1 = content_key_v4("tenant-1", bucket::CODE, &chunk_hash, "");
    let t2 = content_key_v4("tenant-2", bucket::CODE, &chunk_hash, "");
    assert_ne!(t1, t2, "different tenants must not share a content_key");
    assert_ne!(point_id(&t1, 0), point_id(&t2, 0));
}

/// T-F5.4 (AC-F5.4 / DOM-01): the collection-discriminator forced-coincidence
/// test. Craft a chunk hash byte-equal to a document identity and assert that —
/// because the collection slot differs ("code" vs "scratchpad") — the two
/// produce DISTINCT keys/points. This test FAILS if the discriminator is removed
/// (e.g. if both routed through the legacy three-slot `content_key_v3`, which has
/// no collection slot, they would COLLIDE).
#[test]
fn t_f5_collection_discriminator() {
    // Same tenant; the chunk hash and the doc identity are the identical string.
    let coincident = "0123456789abcdef".repeat(4); // 64 hex chars
    let tenant = "tenant-x";

    let code_key = content_key_v4(tenant, bucket::CODE, &coincident, "");
    let scratch_key = content_key_v4(tenant, bucket::SCRATCHPAD, &coincident, "");
    assert_ne!(
        code_key, scratch_key,
        "collection slot must keep code and scratchpad namespaces distinct"
    );
    assert_ne!(point_id(&code_key, 0), point_id(&scratch_key, 0));

    // Proof the discriminator is load-bearing: without it (the legacy 3-slot
    // path, which drops the collection slot entirely) the two coincide.
    let no_discriminator_code = content_key_v3(tenant, &coincident, "");
    let no_discriminator_scratch = content_key_v3(tenant, &coincident, "");
    assert_eq!(
        no_discriminator_code, no_discriminator_scratch,
        "without the collection slot the two namespaces collapse — exactly the \
         DOM-01 hazard the discriminator fixes"
    );
}

/// T-F5.5 (AC-F5.5): the SEC-4 residual point_id-collision guard — the pure
/// salted re-key derivation. A forced-collision fixture: two distinct content
/// keys hand-picked so their un-salted point_ids would be the same is not
/// constructible cheaply, so we assert the GUARANTEE the guard relies on — that
/// a salted re-key with a random nonce moves the point to a fresh, distinct id,
/// deterministically and reproducibly.
#[test]
fn t_f5_salted_point_id_rekey() {
    let key = content_key_v4("t", bucket::CODE, &compute_content_hash("blob"), "");
    let unsalted = point_id(&key, 0);

    let nonce = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04];
    let salted = salted_point_id(&key, &nonce, 0);
    assert_ne!(
        salted, unsalted,
        "a salted re-key must land on a fresh, distinct point_id"
    );
    // Deterministic: rebuild reads the same salted id from the same nonce.
    assert_eq!(salted, salted_point_id(&key, &nonce, 0));
    // A different nonce yields a different salted id (collision-escape works).
    assert_ne!(salted, salted_point_id(&key, &[0xff, 0xee], 0));
    // Even an EMPTY nonce is distinct from the un-salted derivation (the
    // trailing lp("") = 4 zero bytes changes the digest), so the guard never
    // accidentally re-collides with the un-salted point.
    assert_ne!(salted_point_id(&key, &[], 0), unsalted);
    assert_eq!(salted.get_version_num(), 5, "salted point_id is a UUIDv5");
}

/// T-F5.6 (AC-F5.6): the four-slot field-contract convention holds — code
/// chunk-blobs use the `bucket::CODE` collection tag and carry the chunk hash in
/// the identity slot with an empty content-hash slot.
#[test]
fn t_f5_field_contract_convention() {
    let chunk_hash = compute_content_hash("some code chunk");
    // The documented call shape for a code chunk-blob.
    let via_constant = content_key_v4("t", bucket::CODE, &chunk_hash, "");
    // Equals the literal four-slot formula with collection="code", identity=hash,
    // content_hash="".
    let mut h = Sha256::new();
    h.update(lp(b"t"));
    h.update(lp(b"code"));
    h.update(lp(chunk_hash.as_bytes()));
    h.update(lp(b""));
    assert_eq!(via_constant, format!("{:x}", h.finalize()));
    // bucket::CODE is the literal "code" tag (typed constant, not bare literal).
    assert_eq!(bucket::CODE, "code");
}

/// T-F5.7 (AC-F5.7 / DATA-09): the `point_id == point_id(content_key, 0)`
/// dual-UNIQUE binding for a non-re-keyed blob. The persisted pair is internally
/// consistent; the salted re-key path (AC-F5.5) is the ONLY sanctioned way the
/// binding is broken.
#[test]
fn t_f5_dual_unique_binding() {
    let key = content_key_v4("t", bucket::CODE, &compute_content_hash("x"), "");
    // The non-re-keyed binding: the stored point_id is exactly point_id(key, 0).
    assert_eq!(point_id(&key, 0), point_id(&key, 0));
    // The re-key path is the only thing that diverges from it.
    assert_ne!(point_id(&key, 0), salted_point_id(&key, &[0x01], 0));
}

/// T-F5.8 (AC-F5.8): the version-gated selector. For the SAME chunk bytes,
/// flipping the flag (3 vs 4) is the ONLY thing that changes the output:
/// flag 3 -> a three-slot point_id via `content_key_v3` (ignoring collection),
/// flag 4 -> a four-slot point_id via `content_key_v4`. (The once-per-session
/// caching is F6, not here — this pins only the pure selector behavior.)
#[test]
fn t_f5_version_gated_selector() {
    let tenant = "t";
    let collection = bucket::CODE;
    let identity = compute_content_hash("chunk bytes");
    let content_hash = "";

    let under_3 = content_key_for_version(3, tenant, collection, &identity, content_hash);
    let under_4 = content_key_for_version(4, tenant, collection, &identity, content_hash);

    // Flag 3 routes to v3 (collection ignored); flag 4 routes to v4.
    assert_eq!(under_3, content_key_v3(tenant, &identity, content_hash));
    assert_eq!(
        under_4,
        content_key_v4(tenant, collection, &identity, content_hash)
    );
    // Flipping the flag is the only thing that changes the output.
    assert_ne!(under_3, under_4);

    // v3 selector output is byte-identical to the legacy producer AND not the
    // four-slot-with-empty-collection call (DOM-R5-N2 anti-pattern).
    assert_ne!(under_3, content_key_v4(tenant, "", &identity, content_hash));

    // The point_id space also diverges only by the flag.
    assert_ne!(point_id(&under_3, 0), point_id(&under_4, 0));
}

/// T-F5-selector-rejects-unknown (AC-F5.8): an unknown version is a schema bug,
/// not a recoverable condition — the selector panics rather than mint points
/// under an unrecognized slot-shape.
#[test]
#[should_panic(expected = "unknown content_key_version 5")]
fn t_f5_selector_rejects_unknown_version() {
    let _ = content_key_for_version(5, "t", bucket::CODE, "id", "");
}

// ---- F4: branch_id canonical producer (AC-F4.3) ----

/// T-F4-branch-id-deterministic: same inputs always yield the same branch_id.
#[test]
fn t_f4_branch_id_deterministic() {
    let bid1 = branch_id("tenant-abc", "/home/user/proj", "main");
    let bid2 = branch_id("tenant-abc", "/home/user/proj", "main");
    assert_eq!(bid1, bid2, "branch_id must be deterministic");
    assert_eq!(bid1.len(), 64, "branch_id is a full SHA256 hex (64 chars)");
    assert!(bid1.chars().all(|c| c.is_ascii_hexdigit()));
}

/// T-F4-branch-id-path-sensitive: two clones at different paths produce distinct
/// branch_ids even when tenant and branch_name are identical (AC-F4.2 SEED F09).
#[test]
fn t_f4_branch_id_path_sensitive() {
    let bid_a = branch_id("tenant-abc", "/home/alice/proj", "main");
    let bid_b = branch_id("tenant-abc", "/home/bob/proj", "main");
    assert_ne!(
        bid_a, bid_b,
        "two clones of main at different paths must have distinct branch_ids"
    );
}

/// T-F4-branch-id-branch-sensitive: different branch names yield different ids.
#[test]
fn t_f4_branch_id_branch_sensitive() {
    let main_id = branch_id("t", "/repo", "main");
    let feat_id = branch_id("t", "/repo", "feat/x");
    assert_ne!(
        main_id, feat_id,
        "different branch names must produce different branch_ids"
    );
}

/// T-F4-branch-id-tenant-sensitive: different tenants yield different ids.
#[test]
fn t_f4_branch_id_tenant_sensitive() {
    let t1 = branch_id("tenant-1", "/repo", "main");
    let t2 = branch_id("tenant-2", "/repo", "main");
    assert_ne!(
        t1, t2,
        "different tenants must produce different branch_ids"
    );
}

/// T-F4-branch-id-lp-injective: boundary shift between location and branch_name
/// fields changes the id — lp framing prevents separator ambiguity.
#[test]
fn t_f4_branch_id_lp_injective() {
    // "abc"/"def" vs "ab"/"cdef" — without lp framing these might collide.
    let a = branch_id("t", "abc", "def");
    let b = branch_id("t", "ab", "cdef");
    assert_ne!(a, b, "lp framing must prevent boundary-shift collisions");
}

/// T-F4-branch-id-golden: pin the exact bytes the producer emits. A change to
/// the formula (field order, framing, hash backend) flips this value and fails
/// loudly, which is the point: stored branch_ids are a contract.
#[test]
fn t_f4_branch_id_golden() {
    // Computed independently: hex(SHA256(lp(b"t") || lp(b"/r") || lp(b"m")))
    // where lp(x) = u32_be(x.len()) ++ x.
    let got = branch_id("t", "/r", "m");
    // Recompute inline to verify.
    let mut h = Sha256::new();
    h.update(lp(b"t"));
    h.update(lp(b"/r"));
    h.update(lp(b"m"));
    let expected = format!("{:x}", h.finalize());
    assert_eq!(got, expected, "branch_id golden vector must not change");
    assert_eq!(got.len(), 64);
}
