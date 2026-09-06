//! `P04-GT002-WO075` -- `C-ty-secret`.
//!
//! Two of these tests assert what a `Secret` does NOT do. Redaction is only worth
//! anything if the bytes are absent from the formatted output, so each test
//! builds the value from a distinctive marker and looks for that marker rather
//! than trusting that the redaction token being present implies nothing else was
//! written.

use wqm_common::secret::Secret;

/// A byte string no format impl would emit by accident, so finding it in output
/// is unambiguous evidence the credential leaked.
const MARKER: &[u8] = b"grep-me-if-this-leaks-9f3a";

fn a_secret() -> Secret {
    Secret::new(MARKER.to_vec())
}

fn marker_text() -> String {
    String::from_utf8(MARKER.to_vec()).expect("the marker is ASCII")
}

#[test]
fn debug_writes_the_redaction_token_and_not_the_bytes() {
    let rendered = format!("{:?}", a_secret());

    assert_eq!(rendered, "Secret(<redacted>)");
    assert!(
        !rendered.contains(&marker_text()),
        "Debug leaked the credential: {rendered}"
    );
}

#[test]
fn display_writes_the_redaction_token_and_not_the_bytes() {
    let rendered = format!("{}", a_secret());

    assert_eq!(rendered, "<redacted>");
    assert!(
        !rendered.contains(&marker_text()),
        "Display leaked the credential: {rendered}"
    );
}

#[test]
fn a_secret_nested_in_a_derived_debug_stays_redacted() {
    // I7 has to hold wherever the value ends up, not only where it is formatted
    // on purpose -- a config struct that derives `Debug` is the realistic leak.
    #[derive(Debug)]
    struct Config {
        api_key: Secret,
    }

    let config = Config {
        api_key: a_secret(),
    };
    let rendered = format!("{config:?}");

    // The field still holds the credential -- redaction is a formatting property,
    // not a loss of the value.
    assert_eq!(config.api_key.expose(), MARKER);
    assert!(rendered.contains("Secret(<redacted>)"), "{rendered}");
    assert!(
        !rendered.contains(&marker_text()),
        "a derived Debug leaked the credential: {rendered}"
    );
}

#[test]
fn expose_round_trips_the_bytes() {
    assert_eq!(a_secret().expose(), MARKER);
}

#[test]
fn expose_carries_arbitrary_bytes_not_just_text() {
    // A credential is bytes, not a `String`: the type must not assume UTF-8.
    let raw = vec![0x00, 0xff, 0x80, 0x0a];

    assert_eq!(Secret::new(raw.clone()).expose(), raw.as_slice());
}

#[test]
fn a_clone_carries_the_same_value() {
    // The fragment's Behavior section shares a `Secret` by clone or `Arc`; a
    // clone that lost the bytes would fail at the injection site, not here.
    let original = a_secret();
    let copy = original.clone();

    assert_eq!(copy.expose(), original.expose());
}

#[test]
fn the_type_is_zeroize_on_drop() {
    // Reading freed memory to observe the wipe is undefined behaviour, so the
    // property is asserted where it is actually decidable: at the type level.
    //
    // What this test does and does NOT hold. `ZeroizeOnDrop` is an empty marker
    // trait and `secret.rs` implements it by hand, so this bound only pins that
    // the marker stays PRESENT -- it cannot see the field and would pass
    // unchanged if `Zeroizing<Vec<u8>>` became a bare `Vec<u8>`. The field is
    // held by the `const` assertion beside that impl (`the_field_wipes_itself`
    // in `src/secret.rs`), which names `secret.0` and fails to compile if the
    // field stops wiping. Both are needed: this one keeps the marker callers can
    // bind on, that one keeps the marker honest.
    fn assert_zeroize_on_drop<T: zeroize::ZeroizeOnDrop>() {}

    assert_zeroize_on_drop::<Secret>();
}
