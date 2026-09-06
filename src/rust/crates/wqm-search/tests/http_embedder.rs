//! `P04-GT002-WO001` -- `C-adp-N17-http`.
//!
//! Vocabulary only: the adapter binds an endpoint, a dialect, a model and a
//! credential, and does nothing with any of them. What is worth asserting about
//! a type with no behavior is the shape its future implementation depends on,
//! so each test below names the later failure it forecloses rather than
//! restating that a constructor stores its arguments.
//!
//! Two of the guarantees are compile-time and live as `compile_fail` doctests on
//! the type itself (`src/http_embedder.rs`): that there is no `Default`, and
//! that the credential field is not reachable. A runtime test cannot express
//! either -- the absence of an impl is not observable from a value.
//!
//! This file being a separate crate is load-bearing for
//! `the_dialect_set_is_closed_to_other_crates`; see that test.

use std::collections::HashSet;

use wqm_common::secret::Secret;
use wqm_search::http_embedder::{EmbeddingDialect, HttpEmbedder};

/// A key no format impl would emit by accident, so finding it in output is
/// unambiguous evidence the credential leaked.
const PLANTED_KEY: &[u8] = b"sk-PLANTED-KEY-9f3a";

fn planted_key_text() -> String {
    String::from_utf8(PLANTED_KEY.to_vec()).expect("the planted key is ASCII")
}

fn an_adapter() -> HttpEmbedder {
    HttpEmbedder::new(
        "https://api.example.test/v1".to_owned(),
        EmbeddingDialect::OpenAiCompatible,
        "text-embedding-3-small".to_owned(),
        Secret::new(PLANTED_KEY.to_vec()),
    )
}

/// I7/F16: the injected key must not appear in a formatted rendering of the
/// adapter.
///
/// This protects against the realistic leak rather than a deliberate one -- a
/// `Debug` derived on this struct, or on a config that contains it, printed into
/// a log or an error. The assertion searches for the planted bytes and not for
/// the redaction token, because a rendering can contain the token and the key
/// both; finding the token proves only that `Secret` was formatted somewhere.
#[test]
fn debug_does_not_leak_the_injected_key() {
    let rendered = format!("{:?}", an_adapter());

    assert!(
        !rendered.contains(&planted_key_text()),
        "Debug leaked the injected credential: {rendered}"
    );
    assert!(
        rendered.contains("Secret(<redacted>)"),
        "the credential field should render as the redaction token: {rendered}"
    );
}

/// The non-secret bindings ARE readable, which is what makes the secret's
/// absence from that list a decision rather than an oversight.
///
/// Endpoint, dialect and model are configuration and belong in a diagnostic;
/// the credential is not, and has no byte accessor at all. Asserting the
/// readable three here pins that asymmetry: if a `key()` accessor is ever added
/// beside these, the difference in treatment stops being visible.
#[test]
fn the_configured_bindings_are_readable_and_the_credential_is_not() {
    let adapter = an_adapter();

    assert_eq!(adapter.base_url(), "https://api.example.test/v1");
    assert_eq!(adapter.dialect(), EmbeddingDialect::OpenAiCompatible);
    assert_eq!(adapter.model(), "text-embedding-3-small");

    // The only path to bytes is the greppable escape, through the newtype.
    assert_eq!(adapter.secret().expose(), PLANTED_KEY);
}

/// The day-one dialect set is closed, and this test only compiles if it is.
///
/// The `match` below has no wildcard arm. Rust requires one when matching a
/// `#[non_exhaustive]` enum **from another crate**, and an integration test is
/// another crate -- so marking [`EmbeddingDialect`] `#[non_exhaustive]` breaks
/// this file at compile time. That is the guard: S2.19 locked the set by
/// decision, and a wildcard arm downstream is exactly where a sixth dialect
/// would later be swallowed without anyone noticing.
#[test]
fn the_dialect_set_is_closed_to_other_crates() {
    fn is_openai_shaped(dialect: EmbeddingDialect) -> bool {
        match dialect {
            EmbeddingDialect::OpenAiCompatible => true,
            EmbeddingDialect::Cohere
            | EmbeddingDialect::Gemini
            | EmbeddingDialect::Voyage
            | EmbeddingDialect::Bedrock => false,
        }
    }

    assert!(is_openai_shaped(EmbeddingDialect::OpenAiCompatible));
    assert!(!is_openai_shaped(EmbeddingDialect::Bedrock));
}

/// `ALL` lists exactly the five day-one dialects, once each.
///
/// The compile-time check above proves the enum has no hidden sixth variant; it
/// cannot prove that `ALL` enumerates the five faithfully. A copy-paste
/// duplicate there would silently shrink any set built from it -- a factory
/// table, a config-value list, a metrics cardinality -- while still having
/// length five and still compiling. Both halves are asserted.
#[test]
fn all_enumerates_the_five_day_one_dialects_once_each() {
    assert_eq!(
        EmbeddingDialect::ALL.len(),
        5,
        "S2.19's day-one set is five dialects: {:?}",
        EmbeddingDialect::ALL
    );

    let distinct: HashSet<EmbeddingDialect> = EmbeddingDialect::ALL.into_iter().collect();
    assert_eq!(
        distinct.len(),
        EmbeddingDialect::ALL.len(),
        "a dialect is listed twice in ALL: {:?}",
        EmbeddingDialect::ALL
    );
}

/// The adapter can be shared across tasks and surfaces.
///
/// ARCH §4.2 runs this adapter in-process on ANY surface, and S1 uses the same
/// instance for ingest embedding under the slow-burn scheduler -- so it will be
/// held behind an `Arc` and touched from several tasks. `Send + Sync` is a
/// property of the FIELDS, and the one at risk is the credential: swapping
/// `Secret` for a cell, a handle, or anything with interior mutability would
/// remove it here, at construction time, rather than at the first await.
#[test]
fn the_adapter_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<HttpEmbedder>();
    assert_send_sync::<EmbeddingDialect>();
}
