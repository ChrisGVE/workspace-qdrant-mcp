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
//! either -- the absence of an impl is not observable from a value. Both carry
//! the rustc error code they are required to fail with, because a bare
//! `compile_fail` passes on any error at all and would keep reading green once
//! it stopped failing for its own reason; each was verified against that code by
//! hand. Stable rustdoc records the code without checking it -- see the note on
//! the `Default` doctest.
//!
//! A third guarantee is not about the type but about the CRATE: the adapter's
//! doc calls itself the pure-Rust floor of the provider family and says the
//! acceptance test is that `wqm-search`'s default closure carries no
//! `ort`/`fastembed`/`onnxruntime`. `the_crate_declares_no_native_embedding_dependency`
//! is that test.
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

    // And element-wise, because `ALL`'s doc promises DECLARATION ORDER and
    // neither check above can see a reorder: swapping two entries leaves the
    // length at five and the set distinct. Order is load-bearing wherever `ALL`
    // is read positionally -- a numbered list offered to a user, a fixed-column
    // metrics table, an index into a parallel array of wire shapes -- and there
    // the swap silently re-labels two dialects instead of failing.
    assert_eq!(
        EmbeddingDialect::ALL,
        [
            EmbeddingDialect::OpenAiCompatible,
            EmbeddingDialect::Cohere,
            EmbeddingDialect::Gemini,
            EmbeddingDialect::Voyage,
            EmbeddingDialect::Bedrock,
        ],
        "ALL must list the day-one dialects in declaration order"
    );

    // Spelled again by index, so a failure names the position that moved rather
    // than printing two five-element arrays and leaving the reader to diff them.
    assert_eq!(EmbeddingDialect::ALL[0], EmbeddingDialect::OpenAiCompatible);
    assert_eq!(EmbeddingDialect::ALL[1], EmbeddingDialect::Cohere);
    assert_eq!(EmbeddingDialect::ALL[2], EmbeddingDialect::Gemini);
    assert_eq!(EmbeddingDialect::ALL[3], EmbeddingDialect::Voyage);
    assert_eq!(EmbeddingDialect::ALL[4], EmbeddingDialect::Bedrock);
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

/// The crate names `wqm-search` must not declare, from the adapter's own
/// acceptance bullet (`src/http_embedder.rs`, "What is ABSENT here"): this
/// provider is the pure-Rust FLOOR of the provider family, compiled into every
/// build with no native library and no feature gate, and it stops being that the
/// moment an ONNX runtime is linked in beside it.
///
/// Matching is hyphen-normalized, and a name counts as a hit when it equals one
/// of these or begins with one followed by `-`. That catches the `-sys`
/// companions a native crate actually arrives as (`ort-sys`, `onnxruntime-sys`)
/// without flagging an unrelated crate that merely contains the letters --
/// a bare substring test on `ort` would condemn `portable-atomic`.
const NATIVE_EMBEDDING_CRATES: [&str; 3] = ["ort", "fastembed", "onnxruntime"];

/// What a `[...]` line in a manifest opens.
#[derive(Debug, PartialEq, Eq)]
enum Section {
    /// A dependency table. `named` is `Some` for the sub-table spelling
    /// `[dependencies.ort]`, where the header itself is the dependency.
    Dependencies { named: Option<String> },
    /// Anything else -- `[package]`, `[lints]`, `[features]`.
    Other,
}

/// Classify a manifest table header.
///
/// Cargo spells dependency tables four ways and all four are in scope, because
/// the acceptance bullet is about what the crate DECLARES, not about which
/// profile declares it:
///
/// - `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`;
/// - target-qualified, `[target.'cfg(unix)'.dependencies]`;
/// - the sub-table form `[dependencies.ort]`;
/// - and the two composed, `[target.'cfg(unix)'.dev-dependencies.ort]`.
///
/// The first two end in `dependencies`; the last two end in `.<name>` on a
/// table that does.
fn classify_header(line: &str) -> Section {
    let inner = line.trim().trim_start_matches('[').trim_end_matches(']');

    if inner.ends_with("dependencies") {
        return Section::Dependencies { named: None };
    }
    if let Some((table, name)) = inner.rsplit_once('.') {
        if table.ends_with("dependencies") {
            return Section::Dependencies {
                named: Some(name.trim_matches('"').trim().to_owned()),
            };
        }
    }
    Section::Other
}

/// Every crate name a manifest declares as a dependency, paired with the line it
/// was read from so a failure can point at it.
///
/// Two spellings name a crate and both are collected, because one of them hides
/// the other: the table key (`ort = "2"`), and the `package = "..."` value of a
/// renamed dependency (`embedder = { package = "ort" }`), where the key says
/// nothing at all. A scan that reads only keys is blind to precisely the edit
/// someone makes when they know the key is being watched.
fn declared_dependencies(manifest: &str) -> Vec<(String, String)> {
    let mut declared = Vec::new();
    let mut in_dependencies = false;

    for raw in manifest.lines() {
        let line = raw.trim();

        if line.starts_with('[') {
            match classify_header(line) {
                Section::Dependencies { named } => {
                    in_dependencies = true;
                    if let Some(name) = named {
                        declared.push((name, raw.to_owned()));
                    }
                }
                Section::Other => in_dependencies = false,
            }
            continue;
        }
        if !in_dependencies || line.is_empty() || line.starts_with('#') {
            continue;
        }

        // The table key, up to the first `=` or space.
        let key = line.split(['=', ' ']).next().unwrap_or_default().trim();
        if !key.is_empty() {
            declared.push((key.to_owned(), raw.to_owned()));
        }

        // And the rename target, wherever `package = "..."` sits on the line.
        if let Some(rest) = line.split_once("package").map(|(_, rest)| rest) {
            let value = rest
                .trim_start()
                .strip_prefix('=')
                .map(str::trim_start)
                .and_then(|v| v.strip_prefix('"'))
                .and_then(|v| v.split('"').next());
            if let Some(renamed) = value {
                declared.push((renamed.to_owned(), raw.to_owned()));
            }
        }
    }

    declared
}

/// The forbidden names a manifest declares, empty when it is clean.
fn native_embedding_dependencies(manifest: &str) -> Vec<(String, String)> {
    declared_dependencies(manifest)
        .into_iter()
        .filter(|(name, _)| {
            let normalized = name.replace('_', "-");
            NATIVE_EMBEDDING_CRATES.iter().any(|forbidden| {
                normalized == *forbidden || normalized.starts_with(&format!("{forbidden}-"))
            })
        })
        .collect()
}

/// The acceptance bullet, asserted: `wqm-search` declares no ONNX runtime.
///
/// # Why the MANIFEST and not `cargo tree`
///
/// This reads `Cargo.toml` and stops there, so a forbidden crate arriving
/// TRANSITIVELY -- pulled in by some future dependency of a dependency -- is out
/// of this test's reach, and deliberately so. The manifest is the surface this
/// crate DECLARES and the only one it controls: a transitive arrival is a fact
/// about someone else's dependency list, it changes without any edit here, and
/// pinning it would make this test fail on a lockfile bump that nobody in this
/// crate authored. The bullet it defends is likewise about a declaration ("the
/// HTTP client dependency ... is not added at this work order"), and the edit
/// that would break it is an edit to this file. Closing the transitive gap needs
/// a workspace-level closure check over a resolved graph, which is a different
/// instrument with a different owner.
#[test]
fn the_crate_declares_no_native_embedding_dependency() {
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));

    let found = native_embedding_dependencies(&manifest);

    assert!(
        found.is_empty(),
        "wqm-search is the pure-Rust floor of the provider family and must link \
         no ONNX runtime ({NATIVE_EMBEDDING_CRATES:?}); found {found:?}"
    );
}

/// The scanner above is only evidence if it can come out positive.
///
/// An absence test whose detector never fires is indistinguishable from a
/// detector that cannot fire, so the instrument is validated here against
/// hand-written positives -- one per spelling it claims to cover -- and against
/// the negatives that a looser match would wrongly condemn.
#[test]
fn the_native_dependency_scan_catches_every_spelling_it_claims_to() {
    let positives = [
        ("plain", "[dependencies]\nort = \"2\"\n"),
        ("dev table", "[dev-dependencies]\nfastembed = \"4\"\n"),
        (
            "build table",
            "[build-dependencies]\nonnxruntime = \"0.0.14\"\n",
        ),
        (
            "target-qualified",
            "[target.'cfg(unix)'.dependencies]\nort = { version = \"2\" }\n",
        ),
        ("sub-table", "[dependencies.ort]\nversion = \"2\"\n"),
        (
            "renamed, key says nothing",
            "[dependencies]\nembedder = { package = \"ort\", version = \"2\" }\n",
        ),
        (
            "renamed in a sub-table",
            "[dependencies.embedder]\npackage = \"fastembed\"\n",
        ),
        (
            "the -sys companion",
            "[dependencies]\nonnxruntime-sys = \"0.0.14\"\n",
        ),
        ("underscore spelling", "[dependencies]\nort_sys = \"2\"\n"),
    ];
    for (label, manifest) in positives {
        assert!(
            !native_embedding_dependencies(manifest).is_empty(),
            "the scan missed a forbidden dependency spelled as `{label}`:\n{manifest}"
        );
    }

    let negatives = [
        ("the real manifest is clean", include_str!("../Cargo.toml")),
        (
            "a crate that merely contains the letters",
            "[dependencies]\nportable-atomic = \"1\"\n",
        ),
        ("another one", "[dependencies]\nsort-by = \"1\"\n"),
        (
            "the name outside a dependency table",
            "[features]\nort = []\n",
        ),
        ("a commented-out edge", "[dependencies]\n# ort = \"2\"\n"),
    ];
    for (label, manifest) in negatives {
        assert!(
            native_embedding_dependencies(manifest).is_empty(),
            "the scan wrongly condemned `{label}`: {:?}",
            native_embedding_dependencies(manifest)
        );
    }
}
