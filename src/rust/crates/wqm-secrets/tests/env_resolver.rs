//! `P04-GT002-WO057` -- `C-adp-N26-env`.
//!
//! The row is vocabulary only, so these tests assert what a DECLARATION can be
//! held to: that the name exists at the documented path, that it is constructible
//! from OUTSIDE the crate (N28's wiring is not a member of `wqm-secrets`, and this
//! test crate stands in for that vantage point), that it is `Send + Sync` because
//! a resolver behind the port is shared across tasks, and that the crate still
//! declares no wqm dependency.
//!
//! The last one is an asserted ABSENCE, which is the pattern the landed rows use
//! for deliberate ones: the empty dependency table is a decision (nothing here
//! uses `wqm-common` yet), and a decision nobody checks is one that quietly
//! reverses when someone adds the edge "because §9.1 permits it".

use wqm_secrets::env::EnvResolver;

/// Compile-time proof of the bound; it fails to compile rather than to run, which
/// is the whole point of asserting it on a type that has no behavior to exercise.
const fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn env_resolver_is_constructible_from_outside_the_crate() {
    // A unit struct: the bare name IS the value, which is what lets N28's wiring
    // build one with no constructor to call and no builder to keep in step.
    let resolver = EnvResolver;

    // Zero-sized, and that is the statelessness stated as a measurement rather
    // than as prose: a field added later makes this fail and sends the reader back
    // to the carrier argument in the type's doc.
    assert_eq!(
        std::mem::size_of_val(&resolver),
        0,
        "the adapter holds no state -- it reads the environment at call time"
    );

    // Two independently built values compare equal because there is nothing for
    // them to differ in.
    assert_eq!(resolver, EnvResolver, "a stateless adapter has one inhabitant");
}

#[test]
fn env_resolver_is_send_and_sync() {
    // A resolver is installed once and consulted from many tasks; a carrier that
    // was not shareable would force a lock around a value that holds nothing.
    assert_send_sync::<EnvResolver>();
}

#[test]
fn env_resolver_is_debug_and_redaction_safe_by_construction() {
    // `{:?}` on the adapter must not be a place a credential could appear. It
    // cannot be today -- the struct holds nothing -- and this pins that: a field
    // added later changes this output and brings the reader back to the
    // no-plaintext-lingering invariant before it can leak through a log line.
    assert_eq!(format!("{:?}", EnvResolver), "EnvResolver");
}

#[test]
fn the_crate_declares_no_wqm_dependency() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));

    let mut in_deps = false;
    let mut found: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            // Every dependency table, not just `[dependencies]`: a dev- or
            // build-dependency on a wqm crate is an edge too, and §9.1's closure
            // claim does not exempt them.
            in_deps = line.ends_with("dependencies]");
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name = line.split(['=', ' ']).next().unwrap_or_default();
        if name.starts_with("wqm-") || name.starts_with("wqm_") {
            found.push(name.to_string());
        }
    }

    assert!(
        found.is_empty(),
        "wqm-secrets declares no wqm dependency until a row needs one \
         (`Secret`/`SecretId` arrive with the port at P04-GT002-WO068); found {found:?}"
    );
}
