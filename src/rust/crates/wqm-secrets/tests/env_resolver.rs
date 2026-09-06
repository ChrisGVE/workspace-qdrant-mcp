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
//!
//! An absence test is worth only as much as its detector, and this one runs
//! against a manifest with no dependency table at all -- so it would read green
//! with the scan gutted. `the_wqm_dependency_scan_catches_every_spelling_it_claims_to`
//! is the positive control that keeps the green meaningful.

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

/// Every wqm crate a manifest declares as a dependency, empty when it declares
/// none.
///
/// Two spellings name a crate and the scan reads both, because the second hides
/// the first. A dependency's table key is normally its crate name
/// (`wqm-common = { path = ... }`), but Cargo's rename form puts the real name
/// in a `package` value and lets the key say anything at all --
/// `common = { package = "wqm-common", path = "../wqm-common" }` is the same
/// edge under a key a name test cannot see. That is not a hypothetical: it is
/// the ordinary way to shorten a prefixed crate name, and it is exactly the edit
/// someone makes when the key is what is being watched.
fn wqm_dependencies(manifest: &str) -> Vec<String> {
    let mut in_deps = false;
    let mut found: Vec<String> = Vec::new();

    for line in manifest.lines() {
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

        // The table key.
        let name = line.split(['=', ' ']).next().unwrap_or_default();
        if is_wqm_crate(name) {
            found.push(name.to_string());
        }

        // The rename target, wherever `package = "..."` sits on the line.
        if let Some(renamed) = renamed_package(line) {
            if is_wqm_crate(renamed) {
                found.push(renamed.to_string());
            }
        }
    }

    found
}

/// Both spellings of the workspace prefix -- Cargo accepts either separator and
/// normalizes them to the same crate.
fn is_wqm_crate(name: &str) -> bool {
    name.starts_with("wqm-") || name.starts_with("wqm_")
}

/// The `package = "..."` value on one manifest line, when there is one.
fn renamed_package(line: &str) -> Option<&str> {
    line.split_once("package")?
        .1
        .trim_start()
        .strip_prefix('=')
        .map(str::trim_start)?
        .strip_prefix('"')?
        .split('"')
        .next()
}

#[test]
fn the_crate_declares_no_wqm_dependency() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));

    let found = wqm_dependencies(&text);

    assert!(
        found.is_empty(),
        "wqm-secrets declares no wqm dependency until a row needs one \
         (`Secret`/`SecretId` arrive with the port at P04-GT002-WO068); found {found:?}"
    );
}

/// The scan above is only evidence if it can come out positive.
///
/// The manifest it reads is empty of dependencies today, so the assertion passes
/// whether the scan works or not -- an absence test with an inert detector reads
/// exactly like a clean result. The instrument is therefore exercised here on
/// hand-written manifests: one per spelling it claims to cover, and the
/// negatives a looser match would wrongly condemn.
#[test]
fn the_wqm_dependency_scan_catches_every_spelling_it_claims_to() {
    let positives = [
        (
            "plain",
            "[dependencies]\nwqm-common = { path = \"../wqm-common\" }\n",
        ),
        ("underscore", "[dependencies]\nwqm_common = \"0.2\"\n"),
        (
            "dev table",
            "[dev-dependencies]\nwqm-test-harness = { path = \"..\" }\n",
        ),
        (
            "build table",
            "[build-dependencies]\nwqm-common = \"0.2\"\n",
        ),
        (
            "target-qualified",
            "[target.'cfg(unix)'.dependencies]\nwqm-common = \"0.2\"\n",
        ),
        (
            "renamed, key says nothing",
            "[dependencies]\ncommon = { package = \"wqm-common\", path = \"../wqm-common\" }\n",
        ),
        (
            "renamed with the underscore spelling",
            "[dependencies]\ncommon = { package = \"wqm_common\" }\n",
        ),
    ];
    for (label, manifest) in positives {
        assert!(
            !wqm_dependencies(manifest).is_empty(),
            "the scan missed a wqm edge spelled as `{label}`:\n{manifest}"
        );
    }

    let negatives = [
        ("the real manifest is clean", include_str!("../Cargo.toml")),
        ("a non-wqm crate", "[dependencies]\nzeroize = \"1\"\n"),
        (
            "a non-wqm rename",
            "[dependencies]\ncommon = { package = \"anyhow\" }\n",
        ),
        (
            "the name outside a dependency table",
            "[package]\nname = \"wqm-secrets\"\n",
        ),
        (
            "a commented-out edge",
            "[dependencies]\n# wqm-common = \"0.2\"\n",
        ),
    ];
    for (label, manifest) in negatives {
        assert!(
            wqm_dependencies(manifest).is_empty(),
            "the scan wrongly condemned `{label}`: {:?}",
            wqm_dependencies(manifest)
        );
    }
}
