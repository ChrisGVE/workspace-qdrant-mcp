//! wqm-common -- the infra floor of the wqm-0.2 workspace.
//!
//! This crate is the bottom of the dependency DAG (ARCH rev15 §3.3, §9.1): it has
//! no wqm-crate dependencies and every other crate may link it. It will own the
//! declarative registries and value objects that the whole system keys on -- N8
//! canonical names, N9 error taxonomy, N23 path canonicalization, N35 collection
//! profiles + field-family registry, N51 access-capability registry, N7 config,
//! N3 identity value objects + the `fts_key` de-dash transform, N40 types, N12
//! envelope types, the N44 source-access port trait, the `Secret` newtype,
//! and N33's `RulesWriteCap`.
//!
//! The crate grows per work order: `P04-GT001` declares it and seeds N8, and each
//! later nexus arrives with the slice that realizes it (`P04-GT006` for N8's full
//! surface, `P04-GT013` for N7, `P04-GT032` for N3, `P04-GT055` for N35,
//! `P04-GT025` for N51). `mesh/program.db` is the tracker; a slice GT is named
//! `slice-N##`.

// Each module carries its OWN documentation in its `//!` header. These
// declarations are deliberately bare: an outer `///` block on a `pub mod` line
// makes that module's inner intra-doc links resolve against the crate root
// instead of the module, so a link to Thing written inside the module fails to
// resolve even though Thing is declared right there. That cost this crate 13
// unresolved links and a red `cargo doc`; it is a rustdoc scoping behaviour, not
// a naming mistake, and the only fix is to leave the declaration undecorated.

pub mod names;

pub mod envelope;

pub mod plan;

pub mod profile;

pub mod secret;

pub mod rules_write_cap;
