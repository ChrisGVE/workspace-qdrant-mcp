//! wqm-common -- the infra floor of the wqm-0.2 workspace.
//!
//! This crate is the bottom of the dependency DAG (ARCH rev15 §3.3, §9.1): it has
//! no wqm-crate dependencies and every other crate may link it. It will own the
//! declarative registries and value objects that the whole system keys on -- N8
//! canonical names, N9 error taxonomy, N23 path canonicalization, N35 collection
//! profiles + field-family registry, N51 access-capability registry, N7 config,
//! N3 identity value objects + the `fts_key` de-dash transform, N40 types, N12
//! envelope types, the N44 source-access port trait, and the `Secret` newtype.
//!
//! The crate grows per work order: `P04-GT001` declares it and seeds N8, and each
//! later nexus arrives with the slice that realizes it (`P04-GT006` for N8's full
//! surface, `P04-GT013` for N7, `P04-GT032` for N3, `P04-GT055` for N35,
//! `P04-GT025` for N51). `mesh/program.db` is the tracker; a slice GT is named
//! `slice-N##`.

/// N8 -- the canonical name registry: the single owner of the workspace's literal
/// identifier strings (collection names, environment keys, the N51
/// operation-class and consumer vocabularies). See [`names`] for the
/// single-producer rule and the grow-per-phase contract.
pub mod names;

/// N12 -- the response envelope every MCP tool answers in, with §4.3's error
/// vocabulary and §4.4's notice vocabulary. Seeded at `P04-GT001-WO011` with the
/// sealed seven-key shape (MCP-SURFACE.md §3.1); the per-tool `data` shapes live
/// with the tools that produce them.
pub mod envelope;
