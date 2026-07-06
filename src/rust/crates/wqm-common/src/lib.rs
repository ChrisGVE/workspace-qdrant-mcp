//! wqm-common -- the infra floor of the wqm-0.2 workspace.
//!
//! This crate is the bottom of the dependency DAG (ARCH rev08 §3.3, §9.1): it has
//! no wqm-crate dependencies and every other crate may link it. It will own the
//! declarative registries and value objects that the whole system keys on -- N8
//! canonical names, N9 error taxonomy, N23 path canonicalization, N35 collection
//! profiles + field-family registry, N51 access-capability registry, N7 config,
//! N3 identity value objects + the `fts_key` de-dash transform, N40 types, N12
//! envelope types, the N44 source-access port trait, and the `Secret` newtype.
//!
//! Per the grow-per-phase model (PRD F-00), Phase 0 only declares the crate; the
//! contents arrive feature by feature starting at F-01 (N8 name registry).

/// N8 -- the canonical name registry: the single owner of the workspace's literal
/// identifier strings (collection names, environment keys, the N51
/// operation-class and consumer vocabularies). See [`names`] for the
/// single-producer rule and the grow-per-phase contract.
pub mod names;
