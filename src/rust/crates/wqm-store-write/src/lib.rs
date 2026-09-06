//! wqm-store-write -- the S1-only mutation authority (ARCH rev15 §9.1).
//!
//! # What this crate is
//!
//! Stratum 3, and the write closure of the whole product: N6's compare-and-swap,
//! N11's destructive-operation gate, N19's schema evolution and migration, N2's
//! drain and N5's queue are all homed here. The rule the architecture draws is
//! that persistent state has exactly one mutator, and this crate is it.
//!
//! # Why it is S1-only, and why that is a link fact
//!
//! "Only the daemon writes" is worth nothing as a convention -- a client that can
//! *call* a write path will eventually call one. So the boundary is drawn in the
//! dependency graph instead: `ci/link-policy.toml` lists this crate under
//! `[s1_only].crates`, and the N14 link-closure guard default-denies it to every
//! bin except `memexd` and the declared offline `wqm-restore` maintenance binary.
//! A client bin therefore cannot link this crate at all, and the temptation is
//! removed rather than documented. §9.1 pins its own wqm-closure exactly as
//! `{wqm-search, wqm-store, wqm-proto, wqm-common}`, enforced by
//! `[dp7.exact_closures]`; the sibling `wqm-store` carries N41's read leg and
//! deliberately lacks `rebuild_from_sot` for the same reason.
//!
//! # Scope at `P04-GT002-WO047`
//!
//! This is the first row landed in the crate, and it is vocabulary only --
//! creating the crate is part of the work order (CHARTER §2.1/§7.3). It declares
//! [`versions::PerStoreVersions`], the in-memory report of all four store schema
//! versions that N19's `versions()` returns and `guard_boot` evaluates before the
//! process is allowed to serve. Vocabulary comes first because it is what every
//! later N19 row is written against: the reader, the comparison against the
//! binary, and the migration driver all program against this type.
//!
//! Nothing here reads a record, opens a store, or compares anything to a build
//! constant. Zero behavior means zero behavior.

pub mod versions;
