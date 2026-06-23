//! DDL for the `store_meta` table and the cross-tenant BEFORE INSERT trigger on
//! `blobs` (arch §5.2, AC-F3.1, AC-F3.4 / DATA-01).
//!
//! File: `wqm-storage-write/src/schema/store_meta.rs`
//! Location: `src/rust/storage-write/src/schema/` (write-crate schema submodule)
//! Context: `store_meta` is a single-row table that binds a `store.db` file to its
//!   owning tenant. Its DDL is F3-owned (table existence); its single row is populated
//!   once at store creation by F13/registration — F3 does NOT insert the row.
//!
//!   The `blobs_bi` BEFORE INSERT trigger (AC-F3.4) reads `store_meta.tenant_id` and
//!   raises ABORT if `NEW.tenant_id` does not match. This is a non-vacuous structural
//!   backstop: the trigger compares `NEW.tenant_id` against `store_meta` (a separate
//!   source of truth), NOT against `NEW.tenant_id` itself (a reflexive tautology that
//!   would always pass and be useless — SEC-N02 / DATA-NIT-01 prohibition).
//!
//!   Defense-in-depth for cross-tenant isolation (AC-F3.4):
//!     Layer 1 — write-facade assertion in Rust before the INSERT.
//!     Layer 2 — this BEFORE INSERT trigger (structural backstop, enforced by SQLite
//!               even on direct SQL that bypasses the facade).
//!
//! Neighbors: [`super::blobs`] (trigger fires on `blobs`), [`super::mod`] (apply order
//!   requires `blobs` and `store_meta` to exist before this trigger is created).

/// DDL for the `store_meta` single-row owning-tenant binding table.
/// F3 owns this table's existence; F13/registration populates the single row.
pub const CREATE_STORE_META: &str = "CREATE TABLE store_meta (\n    tenant_id  TEXT NOT NULL\n)";

/// BEFORE INSERT trigger on `blobs` — cross-tenant isolation backstop (AC-F3.4).
///
/// Reads `store_meta.tenant_id` (the authoritative bound tenant for this store.db)
/// and raises ABORT if `NEW.tenant_id` does not match. Because `store_meta` is a
/// separate table — not the row being inserted — the comparison is non-vacuous.
///
/// NOTE: this trigger fires even on direct SQL that bypasses the write facade,
/// making it the structural guarantee of the defense-in-depth pairing.
pub const TRIGGER_BLOBS_BI: &str = r#"CREATE TRIGGER blobs_bi BEFORE INSERT ON blobs
WHEN NEW.tenant_id != (SELECT tenant_id FROM store_meta)
BEGIN
    SELECT RAISE(ABORT, 'tenant_id mismatch: blobs insert rejected by store_meta guard');
END"#;

/// All DDL statements for this module, in application order.
/// `store_meta` must precede the trigger; `blobs` (from `blobs.rs`) must also exist.
pub const STATEMENTS: &[&str] = &[CREATE_STORE_META, TRIGGER_BLOBS_BI];
