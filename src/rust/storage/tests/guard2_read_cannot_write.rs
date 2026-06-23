//! Guard 2 (AC-F1.5 / AC-F1.6) — compile-fail proof of the read/write boundary.
//!
//! The read crate (`wqm-storage`) must not be able to:
//!   1. call a `migrations::*` execution function (DDL lives in wqm-storage-write),
//!   2. call a `schema::*` execution function (same),
//!   3. reach a Qdrant-mutating method through `QdrantReadClient` (SEC-01).
//!
//! Each fixture under `tests/compile_fail/` MUST fail to compile; trybuild
//! compares the compiler output against the committed `.stderr` snapshot.

#[test]
fn read_crate_cannot_reach_write_surface() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
