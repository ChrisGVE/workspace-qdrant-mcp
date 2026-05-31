//! TS↔Rust parity corpus suite.
//!
//! Drives Rust canonicalization / hashing functions with inputs captured from
//! the TypeScript MCP server and asserts byte-for-byte parity. The expected
//! values live in committed JSON under `tests/golden/parity/` (see that
//! directory's README for provenance). Per-case `#[test]` wrappers are
//! generated into the `gen_*.rs` files so each corpus row reports individually.
//!
//! Hermetic: no daemon / Qdrant / network. Runs in the default `cargo test`.

#[path = "parity/asserts.rs"]
mod asserts;

#[path = "parity/gen_stable_stringify.rs"]
mod gen_stable_stringify;

#[path = "parity/gen_stable_stringify_2.rs"]
mod gen_stable_stringify_2;

#[path = "parity/gen_idempotency.rs"]
mod gen_idempotency;

#[path = "parity/gen_parse_int.rs"]
mod gen_parse_int;

#[path = "parity/gen_expand_path.rs"]
mod gen_expand_path;

#[path = "parity/error_asserts.rs"]
mod error_asserts;

#[path = "parity/gen_err_rules_action.rs"]
mod gen_err_rules_action;

#[path = "parity/gen_err_unknown_tool.rs"]
mod gen_err_unknown_tool;

#[path = "parity/gen_err_url_validate.rs"]
mod gen_err_url_validate;

#[path = "parity/gen_err_store_inband.rs"]
mod gen_err_store_inband;

#[path = "parity/gen_err_scratchpad_inband.rs"]
mod gen_err_scratchpad_inband;

#[path = "parity/gen_err_grep_inband.rs"]
mod gen_err_grep_inband;
