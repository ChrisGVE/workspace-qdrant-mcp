# TS↔Rust parity corpus

Committed JSON corpora whose **expected** values are captured from the
TypeScript MCP server (or its verbatim canonicalizer). The Rust parity suite
(`tests/parity_corpus.rs`) drives the equivalent Rust functions with the same
inputs and asserts byte-for-byte parity. Per-case `#[test]` wrappers are
generated into `tests/parity/gen_*.rs` so every row reports individually under
`cargo test`.

This corpus is the regression guard for the cross-language invariants — the
documented contract that the TypeScript, Rust daemon, and Rust CLI all produce
identical canonical output (see CLAUDE.md "Idempotency key" invariant).

## Corpora

| File | Rust function under test | TS source of truth |
|------|--------------------------|--------------------|
| `stable_stringify.json` | `canonicalize::stable_stringify::stable_stringify` | `clients/queue-operations.ts` `stableStringify` (recursive key sort, F-008) |
| `idempotency.json` | `stable_stringify` + `wqm_common::hashing::generate_idempotency_key` | daemon contract: `sha256("{it}\|{op}\|{ten}\|{col}\|{payload_json}")[:32]` |

Each idempotency row asserts **two** things: the canonical `payload_json`
matches TS `stableStringify(payload)`, and the 32-hex key matches.

## Provenance / regeneration

Captured by a throwaway harness (`tmp/<ts>_capture_parity.mjs`). It imports the
built TS `dist/` where a real export exists, and verbatim-copies
`stableStringify` (12 lines, not exported) citing its source line. To capture
real TS exports it needs the native addon:

```bash
cd src/rust/common-node && napi build --platform --release   # builds wqm-common-node.<plat>.node (gitignored)
node tmp/<ts>_capture_parity.mjs                              # writes the JSON corpora
```

The per-case Rust wrappers are codegen'd from the JSON (one `#[test]` per row).
When the TS server changes output for a covered case: rebuild `dist/`, re-run
the harness, regenerate the wrappers, and re-run `cargo test -p mcp-server
--test parity_corpus`. Delete the harness before committing.

## Why these are the right cases

The expected values come from TS, not from hand-authored guesses — a Rust
regression cannot silently "agree with itself". Inputs deliberately include the
gnarly variants: unsorted / unicode / numeric-string keys, nested objects,
arrays, escaping (quotes / backslash / control / surrogate pairs), null/bool/
number edges, and every `(item_type, op)` pair used by the queue.
