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
| `parse_int.json` | `config::parse_int_prefix` | native JS `parseInt(s, 10)` |
| `expand_path.json` | `config::expand_path_ts_with_home` | `config.ts` `expandPath` = Node `path.join(home, path.slice(1))` |

Each idempotency row asserts **two** things: the canonical `payload_json`
matches TS `stableStringify(payload)`, and the 32-hex key matches.

`parse_int.json` rows assert `parse_int_prefix(input) == parseInt(input, 10)`,
mapping JS `NaN` → Rust `None`. The corpus is restricted to JS results in the
safe-integer range (±2^53-1), which the Rust `Option<i64>` represents exactly;
larger values are out of the parser's operative domain (it parses a port) and
would test a JS double artifact, not integer-parse parity.

`expand_path.json` rows assert `expand_path_ts_with_home(input, FIXED_HOME)`
matches TS `expandPath`. TS uses Node `path.join`, which **normalizes** the
result (collapses `//`, resolves `.`/`..`, keeps one trailing slash); the Rust
side reimplements `path.posix.join` faithfully so the gnarly cases (`~/a/../b`,
`~//x`, `~/trailing/..`) stay byte-for-byte identical. A fixed home is used so
the Rust test does not depend on the real `$HOME`.

### Operative-path note (not a Rust bug)

`stableStringify` (the recursive canonicalizer in `queue-operations.ts`) is the
**operative** enqueue serializer — `queue-operations.ts` does
`payload_json: stableStringify(payload)`. A *separate* helper,
`queue-payload-builders.ts generateIdempotencyKey`, uses the replacer-array form
`JSON.stringify(payload, Object.keys(payload).sort())` which only sorts the
**top-level** keys (nested keys are not re-sorted), so it diverges from
`stableStringify` for nested payloads. That helper is **not** on the enqueue
path, so the corpus (and the Rust MCP server) correctly use the recursive form.
The replacer-array helper's non-recursive behavior is a latent TS inconsistency,
not a Rust parity defect.

## Provenance / regeneration

Captured by a throwaway harness (`tmp/<ts>_capture_parity.mjs`). It imports the
built TS `dist/` where a real export exists, and verbatim-copies
`stableStringify` (12 lines, not exported) citing its source line. To capture
real TS exports it needs the native addon:

```bash
cd src/rust/common-node && napi build --platform --release   # builds wqm-common-node.<plat>.node (gitignored)
node tmp/<ts>_capture_parity.mjs                              # writes the JSON corpora
```

The per-case Rust wrappers are codegen'd from the JSON (one `#[test]` per row)
by a second throwaway script (`tmp/<ts>_gen_parity.mjs`): it reads each
`<stem>.json` and writes `tests/parity/gen_<stem>.rs`. When the TS server
changes output for a covered case: rebuild `dist/`, re-run the capture harness,
re-run the codegen, and re-run `cargo test -p mcp-server --test parity_corpus`.
Delete both throwaway scripts before committing (they live under gitignored
`tmp/`).

## Why these are the right cases

The expected values come from TS, not from hand-authored guesses — a Rust
regression cannot silently "agree with itself". Inputs deliberately include the
gnarly variants: unsorted / unicode / numeric-string keys, nested objects,
arrays, escaping (quotes / backslash / control / surrogate pairs), null/bool/
number edges, and every `(item_type, op)` pair used by the queue.
