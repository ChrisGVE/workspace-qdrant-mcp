# TS↔Rust error/validation parity matrix

Committed JSON corpora whose **expected** values are captured from the
TypeScript MCP server (real `dist/` exports or verbatim-copied non-exported
validators). The Rust error-parity suite drives the **real** Rust
functions/handlers and asserts byte-for-byte parity with the TS-sourced
message. Per-case `#[test]` / `#[tokio::test]` wrappers are generated into
`tests/parity/gen_err_*.rs` (one per corpus row) and run inside the
`parity_corpus` binary via `tests/parity/error_asserts.rs`.

All drivers are hermetic — no daemon / Qdrant / network. The validation
branches under test return before any I/O, so the always-failing stub daemon is
never actually invoked on an error row.

## Corpora

| File | Rust function/handler under test | TS source of truth |
|------|----------------------------------|--------------------|
| `rules_action.json` | `RulesInput::from_args` (`js_coerce_action`) | `tool-builders/rules.ts buildRuleOptions` throw `Invalid rules action: ${action}` |
| `unknown_tool.json` | `tools::envelope::unknown_tool` | `tool-dispatcher.ts` default branch `Unknown tool: ${toolName}` |
| `url_validate.json` | `store_tool` type=url → `validate_url` | `store-handlers.ts validateUrlInput` |
| `store_inband.json` | `store_tool` library/doc path | `tools/store.ts StoreTool.store` + `resolveTenant` |
| `scratchpad_inband.json` | `store_tool` type=scratchpad | `store-handlers.ts storeScratchpad` |
| `grep_inband.json` | `grep_tool` empty-pattern branch | `tools/grep.ts grepError('Search pattern is required', 0)` |

`rules_action` covers the JS template-literal coercion of a non-enum `action`:
absent → "undefined", null → "null", number/bool → JS string form, array →
element `String()`s joined by "," (null elements → ""), object →
"[object Object]". The Rust `js_coerce_action` was fixed to mirror this exactly
(it previously emitted serde-JSON reprs like `["add"]` / `{}`).

## Known divergence (documented, not asserted)

`validate_url` in the Rust port uses a **manual** (non-WHATWG) URL parser to
avoid a `url`-crate dependency, while TS uses `new URL()`. They agree on the
operative inputs (clean http/https URLs, empty/whitespace, clearly non-http
schemes written with `://`). They diverge on WHATWG-boundary inputs, e.g.:

| input | TS `new URL` | Rust manual parser |
|-------|--------------|--------------------|
| `http://` | malformed | "url has empty hostname" |
| `mailto:a@b` | "must use http(s) (got mailto:)" | malformed (no `://`) |
| `http:/x` | OK (host `x`) | malformed (no `://`) |
| `http://a b.com` | malformed | OK |

These rows are **excluded** from `url_validate.json` (the harness recomputes the
Rust parser inline and drops any input where the two disagree) rather than
silently "agreeing with Rust". This is a real, pre-existing parser limitation,
not introduced here — flagged for follow-up if WHATWG-exact URL validation is
required.

## Provenance / regeneration

Captured by throwaway harnesses under gitignored `tmp/`:
`tmp/<ts>_capture_errors.mjs` writes the JSON, `tmp/<ts>_gen_errors.mjs`
codegens the `gen_err_*.rs` wrappers. Regenerate after a TS change, re-run
`cargo test -p mcp-server --test parity_corpus`, and delete the scripts before
committing.
