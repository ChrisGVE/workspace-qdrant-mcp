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

## URL validation: WHATWG-exact (no divergence)

`validate_url` uses the `url` crate — the same WHATWG URL Standard that JS
`new URL()` implements — so it matches TS byte-for-byte on the accept/reject
decision AND the error message, including the boundary inputs that an earlier
manual parser got wrong. The corpus captures **every** input directly from TS
with **no exclusions**:

| input | TS `new URL` / Rust `url` crate (identical) |
|-------|---------------------------------------------|
| `http://` | "url is malformed (failed to parse)" |
| `mailto:a@b` | "url must use http:// or https:// (got mailto:)" |
| `http:/x` | OK (special-scheme single-slash → host `x`) |
| `http://a b.com` | "url is malformed (failed to parse)" |
| `http://...` | "url has invalid hostname (dots/whitespace only)" |

(The earlier manual `://`-split parser diverged on these; replacing it with the
`url` crate closed the gap — see commit history.)

## Provenance / regeneration

Captured by throwaway harnesses under gitignored `tmp/`:
`tmp/<ts>_capture_errors.mjs` writes the JSON, `tmp/<ts>_gen_errors.mjs`
codegens the `gen_err_*.rs` wrappers. Regenerate after a TS change, re-run
`cargo test -p mcp-server --test parity_corpus`, and delete the scripts before
committing.
