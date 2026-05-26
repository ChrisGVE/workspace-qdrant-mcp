# Cross-language path fixtures

`cases.json` is the single source of truth for path-normalization
behavior. Three implementations consume it:

| Language | Runner | Status |
|---|---|---|
| Rust  | `src/rust/common/tests/cross_lang_path_fixtures.rs` | ✅ implemented |
| sh    | `tests/path-fixtures/run-sh.sh`                     | ✅ implemented |
| TS    | (TBD — wire into vitest)                            | ⏳ follow-up |
| PS    | (TBD — wire into pester or inline)                  | ⏳ follow-up |

## Sections

- `normalize` — positive cases. Each `{input, expected}` pair: feed
  `input` to the normalizer and assert the canonical form matches
  `expected` exactly.
- `normalize_errors` — negative cases. Each `{input, expectedKind}` pair:
  feeding `input` must throw/return an error of the named kind.
  Kinds: `empty`, `relative`, `dot-dot`, `nul-byte`, `non-utf8`.
- `is_windows_absolute`, `is_absolute` — boolean predicate cases for the
  shared "is absolute path" helpers.
- `normalize_slashes` — the `C:\foo` → `C:/foo` transform used as a
  building block by the normalizer.

## Adding cases

Add a case whenever you fix a divergence between implementations or
discover a new edge. Keep the structure flat — one case per object,
each with a unique `name`. Re-run every runner before committing.

## Embedded NUL

NUL bytes can't be carried in JSON. Each implementation asserts NUL
rejection in its own test suite (see `comment_nul` in `cases.json`).
