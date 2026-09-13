//! The editor engine under every text input: `modalkit` driven through ONE engine and TWO
//! keymaps, with the §3 caret drawn by this crate.
//!
//! Adopted 2026-09-13 21:57 (Chris: *"Yes A, go ahead and adopt it"*) on the evidence of
//! `PASS2-EVAL.md` §E-2. Ruling B (21:42) made the editor an operator grammar — `d`/`c`/`y` over
//! any motion or text object, counts on operators and motions, `t`/`T`/`f`/`F` with `;`/`,`, and
//! counts composing inside visual mode (`v2t,` is valid) — and `modalkit` is the candidate that
//! passes it without a fork. [`acceptance`] is the §E-2 probe as a test: it is what the pin in
//! `Cargo.toml` has to keep satisfying.

#[cfg(test)]
mod acceptance;
