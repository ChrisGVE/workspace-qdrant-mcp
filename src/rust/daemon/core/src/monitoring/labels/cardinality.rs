//! Bounded-cardinality label helper (PRD A1).
//!
//! `language` and `file_type` are otherwise-unbounded metric labels: emitting
//! the raw value would create one Prometheus series per distinct value and
//! explode cardinality. This module maps a raw value to either itself (when it
//! is in the bounded allow-set) or the sentinel [`OTHER`].
//!
//! The allow-set is seeded from the bundled Language Registry
//! (`src/rust/common/src/language_registry/language_registry.yaml`) and is
//! ordered by rough real-world prevalence (most common first). A configurable
//! cap `N` (default [`DEFAULT_LABEL_CARDINALITY_CAP`], env
//! `WQM_LABEL_CARDINALITY_CAP`) is enforced as a top-`N` truncation over that
//! ordered list, so at most `N + 1` distinct values (the `N` highest-prevalence
//! languages plus [`OTHER`]) are ever emitted for a label (AC3). The tail
//! languages beyond `N` collapse to `OTHER`.
//!
//! Both lookups are deterministic and allocate nothing on the hot path for
//! known values (the returned strings are `&'static`).
//!
//! NOTE: [`BUNDLED_LANGUAGES`] and [`EXTENSION_TO_LANGUAGE`] mirror the
//! `language:` / `extensions:` fields of the registry YAML; keep them in sync
//! when the registry changes.

use std::path::Path;

use once_cell::sync::Lazy;

/// Sentinel emitted for any value outside the bounded allow-set.
pub const OTHER: &str = "other";

/// Default cap on distinct `language` / `file_type` label values (decision
/// §12 Q3). Overridable at runtime via `WQM_LABEL_CARDINALITY_CAP`.
pub const DEFAULT_LABEL_CARDINALITY_CAP: usize = 40;

/// Bundled language identifiers, ordered by rough real-world prevalence
/// (most common first). Truncating at the cap therefore drops the least common
/// languages to [`OTHER`]. Mirror of the registry `language:` fields.
pub const BUNDLED_LANGUAGES: &[&str] = &[
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "c",
    "cpp",
    "c-sharp",
    "php",
    "ruby",
    "swift",
    "kotlin",
    "scala",
    "dart",
    "lua",
    "perl",
    "r",
    "julia",
    "html",
    "css",
    "json",
    "yaml",
    "toml",
    "markdown",
    "sql",
    "bash",
    "haskell",
    "elixir",
    "erlang",
    "clojure",
    "ocaml",
    "scheme",
    "lisp",
    "fortran",
    "pascal",
    "ada",
    "nix",
    "tsx",
    "vue",
    // --- tail (dropped at the default cap of 40) ---
    "vala",
    "latex",
    "elm",
    "odin",
    "zig",
];

/// Lowercase, dot-stripped file extension -> bundled language identifier.
/// Mirror of the registry `extensions:` fields (deduplicated, lowercased).
const EXTENSION_TO_LANGUAGE: &[(&str, &str)] = &[
    ("adb", "ada"),
    ("ads", "ada"),
    ("sh", "bash"),
    ("bash", "bash"),
    ("zsh", "bash"),
    ("c", "c"),
    ("h", "c"),
    ("cs", "c-sharp"),
    ("clj", "clojure"),
    ("cljs", "clojure"),
    ("cljc", "clojure"),
    ("edn", "clojure"),
    ("cpp", "cpp"),
    ("cxx", "cpp"),
    ("cc", "cpp"),
    ("c++", "cpp"),
    ("hpp", "cpp"),
    ("hxx", "cpp"),
    ("hh", "cpp"),
    ("h++", "cpp"),
    ("ipp", "cpp"),
    ("tpp", "cpp"),
    ("css", "css"),
    ("dart", "dart"),
    ("ex", "elixir"),
    ("exs", "elixir"),
    ("elm", "elm"),
    ("erl", "erlang"),
    ("hrl", "erlang"),
    ("f", "fortran"),
    ("f90", "fortran"),
    ("f95", "fortran"),
    ("f03", "fortran"),
    ("f08", "fortran"),
    ("for", "fortran"),
    ("fpp", "fortran"),
    ("go", "go"),
    ("hs", "haskell"),
    ("lhs", "haskell"),
    ("html", "html"),
    ("htm", "html"),
    ("xhtml", "html"),
    ("java", "java"),
    ("js", "javascript"),
    ("mjs", "javascript"),
    ("cjs", "javascript"),
    ("jsx", "javascript"),
    ("json", "json"),
    ("jsonc", "json"),
    ("jl", "julia"),
    ("kt", "kotlin"),
    ("kts", "kotlin"),
    ("tex", "latex"),
    ("sty", "latex"),
    ("cls", "latex"),
    ("lisp", "lisp"),
    ("lsp", "lisp"),
    ("cl", "lisp"),
    ("fasl", "lisp"),
    ("lua", "lua"),
    ("md", "markdown"),
    ("markdown", "markdown"),
    ("mdx", "markdown"),
    ("nix", "nix"),
    ("ml", "ocaml"),
    ("mli", "ocaml"),
    ("mll", "ocaml"),
    ("mly", "ocaml"),
    ("odin", "odin"),
    ("pas", "pascal"),
    ("pp", "pascal"),
    ("dpr", "pascal"),
    ("dpk", "pascal"),
    ("lfm", "pascal"),
    ("pl", "perl"),
    ("pm", "perl"),
    ("pod", "perl"),
    ("t", "perl"),
    ("psgi", "perl"),
    ("php", "php"),
    ("phtml", "php"),
    ("php3", "php"),
    ("php4", "php"),
    ("php5", "php"),
    ("php7", "php"),
    ("phps", "php"),
    ("py", "python"),
    ("pyw", "python"),
    ("pyi", "python"),
    ("r", "r"),
    ("rmd", "r"),
    ("rnw", "r"),
    ("rb", "ruby"),
    ("rbw", "ruby"),
    ("rake", "ruby"),
    ("gemspec", "ruby"),
    ("rs", "rust"),
    ("scala", "scala"),
    ("sc", "scala"),
    ("sbt", "scala"),
    ("scm", "scheme"),
    ("ss", "scheme"),
    ("rkt", "scheme"),
    ("sql", "sql"),
    ("swift", "swift"),
    ("toml", "toml"),
    ("tsx", "tsx"),
    ("ts", "typescript"),
    ("mts", "typescript"),
    ("cts", "typescript"),
    ("vala", "vala"),
    ("vapi", "vala"),
    ("vue", "vue"),
    ("yaml", "yaml"),
    ("yml", "yaml"),
    ("zig", "zig"),
];

/// Effective cap: env override `WQM_LABEL_CARDINALITY_CAP` (clamped to the
/// number of bundled languages) or [`DEFAULT_LABEL_CARDINALITY_CAP`].
static CONFIGURED_CAP: Lazy<usize> = Lazy::new(|| {
    std::env::var("WQM_LABEL_CARDINALITY_CAP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_LABEL_CARDINALITY_CAP)
        .min(BUNDLED_LANGUAGES.len())
});

/// Return `value` verbatim (as a `&'static str`) when it is one of the top-`cap`
/// bundled languages, else [`OTHER`].
fn bounded_language_with_cap(value: &str, cap: usize) -> &'static str {
    match BUNDLED_LANGUAGES.iter().position(|&l| l == value) {
        Some(i) if i < cap => BUNDLED_LANGUAGES[i],
        _ => OTHER,
    }
}

/// Bound a `language` label value (AC1/AC2): in-allow-set -> verbatim,
/// otherwise [`OTHER`].
pub fn bounded_language(value: &str) -> &'static str {
    bounded_language_with_cap(value, *CONFIGURED_CAP)
}

fn bounded_file_type_with_cap(path: &Path, cap: usize) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => EXTENSION_TO_LANGUAGE
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(ext))
            .map(|(_, lang)| bounded_language_with_cap(lang, cap))
            .unwrap_or(OTHER),
        None => OTHER,
    }
}

/// Bound a `file_type` label derived from a path (AC4): the extension is matched
/// case-insensitively against the registry extension map and the resulting
/// language is bounded; unknown or missing extensions yield [`OTHER`].
pub fn bounded_file_type(path: &Path) -> &'static str {
    bounded_file_type_with_cap(path, *CONFIGURED_CAP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac1_in_allow_set_returns_verbatim() {
        let cap = DEFAULT_LABEL_CARDINALITY_CAP;
        assert_eq!(bounded_language_with_cap("python", cap), "python");
        assert_eq!(bounded_language_with_cap("rust", cap), "rust");
        assert_eq!(bounded_language_with_cap("c-sharp", cap), "c-sharp");
    }

    #[test]
    fn ac2_outside_allow_set_returns_other() {
        let cap = DEFAULT_LABEL_CARDINALITY_CAP;
        assert_eq!(bounded_language_with_cap("cobol", cap), OTHER);
        assert_eq!(bounded_language_with_cap("", cap), OTHER);
        // Tail language beyond the default cap collapses to OTHER.
        assert_eq!(bounded_language_with_cap("zig", cap), OTHER);
    }

    #[test]
    fn ac3_at_most_n_plus_one_distinct() {
        let n = DEFAULT_LABEL_CARDINALITY_CAP;
        let mut seen = std::collections::HashSet::new();
        // Feed 10*N deterministic pseudo-random values plus every bundled name.
        for i in 0..(10 * n) {
            let v = format!("lang_{}_{}", i % 97, (i * 31) % 13);
            seen.insert(bounded_language_with_cap(&v, n));
        }
        for &l in BUNDLED_LANGUAGES {
            seen.insert(bounded_language_with_cap(l, n));
        }
        assert!(
            seen.len() <= n + 1,
            "distinct label values {} exceeds N+1 ({})",
            seen.len(),
            n + 1
        );
        assert!(seen.contains(OTHER));
    }

    #[test]
    fn ac4_file_type_case_insensitive_and_unknown_other() {
        let cap = DEFAULT_LABEL_CARDINALITY_CAP;
        assert_eq!(
            bounded_file_type_with_cap(Path::new("a/b/main.PY"), cap),
            "python"
        );
        assert_eq!(
            bounded_file_type_with_cap(Path::new("Main.Rs"), cap),
            "rust"
        );
        assert_eq!(
            bounded_file_type_with_cap(Path::new("data.JSON"), cap),
            "json"
        );
        assert_eq!(
            bounded_file_type_with_cap(Path::new("notes.xyz"), cap),
            OTHER
        );
        assert_eq!(
            bounded_file_type_with_cap(Path::new("Makefile"), cap),
            OTHER
        );
    }

    #[test]
    fn extension_map_targets_are_known_languages() {
        // Every mapped language must exist in the bundled list.
        for (_, lang) in EXTENSION_TO_LANGUAGE {
            assert!(
                BUNDLED_LANGUAGES.contains(lang),
                "extension maps to unknown language {lang}"
            );
        }
    }
}
