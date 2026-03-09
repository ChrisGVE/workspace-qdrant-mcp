//! Registry of known tree-sitter grammar sources.
//!
//! Maps language names to their GitHub repository and build metadata.
//! Grammar source tarballs are downloaded from GitHub releases when available,
//! or from the default branch archive as a fallback.

/// Information about a tree-sitter grammar source.
#[derive(Debug, Clone)]
pub struct GrammarSource {
    /// GitHub owner/org (e.g., "tree-sitter")
    pub owner: &'static str,
    /// GitHub repository name (e.g., "tree-sitter-rust")
    pub repo: &'static str,
    /// The symbol name suffix exported by the grammar.
    /// Usually matches the language name, but some grammars differ
    /// (e.g., "commonlisp" grammar exports `tree_sitter_commonlisp`).
    /// If None, derived from the language name.
    pub symbol_name: Option<&'static str>,
    /// Whether the grammar includes a C++ scanner (scanner.cc).
    /// Requires a C++ compiler instead of just a C compiler.
    pub has_cpp_scanner: bool,
    /// Subdirectory within the repo that contains the grammar src/.
    /// Most grammars have src/ at the root. Some monorepos (like
    /// tree-sitter-typescript) have it under a subdirectory.
    pub src_subdir: Option<&'static str>,
    /// Non-standard branch to use for archive fallback.
    /// Defaults to trying "main" then "master". Some repos keep
    /// generated parser.c only on a "release" branch.
    pub archive_branch: Option<&'static str>,
}

impl GrammarSource {
    const fn new(owner: &'static str, repo: &'static str) -> Self {
        Self {
            owner,
            repo,
            symbol_name: None,
            has_cpp_scanner: false,
            src_subdir: None,
            archive_branch: None,
        }
    }

    const fn with_cpp_scanner(mut self) -> Self {
        self.has_cpp_scanner = true;
        self
    }

    const fn with_symbol(mut self, symbol: &'static str) -> Self {
        self.symbol_name = Some(symbol);
        self
    }

    const fn with_subdir(mut self, subdir: &'static str) -> Self {
        self.src_subdir = Some(subdir);
        self
    }

    const fn with_branch(mut self, branch: &'static str) -> Self {
        self.archive_branch = Some(branch);
        self
    }

    /// Get the C symbol name for this grammar.
    /// Returns the explicit symbol_name if set, otherwise derives from repo name.
    pub fn c_symbol_name(&self, language: &str) -> String {
        if let Some(sym) = self.symbol_name {
            format!("tree_sitter_{sym}")
        } else {
            format!("tree_sitter_{}", language.replace('-', "_"))
        }
    }

    /// Get the GitHub release tarball URL for a specific tag.
    pub fn release_tarball_url(&self, tag: &str) -> String {
        format!(
            "https://github.com/{}/{}/releases/download/{}/{}.tar.gz",
            self.owner, self.repo, tag, self.repo
        )
    }

    /// Get the GitHub archive URL for a branch/tag (fallback when no releases exist).
    pub fn archive_tarball_url(&self, git_ref: &str) -> String {
        format!(
            "https://github.com/{}/{}/archive/refs/heads/{}.tar.gz",
            self.owner, self.repo, git_ref
        )
    }
}

/// Look up the grammar source for a language.
///
/// Returns `None` for unknown languages — the caller should fall back
/// to guessing `tree-sitter/tree-sitter-{language}`.
pub fn lookup(language: &str) -> Option<GrammarSource> {
    // Normalize language name
    let lang = match language {
        "shell" | "sh" | "zsh" => "bash",
        "commonlisp" | "common_lisp" | "common-lisp" => "lisp",
        "c_sharp" | "csharp" => "c-sharp",
        "objective-c" | "objc" => "objc",
        "objectpascal" => "pascal",
        other => other,
    };

    let source = match lang {
        "ada" => GrammarSource::new("briot", "tree-sitter-ada"),
        "bash" => GrammarSource::new("tree-sitter", "tree-sitter-bash"),
        "c" => GrammarSource::new("tree-sitter", "tree-sitter-c"),
        "c-sharp" => GrammarSource::new("tree-sitter", "tree-sitter-c-sharp")
            .with_symbol("c_sharp")
            .with_cpp_scanner(),
        "clojure" => GrammarSource::new("sogaiu", "tree-sitter-clojure"),
        "cpp" => GrammarSource::new("tree-sitter", "tree-sitter-cpp").with_cpp_scanner(),
        "css" => GrammarSource::new("tree-sitter", "tree-sitter-css"),
        "dart" => GrammarSource::new("UserNobworthy", "tree-sitter-dart").with_cpp_scanner(),
        "elixir" => GrammarSource::new("elixir-lang", "tree-sitter-elixir").with_cpp_scanner(),
        "elm" => GrammarSource::new("elm-tooling", "tree-sitter-elm"),
        "erlang" => GrammarSource::new("WhatsApp", "tree-sitter-erlang"),
        "fortran" => GrammarSource::new("stadelmanma", "tree-sitter-fortran"),
        "go" => GrammarSource::new("tree-sitter", "tree-sitter-go"),
        "haskell" => GrammarSource::new("tree-sitter", "tree-sitter-haskell").with_cpp_scanner(),
        "html" => GrammarSource::new("tree-sitter", "tree-sitter-html").with_cpp_scanner(),
        "java" => GrammarSource::new("tree-sitter", "tree-sitter-java"),
        "javascript" => GrammarSource::new("tree-sitter", "tree-sitter-javascript"),
        "json" => GrammarSource::new("tree-sitter", "tree-sitter-json"),
        "julia" => GrammarSource::new("tree-sitter", "tree-sitter-julia").with_cpp_scanner(),
        "kotlin" => GrammarSource::new("fwcd", "tree-sitter-kotlin"),
        "latex" => GrammarSource::new("latex-lsp", "tree-sitter-latex"),
        "lisp" => {
            GrammarSource::new("theHamsta", "tree-sitter-commonlisp").with_symbol("commonlisp")
        }
        "lua" => GrammarSource::new("tree-sitter-grammars", "tree-sitter-lua"),
        "markdown" => GrammarSource::new("tree-sitter-grammars", "tree-sitter-markdown")
            .with_subdir("tree-sitter-markdown"),
        "nix" => GrammarSource::new("nix-community", "tree-sitter-nix"),
        "ocaml" => GrammarSource::new("tree-sitter", "tree-sitter-ocaml")
            .with_subdir("grammars/ocaml")
            .with_cpp_scanner(),
        "odin" => GrammarSource::new("tree-sitter-grammars", "tree-sitter-odin"),
        "pascal" => GrammarSource::new("Isopod", "tree-sitter-pascal"),
        "perl" => GrammarSource::new("tree-sitter-perl", "tree-sitter-perl")
            .with_cpp_scanner()
            .with_branch("release"),
        "php" => GrammarSource::new("tree-sitter", "tree-sitter-php")
            .with_subdir("php")
            .with_cpp_scanner(),
        "python" => GrammarSource::new("tree-sitter", "tree-sitter-python"),
        "r" => GrammarSource::new("r-lib", "tree-sitter-r"),
        "ruby" => GrammarSource::new("tree-sitter", "tree-sitter-ruby").with_cpp_scanner(),
        "rust" => GrammarSource::new("tree-sitter", "tree-sitter-rust"),
        "scala" => GrammarSource::new("tree-sitter", "tree-sitter-scala").with_cpp_scanner(),
        "scheme" => GrammarSource::new("6cdh", "tree-sitter-scheme"),
        "sql" => GrammarSource::new("derekstride", "tree-sitter-sql").with_cpp_scanner(),
        "swift" => GrammarSource::new("alex-pinkus", "tree-sitter-swift").with_cpp_scanner(),
        "toml" => GrammarSource::new("tree-sitter-grammars", "tree-sitter-toml"),
        "tsx" => GrammarSource::new("tree-sitter", "tree-sitter-typescript")
            .with_subdir("tsx")
            .with_cpp_scanner(),
        "typescript" => GrammarSource::new("tree-sitter", "tree-sitter-typescript")
            .with_subdir("typescript")
            .with_cpp_scanner(),
        "vala" => GrammarSource::new("vala-lang", "tree-sitter-vala"),
        "vue" => GrammarSource::new("tree-sitter-grammars", "tree-sitter-vue").with_cpp_scanner(),
        "yaml" => GrammarSource::new("tree-sitter-grammars", "tree-sitter-yaml").with_cpp_scanner(),
        "zig" => GrammarSource::new("tree-sitter-grammars", "tree-sitter-zig"),
        _ => return None,
    };

    Some(source)
}

/// List all languages with known grammar sources.
pub fn known_languages() -> Vec<&'static str> {
    vec![
        "ada",
        "bash",
        "c",
        "c-sharp",
        "clojure",
        "cpp",
        "css",
        "dart",
        "elixir",
        "elm",
        "erlang",
        "fortran",
        "go",
        "haskell",
        "html",
        "java",
        "javascript",
        "json",
        "julia",
        "kotlin",
        "latex",
        "lisp",
        "lua",
        "markdown",
        "nix",
        "ocaml",
        "odin",
        "pascal",
        "perl",
        "php",
        "python",
        "r",
        "ruby",
        "rust",
        "scala",
        "scheme",
        "sql",
        "swift",
        "toml",
        "tsx",
        "typescript",
        "vala",
        "vue",
        "yaml",
        "zig",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_known_language() {
        let rust = lookup("rust").unwrap();
        assert_eq!(rust.owner, "tree-sitter");
        assert_eq!(rust.repo, "tree-sitter-rust");
        assert!(!rust.has_cpp_scanner);
        assert!(rust.src_subdir.is_none());
    }

    #[test]
    fn test_lookup_with_cpp_scanner() {
        let cpp = lookup("cpp").unwrap();
        assert!(cpp.has_cpp_scanner);
    }

    #[test]
    fn test_lookup_with_subdir() {
        let ts = lookup("typescript").unwrap();
        assert_eq!(ts.src_subdir, Some("typescript"));
    }

    #[test]
    fn test_lookup_alias() {
        let bash = lookup("shell").unwrap();
        assert_eq!(bash.repo, "tree-sitter-bash");

        let lisp = lookup("commonlisp").unwrap();
        assert_eq!(lisp.repo, "tree-sitter-commonlisp");
    }

    #[test]
    fn test_lookup_unknown() {
        assert!(lookup("brainfuck").is_none());
    }

    #[test]
    fn test_c_symbol_name() {
        let rust = lookup("rust").unwrap();
        assert_eq!(rust.c_symbol_name("rust"), "tree_sitter_rust");

        let lisp = lookup("lisp").unwrap();
        assert_eq!(lisp.c_symbol_name("lisp"), "tree_sitter_commonlisp");

        let csharp = lookup("c-sharp").unwrap();
        assert_eq!(csharp.c_symbol_name("c-sharp"), "tree_sitter_c_sharp");
    }

    #[test]
    fn test_release_tarball_url() {
        let rust = lookup("rust").unwrap();
        assert_eq!(
            rust.release_tarball_url("v0.24.0"),
            "https://github.com/tree-sitter/tree-sitter-rust/releases/download/v0.24.0/tree-sitter-rust.tar.gz"
        );
    }

    #[test]
    fn test_known_languages_not_empty() {
        let langs = known_languages();
        assert!(langs.len() > 40);
        assert!(langs.contains(&"rust"));
        assert!(langs.contains(&"python"));
    }
}
