//! Grammar-install security gate + registry summary (WI-e1, #82).
//!
//! The daemon's `LanguageService::InstallGrammar` handler must validate every
//! request BEFORE any download / compile / dlopen happens. This module owns that
//! gate as a pure, unit-testable function so the security contract is verified
//! without standing up gRPC:
//!
//! 1. the language name is a safe identifier (no path separators / URL
//!    metacharacters) — it is used to build cache paths and a download URL;
//! 2. the name is in the bundled registry allowlist AND has a grammar source;
//! 3. the configured grammar source base URL uses `https://` (config-sanity
//!    guard). Note: the actual fetch URLs are derived from the bundled grammar
//!    registry and are hardcoded `https://github.com/{owner}/{repo}/...` — so
//!    the transport is HTTPS-pinned by construction regardless of this value;
//! 4. checksum verification is enabled (fail-closed config gate). Note: the
//!    downloader currently records the compiled-library checksum as metadata but
//!    does NOT yet verify the downloaded tarball against a pinned expected
//!    checksum — tarball checksum pinning is tracked as a follow-up. This gate
//!    only refuses installs when verification is configured off.
//!
//! [`registry_summary`] backs `RefreshLanguageRegistry`.

use crate::language_registry::providers::registry::RegistryProvider;
use crate::language_registry::LanguageDefinition;

/// Reasons an install request is rejected by the security gate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GrammarInstallError {
    /// The language name contains path separators, URL metacharacters, or is
    /// otherwise not a bare lowercase identifier.
    #[error("invalid language name: {0}")]
    InvalidName(String),

    /// The language is not present in the bundled registry allowlist.
    #[error("unknown language (not in registry allowlist): {0}")]
    UnknownLanguage(String),

    /// The language is known but has no grammar source defined.
    #[error("language has no grammar source: {0}")]
    NoGrammarSource(String),

    /// The configured grammar source base URL is not `https://`.
    #[error("insecure grammar source URL (https required): {0}")]
    InsecureSource(String),

    /// Checksum verification is disabled — installs are refused fail-closed.
    #[error("checksum verification is disabled; refusing to install grammars")]
    ChecksumVerificationDisabled,

    /// The registry could not be read.
    #[error("registry error: {0}")]
    Registry(String),
}

/// Validate that `name` is a bare lowercase grammar identifier.
///
/// Accepts `[a-z0-9]` plus `_ + - .` in interior positions, but rejects:
/// path separators (`/`, `\`), `..`, leading `.`/`-`, whitespace, and URL
/// metacharacters (`: ? # % @ & = ~` etc.). This runs before any allowlist
/// lookup so a malicious name never reaches path/URL construction.
pub fn validate_language_name(name: &str) -> Result<(), GrammarInstallError> {
    let invalid = || GrammarInstallError::InvalidName(name.to_string());
    if name.is_empty() || name.len() > 64 {
        return Err(invalid());
    }
    if name.contains("..") {
        return Err(invalid());
    }
    let mut chars = name.chars();
    let first = chars.next().ok_or_else(invalid)?;
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return Err(invalid());
    }
    for c in name.chars() {
        let ok = c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '+' | '.');
        if !ok {
            return Err(invalid());
        }
    }
    Ok(())
}

/// The single security gate for `InstallGrammar`.
///
/// Runs the four checks documented at the module level and, on success, returns
/// the matched [`LanguageDefinition`] (so the handler can report status without a
/// second registry lookup). Performs NO download / compile / dlopen itself.
pub fn validate_install_request(
    name: &str,
    download_base_url: &str,
    verify_checksums: bool,
) -> Result<LanguageDefinition, GrammarInstallError> {
    // (1) safe identifier — before any path/URL construction.
    validate_language_name(name)?;

    // (3) pinned https source.
    if !is_https_url(download_base_url) {
        return Err(GrammarInstallError::InsecureSource(
            download_base_url.to_string(),
        ));
    }

    // (4) checksum verification must be on — fail closed.
    if !verify_checksums {
        return Err(GrammarInstallError::ChecksumVerificationDisabled);
    }

    // (2) allowlist membership + grammar source present.
    let def = find_language(name)?
        .ok_or_else(|| GrammarInstallError::UnknownLanguage(name.to_string()))?;
    if !def.has_grammar() {
        return Err(GrammarInstallError::NoGrammarSource(name.to_string()));
    }
    Ok(def)
}

/// True when `url` is a syntactically https URL (scheme check only).
fn is_https_url(url: &str) -> bool {
    let lower = url.trim().to_ascii_lowercase();
    lower.starts_with("https://") && lower.len() > "https://".len()
}

/// Look up a language definition by canonical id (lowercased name) in the
/// bundled registry allowlist.
pub fn find_language(name: &str) -> Result<Option<LanguageDefinition>, GrammarInstallError> {
    let provider =
        RegistryProvider::new().map_err(|e| GrammarInstallError::Registry(e.to_string()))?;
    let target = name.to_ascii_lowercase();
    Ok(provider
        .definitions()
        .iter()
        .find(|d| d.id() == target)
        .cloned())
}

/// Summary counts for `RefreshLanguageRegistry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegistrySummary {
    pub total: u32,
    pub with_grammars: u32,
    pub with_lsp: u32,
    pub with_semantic_patterns: u32,
}

/// Compute registry summary counts from the bundled language registry.
pub fn registry_summary() -> Result<RegistrySummary, GrammarInstallError> {
    let provider =
        RegistryProvider::new().map_err(|e| GrammarInstallError::Registry(e.to_string()))?;
    let defs = provider.definitions();
    Ok(RegistrySummary {
        total: defs.len() as u32,
        with_grammars: defs.iter().filter(|d| d.has_grammar()).count() as u32,
        with_lsp: defs.iter().filter(|d| d.has_lsp()).count() as u32,
        with_semantic_patterns: defs.iter().filter(|d| d.has_semantic_patterns()).count() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HTTPS: &str = "https://github.com/example/grammars/releases/download";

    #[test]
    fn rejects_path_separators_and_metacharacters() {
        for bad in [
            "rust/../etc",
            "../rust",
            "rust/evil",
            "rust\\evil",
            "ru:st",
            "rust?x",
            "rust#x",
            "rust%2e",
            "http://x",
            "rust ",
            " rust",
            "RUST", // uppercase first char rejected (canonical ids are lowercase)
            "",
        ] {
            assert!(
                validate_language_name(bad).is_err(),
                "expected reject for {bad:?}"
            );
        }
    }

    #[test]
    fn accepts_bare_identifiers() {
        for good in [
            "rust", "c", "cpp", "c-sharp", "f_star", "ocaml", "html5", "x86",
        ] {
            assert!(
                validate_language_name(good).is_ok(),
                "expected accept for {good:?}"
            );
        }
    }

    #[test]
    fn out_of_allowlist_name_is_rejected_without_touching_the_downloader() {
        // A well-formed but unknown language must fail at the allowlist step.
        let err = validate_install_request("definitelynotalanguage", HTTPS, true).unwrap_err();
        assert_eq!(
            err,
            GrammarInstallError::UnknownLanguage("definitelynotalanguage".to_string())
        );
    }

    #[test]
    fn non_https_source_is_rejected() {
        let err = validate_install_request("rust", "http://insecure/grammars", true).unwrap_err();
        assert!(matches!(err, GrammarInstallError::InsecureSource(_)));
    }

    #[test]
    fn checksum_disabled_is_rejected_fail_closed() {
        let err = validate_install_request("rust", HTTPS, false).unwrap_err();
        assert_eq!(err, GrammarInstallError::ChecksumVerificationDisabled);
    }

    #[test]
    fn valid_known_language_passes_the_gate() {
        // `rust` is in the bundled registry with a grammar source.
        let def = validate_install_request("rust", HTTPS, true).expect("rust must pass the gate");
        assert_eq!(def.id(), "rust");
        assert!(def.has_grammar());
    }

    #[test]
    fn registry_summary_counts_are_sane() {
        let s = registry_summary().expect("summary");
        assert!(
            s.total >= 40,
            "expected the bundled ~44 languages, got {}",
            s.total
        );
        assert!(s.with_grammars > 0);
        assert!(s.with_grammars <= s.total);
        assert!(s.with_semantic_patterns <= s.total);
        assert!(s.with_lsp <= s.total);
    }
}
