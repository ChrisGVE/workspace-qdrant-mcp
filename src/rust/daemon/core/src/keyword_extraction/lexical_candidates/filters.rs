//! Stop-lists and junk-pattern filters for lexical candidate extraction.

use regex::Regex;

/// English prose boilerplate terms to filter out.
pub(super) const PROSE_STOPLIST: &[&str] = &[
    "introduction",
    "section",
    "references",
    "conclusion",
    "abstract",
    "chapter",
    "appendix",
    "figure",
    "table",
    "example",
    "note",
    "see",
    "also",
    "shall",
    "hereby",
    "therefore",
    "whereas",
    "overview",
    "summary",
    "background",
    "related",
    "previous",
];

/// Code boilerplate terms to filter out.
pub(super) const CODE_STOPLIST: &[&str] = &[
    "impl",
    "mod",
    "struct",
    "pub",
    "self",
    "async",
    "let",
    "mut",
    "fn",
    "use",
    "crate",
    "super",
    "return",
    "match",
    "enum",
    "trait",
    "const",
    "static",
    "type",
    "where",
    "move",
    "ref",
    "dyn",
    "box",
    "data",
    "value",
    "item",
    "result",
    "error",
    "option",
    "none",
    "some",
    "true",
    "false",
    "null",
    "undefined",
    "var",
    "def",
    "class",
    "import",
    "export",
    "from",
    "require",
    "module",
    "function",
    "new",
    "delete",
    "void",
    "int",
    "string",
    "bool",
    "float",
    "double",
    "char",
    "byte",
    "args",
    "kwargs",
    "self",
    "this",
    "super",
    "extends",
    "implements",
    "override",
    "final",
    "abstract",
    "virtual",
    "inline",
    "extern",
    "static",
    "const",
    "volatile",
    "register",
    "sizeof",
    "typeof",
    "instanceof",
    "throw",
    "catch",
    "try",
    "finally",
    "yield",
    "await",
    "break",
    "continue",
    "goto",
    "switch",
    "case",
    "default",
    "while",
    "for",
    "do",
    "if",
    "else",
    "elif",
    "unless",
    "until",
    "loop",
    "begin",
    "end",
    "then",
    "elsif",
    "rescue",
    "ensure",
    "raise",
    "pass",
    "lambda",
    "with",
    "assert",
    "print",
    "println",
    "printf",
    "fmt",
    "todo",
    "fixme",
    "hack",
    "xxx",
    "note",
    "warn",
];

/// Junk pattern regexes compiled once.
pub(super) struct JunkPatterns {
    hex_hash: Regex,
    version: Regex,
    path: Regex,
    hex_literal: Regex,
    uuid: Regex,
    single_letter: Regex,
    all_digits: Regex,
}

impl JunkPatterns {
    pub(super) fn new() -> Self {
        Self {
            hex_hash: Regex::new(r"^[a-f0-9]{8,}$").unwrap(),
            version: Regex::new(r"^v?\d+\.\d+").unwrap(),
            path: Regex::new(r"^[/\\]|[/\\]").unwrap(),
            hex_literal: Regex::new(r"^0x[a-f0-9]+$").unwrap(),
            uuid: Regex::new(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
                .unwrap(),
            single_letter: Regex::new(r"^[a-z]$").unwrap(),
            all_digits: Regex::new(r"^\d+$").unwrap(),
        }
    }

    pub(super) fn is_junk(&self, phrase: &str) -> bool {
        self.hex_hash.is_match(phrase)
            || self.version.is_match(phrase)
            || self.path.is_match(phrase)
            || self.hex_literal.is_match(phrase)
            || self.uuid.is_match(phrase)
            || self.single_letter.is_match(phrase)
            || self.all_digits.is_match(phrase)
    }
}

/// Check if a phrase is in the stoplist.
pub(super) fn is_stopword(phrase: &str, is_code: bool) -> bool {
    if wqm_common::nlp::ENGLISH_STOPWORDS.contains(&phrase) {
        return true;
    }
    if PROSE_STOPLIST.contains(&phrase) {
        return true;
    }
    if is_code && CODE_STOPLIST.contains(&phrase) {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_junk_patterns_hex() {
        let junk = JunkPatterns::new();
        assert!(junk.is_junk("abc123def456"));
        assert!(junk.is_junk("0xdeadbeef"));
        assert!(!junk.is_junk("hello"));
    }

    #[test]
    fn test_junk_patterns_version() {
        let junk = JunkPatterns::new();
        assert!(junk.is_junk("v2.3.1"));
        assert!(junk.is_junk("1.0.0"));
        assert!(!junk.is_junk("vector"));
    }

    #[test]
    fn test_junk_patterns_uuid() {
        let junk = JunkPatterns::new();
        assert!(junk.is_junk("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!junk.is_junk("hello-world"));
    }

    #[test]
    fn test_junk_patterns_digits() {
        let junk = JunkPatterns::new();
        assert!(junk.is_junk("12345"));
        assert!(!junk.is_junk("12abc"));
    }
}
