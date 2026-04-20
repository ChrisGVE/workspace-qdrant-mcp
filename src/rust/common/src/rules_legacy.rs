//! Parser for the legacy `RULE` header embedded in rule content.
//!
//! Pre-schema rules were stored with their label/scope/type/priority
//! tokens baked into the `content` text rather than the Qdrant payload:
//!
//! ```text
//! RULE
//! label:use-common-crate
//! type:constraint
//! scope:project:4ed81466dec7
//! priority:8
//! ---
//! <actual rule body>
//! ```
//!
//! `parse_rule_header` recovers the tokens for the payload backfill
//! (daemon-side) and the inject-time fallback (CLI-side).
//!
//! See issue #58.
//!
//! The parser is intentionally permissive: it accepts trailing whitespace,
//! returns all key/value pairs verbatim (`scope:project:ID` stays a single
//! value — callers split on the `project:` prefix themselves), and leaves
//! the body untouched aside from stripping the `---` separator line.

use std::collections::HashMap;

use crate::constants::TENANT_GLOBAL;

/// Marker that indicates a rule content starts with the legacy header block.
pub const RULE_HEADER_MARKER: &str = "RULE";

/// Separator line that terminates the header block.
pub const RULE_BODY_SEPARATOR: &str = "---";

/// Parsed fields recovered from a legacy `RULE` header.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LegacyRuleHeader {
    /// Map of `key -> value` pairs from the header. Only the first `:` splits
    /// the line, so values like `project:<id>` are preserved intact.
    pub fields: HashMap<String, String>,
    /// Rule body with the header and separator stripped.
    pub body: String,
}

impl LegacyRuleHeader {
    /// Return the `label` field if present and non-empty.
    pub fn label(&self) -> Option<&str> {
        self.fields
            .get("label")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
    }

    /// Return the raw scope token (e.g. `"global"` or `"project:abc123"`).
    pub fn raw_scope(&self) -> Option<&str> {
        self.fields
            .get("scope")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
    }

    /// Return `(scope, project_id)` suitable for the Qdrant payload schema.
    ///
    /// - `"global"` → `("global", None)`
    /// - `"project:<id>"` → `("project", Some("<id>"))`
    /// - any other value → `(value, None)` so callers can decide.
    /// - no scope field at all → `("global", None)` (defensive default,
    ///   mirrors how add/inject treat an empty scope).
    pub fn split_scope(&self) -> (String, Option<String>) {
        match self.raw_scope() {
            Some(s) if s == TENANT_GLOBAL => (TENANT_GLOBAL.to_string(), None),
            Some(s) => {
                if let Some(id) = s.strip_prefix("project:") {
                    ("project".to_string(), Some(id.to_string()))
                } else {
                    (s.to_string(), None)
                }
            }
            None => (TENANT_GLOBAL.to_string(), None),
        }
    }

    /// Return the rule type (`type:` header) if present and non-empty.
    pub fn rule_type(&self) -> Option<&str> {
        self.fields
            .get("type")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
    }

    /// Parse the priority field as `u32` if present and numeric.
    pub fn priority(&self) -> Option<u32> {
        self.fields.get("priority").and_then(|s| s.parse().ok())
    }
}

/// Attempt to parse a legacy `RULE` header from `content`.
///
/// Returns `Some(header)` only when the content starts with the literal
/// `RULE` marker on its own line. Otherwise returns `None` and callers
/// should leave the content alone.
///
/// The returned `body` is the content following the `---` separator; if
/// the separator is absent everything after the last `key:value` line is
/// treated as body.
pub fn parse_rule_header(content: &str) -> Option<LegacyRuleHeader> {
    let mut lines = content.lines();
    let first = lines.next()?.trim_end();
    if first != RULE_HEADER_MARKER {
        return None;
    }

    let mut fields = HashMap::new();
    let mut body_start: Option<usize> = None;
    let mut consumed = first.len();
    // Track bytes consumed so we can slice the raw body (preserves \r\n etc).
    for line in lines.by_ref() {
        consumed += 1 + line.len(); // newline + line length
        let trimmed = line.trim_end();
        if trimmed == RULE_BODY_SEPARATOR {
            body_start = Some(consumed);
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let key = k.trim().to_string();
            let value = v.trim().to_string();
            if !key.is_empty() {
                fields.insert(key, value);
            }
        } else if trimmed.is_empty() {
            // blank line inside header — tolerate
            continue;
        } else {
            // Non-`key:value` line inside the header area means we've run
            // out of header. Treat the current line as the start of the body.
            body_start = Some(consumed - line.len() - 1);
            break;
        }
    }

    let body = match body_start {
        Some(off) if off < content.len() => content[off..]
            .strip_prefix('\n')
            .unwrap_or(&content[off..])
            .to_string(),
        Some(_) => String::new(),
        None => String::new(),
    };

    Some(LegacyRuleHeader { fields, body })
}

/// Convenience: does `content` look like a legacy `RULE`-headered rule?
pub fn is_legacy_rule_content(content: &str) -> bool {
    content
        .lines()
        .next()
        .map(|l| l.trim_end() == RULE_HEADER_MARKER)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "RULE\n\
label:use-common-crate\n\
type:constraint\n\
scope:project:4ed81466dec7\n\
priority:8\n\
---\n\
When introducing or modifying shared data structures...";

    #[test]
    fn parses_full_header() {
        let header = parse_rule_header(SAMPLE).expect("should parse");
        assert_eq!(header.label(), Some("use-common-crate"));
        assert_eq!(header.rule_type(), Some("constraint"));
        assert_eq!(header.priority(), Some(8));
        let (scope, project_id) = header.split_scope();
        assert_eq!(scope, "project");
        assert_eq!(project_id.as_deref(), Some("4ed81466dec7"));
        assert!(header
            .body
            .starts_with("When introducing or modifying shared"));
    }

    #[test]
    fn rejects_non_rule_content() {
        assert!(parse_rule_header("some free text").is_none());
        assert!(parse_rule_header("").is_none());
        assert!(parse_rule_header("rule\nlabel:x").is_none()); // case sensitive
    }

    #[test]
    fn global_scope_has_no_project_id() {
        let src = "RULE\nlabel:foo\nscope:global\n---\nbody";
        let header = parse_rule_header(src).expect("should parse");
        let (scope, project_id) = header.split_scope();
        assert_eq!(scope, "global");
        assert!(project_id.is_none());
    }

    #[test]
    fn missing_scope_defaults_to_global() {
        let src = "RULE\nlabel:foo\n---\nbody";
        let header = parse_rule_header(src).expect("should parse");
        let (scope, project_id) = header.split_scope();
        assert_eq!(scope, TENANT_GLOBAL);
        assert!(project_id.is_none());
    }

    #[test]
    fn missing_separator_yields_empty_body() {
        let src = "RULE\nlabel:foo\nscope:global";
        let header = parse_rule_header(src).expect("should parse");
        assert_eq!(header.label(), Some("foo"));
        assert_eq!(header.body, "");
    }

    #[test]
    fn value_with_colon_preserved() {
        let src = "RULE\nscope:project:abc:def\n---\nbody";
        let header = parse_rule_header(src).expect("should parse");
        let (scope, project_id) = header.split_scope();
        assert_eq!(scope, "project");
        assert_eq!(project_id.as_deref(), Some("abc:def"));
    }

    #[test]
    fn is_legacy_rule_content_detects_marker() {
        assert!(is_legacy_rule_content("RULE\nlabel:foo\n---\nbody"));
        assert!(!is_legacy_rule_content("label:foo\nscope:global\nbody"));
        assert!(!is_legacy_rule_content(""));
    }

    #[test]
    fn priority_non_numeric_is_none() {
        let src = "RULE\nlabel:foo\npriority:high\n---\nbody";
        let header = parse_rule_header(src).expect("should parse");
        assert!(header.priority().is_none());
    }

    #[test]
    fn trailing_whitespace_tolerated() {
        let src = "RULE \nlabel: foo \nscope: global \n---\nbody";
        let header = parse_rule_header(src).expect("should parse");
        assert_eq!(header.label(), Some("foo"));
        assert_eq!(header.raw_scope(), Some("global"));
    }
}
