//! The grammar subset, and the line between "I do not understand you" and "I
//! understand you and this build cannot do that".
//!
//! That line is the whole reason the parser accepts the full shape and refuses by
//! name. A caller who gets `grammar_parse` has written something wrong; a caller
//! who gets `grammar_unsupported` has written something right that this build does
//! not run, and the two corrections are different.

use wqm_common::names::Collection;
use wqm_common::plan::{Mode, ObjectKind};
use wqm_search::{parse, QueryError};

fn q(text: &str) -> String {
    format!(
        "SELECT TEXT note FROM {} WHERE q MATCH '{text}'",
        Collection::Scratchpad.name()
    )
}

#[test]
fn the_worked_example_parses() {
    let parsed = parse(&q("skeleton")).expect("the shipped example parses");
    assert_eq!(parsed.mode, Some(Mode::Text));
    assert_eq!(parsed.object, ObjectKind::Note);
    assert_eq!(parsed.source, Collection::Scratchpad);
    assert_eq!(parsed.match_text.as_deref(), Some("skeleton"));
    assert_eq!(parsed.limit, None);
}

#[test]
fn keywords_are_case_insensitive_and_search_is_a_synonym_of_select() {
    let lowered = format!(
        "search text note from {} where q match 'seam' limit 3",
        Collection::Scratchpad.name()
    );
    let parsed = parse(&lowered).expect("lowercase parses");
    assert_eq!(parsed.match_text.as_deref(), Some("seam"));
    assert_eq!(parsed.limit, Some(3));
}

/// §1.6: `EXACT` is a deprecated alias of `TEXT`, "echoed back normalized". The
/// flag is what lets the response disclose that the normalization happened.
#[test]
fn exact_is_an_alias_of_text_and_says_so() {
    let parsed = parse(&format!(
        "SELECT EXACT note FROM {} WHERE q MATCH 'seam'",
        Collection::Scratchpad.name()
    ))
    .expect("the alias parses");
    assert_eq!(parsed.mode, Some(Mode::Text));
    assert!(
        parsed.mode_was_alias,
        "the caller must be told it was normalized"
    );
}

#[test]
fn an_unterminated_quote_is_a_parse_error_with_a_position_and_a_fix() {
    let err = parse(&format!(
        "SELECT TEXT note FROM {} WHERE q MATCH 'seam",
        Collection::Scratchpad.name()
    ))
    .expect_err("an open quote cannot parse");
    let QueryError::Parse {
        position,
        suggestion,
        ..
    } = err
    else {
        panic!("an unterminated quote is a parse failure, not a capability one");
    };
    assert!(position > 0, "the error points at the opening quote");
    assert!(
        suggestion.is_some_and(|s| s.ends_with('\'')),
        "the suggestion closes the quote, so the second call is a correction"
    );
}

/// A clause this build understands and does not execute is `grammar_unsupported`
/// carrying the manifest key -- never a parse failure, and never silence.
#[test]
fn an_order_by_clause_is_unsupported_and_names_its_manifest_key() {
    let err = parse(&format!(
        "SELECT TEXT note FROM {} WHERE q MATCH 'seam' ORDER BY score",
        Collection::Scratchpad.name()
    ))
    .expect_err("ORDER BY is not executed here");
    let QueryError::Unsupported { capability_key, .. } = err else {
        panic!("a well-formed unsupported clause must not be reported as bad syntax");
    };
    assert_eq!(capability_key, "order_by");
}

#[test]
fn a_predicate_on_another_field_is_unsupported_and_names_the_field_key() {
    let err = parse(&format!(
        "SELECT TEXT note FROM {} WHERE path ~ 'src/**'",
        Collection::Scratchpad.name()
    ))
    .expect_err("this build filters on `q` only");
    let QueryError::Unsupported {
        capability_key,
        message,
        ..
    } = err
    else {
        panic!("a known field this build cannot filter on is a capability limit");
    };
    assert_eq!(capability_key, "fields");
    assert!(
        message.contains("path"),
        "the refusal names what was asked for"
    );
}

#[test]
fn an_unknown_source_lists_what_does_exist() {
    let err = parse("SELECT TEXT note FROM notacollection WHERE q MATCH 'x'")
        .expect_err("an unknown source cannot resolve");
    let QueryError::UnknownReference { known, kind, .. } = err else {
        panic!("addressing something that does not exist is `unknown_reference`");
    };
    assert_eq!(kind, "source");
    assert!(
        known.contains(&Collection::Scratchpad.name()),
        "the correction is in the error, so it costs no second call"
    );
}

/// `FROM` is required on this build, and the requirement is declared
/// (`core.from_optional: false`) rather than discovered.
#[test]
fn a_missing_from_clause_is_a_parse_error() {
    let err = parse("SELECT TEXT note WHERE q MATCH 'seam'")
        .expect_err("this build cannot resolve a default scope");
    assert!(matches!(err, QueryError::Parse { .. }));
}
