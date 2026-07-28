//! The SQL-family query grammar, at the subset this build executes.
//!
//! # Why a parser exists at all in a walking skeleton
//!
//! MCP-SURFACE.md §1.6 rules that the agent's read surface is **one verb whose `q`
//! parameter carries a SQL-family grammar**, on Chris's 20260723 rationale that SQL
//! is the highest-density language in a model's pretraining corpus. `query` cannot
//! be served without parsing something, and serving a *different* input shape
//! "for now" would mean the first agent to meet this build learns a grammar the
//! product does not have.
//!
//! # The subset is declared, never assumed
//!
//! §5.4's law: every member of `core` is executed by every conforming build, and a
//! clause outside it returns `grammar_unsupported` **naming the manifest key**. So
//! this parser accepts the whole shape and refuses by name, rather than failing to
//! parse what it merely cannot run -- the difference matters to a caller, because
//! "I do not understand you" and "I understand you and this build cannot do that"
//! are different corrections.
//!
//! Executed here:
//!
//! ```text
//! (SELECT|SEARCH) [SEMANTIC|TEXT|EXACT|REGEX] <object> FROM <collection>
//!                 [WHERE q MATCH '<text>'] [LIMIT <n>]
//! ```
//!
//! `FROM` is **required** in this build and the manifest says so
//! (`core.from_optional: false`): making it optional means resolving the project
//! containing the caller's working directory, and this build has no project
//! detection -- `status` reports `project: null`. Defaulting to a project that
//! cannot be resolved would be the silent-wrong-scope failure, so the clause is
//! required and the requirement is declared.

use wqm_common::names::{collection_name, Collection};
use wqm_common::plan::{Mode, ObjectKind};

use crate::{known_collections, QueryError};

/// A parsed query, before any planning decision has been made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedQuery {
    /// The stated mode, or `None` when the caller left it to the default.
    pub mode: Option<Mode>,
    /// Whether the caller spelled the mode `EXACT`, §1.6's deprecated alias of
    /// `TEXT`. Carried so the response can echo the normalized spelling and the
    /// caller can see the normalization happened.
    pub mode_was_alias: bool,
    /// What kind of result was asked for.
    pub object: ObjectKind,
    /// The collection named by `FROM`.
    pub source: Collection,
    /// The literal the `q MATCH` predicate compares against.
    pub match_text: Option<String>,
    /// The `LIMIT` clause's value, when one was given.
    pub limit: Option<u32>,
}

/// One lexical token and where it started, so an error can point at it.
struct Token {
    text: String,
    quoted: bool,
    position: usize,
}

/// Parse `q` into a [`ParsedQuery`], or fail with a position and a suggestion.
pub fn parse(q: &str) -> Result<ParsedQuery, QueryError> {
    let tokens = tokenize(q)?;
    let mut cursor = Cursor {
        tokens: &tokens,
        at: 0,
        input_len: q.len(),
    };

    cursor.expect_keyword(&["SELECT", "SEARCH"], &["SELECT", "SEARCH"])?;
    let (mode, mode_was_alias) = cursor.optional_mode();
    let object = cursor.object()?;
    cursor.expect_keyword(&["FROM"], &["FROM"])?;
    let source = cursor.source()?;
    let match_text = cursor.optional_where()?;
    let limit = cursor.optional_limit()?;
    cursor.expect_end()?;

    Ok(ParsedQuery {
        mode,
        mode_was_alias,
        object,
        source,
        match_text,
        limit,
    })
}

/// Split `q` into words, treating a `'…'` run as one token. An unterminated quote
/// is a parse error rather than a token that swallows the rest of the query.
fn tokenize(q: &str) -> Result<Vec<Token>, QueryError> {
    let mut tokens = Vec::new();
    let bytes = q.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        if bytes[i] == b'\'' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'\'' {
                i += 1;
            }
            if i == bytes.len() {
                return Err(QueryError::Parse {
                    position: start,
                    message: "the quoted value is never closed".into(),
                    expected: vec!["'"],
                    suggestion: Some(format!("{q}'")),
                });
            }
            tokens.push(Token {
                text: q[start + 1..i].to_string(),
                quoted: true,
                position: start,
            });
            i += 1;
        } else {
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'\'' {
                i += 1;
            }
            tokens.push(Token {
                text: q[start..i].to_string(),
                quoted: false,
                position: start,
            });
        }
    }
    Ok(tokens)
}

/// A position in the token stream. Every method that consumes reports where it
/// stopped, which is what makes §1.6's mandatory position/expected/suggestion
/// triple derivable rather than approximated.
struct Cursor<'a> {
    tokens: &'a [Token],
    at: usize,
    input_len: usize,
}

impl Cursor<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    /// Where the *next* token starts, or the end of the input when there is none.
    fn position(&self) -> usize {
        self.peek().map_or(self.input_len, |t| t.position)
    }

    fn found(&self) -> String {
        self.peek()
            .map_or_else(|| "the end of the query".to_string(), |t| t.text.clone())
    }

    fn parse_error(&self, message: String, expected: &[&'static str]) -> QueryError {
        QueryError::Parse {
            position: self.position(),
            message,
            expected: expected.to_vec(),
            suggestion: None,
        }
    }

    /// Consume the next token if it matches one of `words`, case-insensitively.
    fn take_keyword(&mut self, words: &[&str]) -> Option<String> {
        let token = self.peek()?;
        if token.quoted {
            return None;
        }
        let upper = token.text.to_ascii_uppercase();
        if words.iter().any(|w| *w == upper) {
            self.at += 1;
            return Some(upper);
        }
        None
    }

    fn expect_keyword(
        &mut self,
        words: &[&str],
        expected: &[&'static str],
    ) -> Result<String, QueryError> {
        self.take_keyword(words).ok_or_else(|| {
            let found = self.found();
            self.parse_error(
                format!("expected {} but found `{found}`", expected[0]),
                expected,
            )
        })
    }

    /// `[SEMANTIC|TEXT|EXACT|REGEX]`. Absence is not an error here -- the mode has
    /// a default, and whether this build can serve that default is the planner's
    /// question, not the parser's.
    fn optional_mode(&mut self) -> (Option<Mode>, bool) {
        match self
            .take_keyword(&["SEMANTIC", "TEXT", "EXACT", "REGEX"])
            .as_deref()
        {
            Some("SEMANTIC") => (Some(Mode::Semantic), false),
            Some("TEXT") => (Some(Mode::Text), false),
            // §1.6: `EXACT` is accepted as a deprecated alias of `TEXT` and echoed
            // back normalized, so a caller sees which spelling actually ran.
            Some("EXACT") => (Some(Mode::Text), true),
            Some("REGEX") => (Some(Mode::Regex), false),
            _ => (None, false),
        }
    }

    fn object(&mut self) -> Result<ObjectKind, QueryError> {
        const EXPECTED: &[&str] = &["an object name"];
        let token = self.peek().ok_or_else(|| {
            self.parse_error("the query names no object to select".into(), EXPECTED)
        })?;
        let name = token.text.to_ascii_lowercase();
        let object = object_by_name(&name).ok_or_else(|| QueryError::UnknownReference {
            kind: "object",
            name: name.clone(),
            known: OBJECT_NAMES.to_vec(),
            message: format!("`{name}` is not an object this surface addresses"),
        })?;
        self.at += 1;
        Ok(object)
    }

    fn source(&mut self) -> Result<Collection, QueryError> {
        const EXPECTED: &[&str] = &["a collection name"];
        let token = self
            .peek()
            .ok_or_else(|| self.parse_error("`FROM` names no source".into(), EXPECTED))?;
        let name = token.text.to_ascii_lowercase();
        let source = Collection::ALL
            .into_iter()
            .find(|c| collection_name(*c) == name)
            .ok_or_else(|| QueryError::UnknownReference {
                kind: "source",
                name: name.clone(),
                known: known_collections(),
                message: format!("`{name}` is not a source this build searches"),
            })?;
        self.at += 1;
        Ok(source)
    }

    /// `[WHERE q MATCH '<text>']` -- the one predicate this build executes.
    fn optional_where(&mut self) -> Result<Option<String>, QueryError> {
        if self.take_keyword(&["WHERE"]).is_none() {
            return Ok(None);
        }
        let field = self
            .expect_keyword(&["Q"], &["the field `q`"])
            .map_err(|e| self.unsupported_field(e))?;
        debug_assert_eq!(field, "Q");
        self.expect_keyword(&["MATCH"], &["MATCH"])?;

        let token = self.peek().ok_or_else(|| {
            self.parse_error(
                "`MATCH` compares against nothing".into(),
                &["a quoted value"],
            )
        })?;
        if !token.quoted {
            return Err(self.parse_error(
                format!("`MATCH` takes a quoted value; found `{}`", token.text),
                &["a quoted value"],
            ));
        }
        let text = token.text.clone();
        self.at += 1;
        Ok(Some(text))
    }

    /// A `WHERE` on any field other than `q` is understood and not executed, which
    /// is `grammar_unsupported` naming the manifest key -- not a parse failure.
    fn unsupported_field(&self, fallback: QueryError) -> QueryError {
        let Some(token) = self.peek() else {
            return fallback;
        };
        QueryError::Unsupported {
            capability_key: "fields",
            message: format!(
                "this build filters on `q` only; `{}` is declared in `core.fields` \
                 by builds that execute it",
                token.text
            ),
            suggestion: None,
        }
    }

    fn optional_limit(&mut self) -> Result<Option<u32>, QueryError> {
        if self.take_keyword(&["LIMIT"]).is_none() {
            return Ok(None);
        }
        let token = self
            .peek()
            .ok_or_else(|| self.parse_error("`LIMIT` names no number".into(), &["a number"]))?;
        let value = token.text.parse::<u32>().map_err(|_| {
            self.parse_error(
                format!("`LIMIT` takes a number; found `{}`", token.text),
                &["a number"],
            )
        })?;
        self.at += 1;
        Ok(Some(value))
    }

    /// Anything left over is a clause this build does not execute. Reporting it as
    /// `grammar_unsupported` rather than as trailing junk is what lets a caller
    /// tell "wrong syntax" from "right syntax, absent capability".
    fn expect_end(&self) -> Result<(), QueryError> {
        let Some(token) = self.peek() else {
            return Ok(());
        };
        let upper = token.text.to_ascii_uppercase();
        let key = match upper.as_str() {
            "ORDER" => "order_by",
            "AS" => "shapes",
            _ => "core",
        };
        Err(QueryError::Unsupported {
            capability_key: key,
            message: format!(
                "`{}` is a clause this build does not execute; `status` reports the \
                 clause set it does",
                token.text
            ),
            suggestion: None,
        })
    }
}

/// Every object the sealed surface addresses (§3.4's `object` enum).
const OBJECT_NAMES: &[&str] = &[
    "chunk", "line", "document", "symbol", "rule", "note", "tag", "relation", "topic",
];

fn object_by_name(name: &str) -> Option<ObjectKind> {
    Some(match name {
        "chunk" => ObjectKind::Chunk,
        "line" => ObjectKind::Line,
        "document" => ObjectKind::Document,
        "symbol" => ObjectKind::Symbol,
        "rule" => ObjectKind::Rule,
        "note" => ObjectKind::Note,
        "tag" => ObjectKind::Tag,
        "relation" => ObjectKind::Relation,
        "topic" => ObjectKind::Topic,
        _ => return None,
    })
}
