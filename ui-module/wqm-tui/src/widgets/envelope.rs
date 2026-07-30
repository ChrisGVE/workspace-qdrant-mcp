//! Notices and errors — the visible half of N12, the sealed response envelope.
//!
//! `Envelope` itself is opaque by design (its fields are private so the seven-key shape
//! cannot be assembled wrongly), but the pieces a screen must *show* — [`Notice`],
//! [`ToolError`] and their two closed vocabularies — are public, and those are what this
//! widget renders.
//!
//! Codes are never spelled here. Each is serialized out of its own type, so the string on
//! screen is the string on the wire by construction; a rename in N12 moves the frame with
//! it. The variant lists in the pantry ingredients are the one place this crate does
//! enumerate the vocabularies by hand — a new member added to N12 will not appear here
//! until someone adds it, which is why "All Codes" exists as a variant: it makes the
//! omission visible instead of silent.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};
use serde::Serialize;
use wqm_common::envelope::{Notice, Severity, ToolError};

use crate::tokens::{self, Health};

/// The wire spelling of a closed-vocabulary member, taken from its own serde attributes
/// rather than retyped.
fn wire<T: Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        // A non-string serialization would mean N12 changed shape underneath us; saying so
        // beats rendering an empty cell.
        .unwrap_or_else(|| "<not a code>".to_string())
}

/// §4's glyph vocabulary reused for severity: `info` is quiet, `warn` is the degraded
/// triangle. Reusing the shapes keeps one mechanism to one meaning across the screen.
fn severity_marks(severity: Severity) -> (&'static str, Style) {
    match severity {
        Severity::Info => ("·", tokens::faint_style()),
        Severity::Warn => (
            Health::Degraded.glyph(),
            Style::default().fg(Health::Degraded.color()),
        ),
    }
}

/// A list of notices, as the status zone shows them.
pub struct Notices {
    notices: Vec<Notice>,
}

impl Notices {
    pub fn new(notices: Vec<Notice>) -> Self {
        Self { notices }
    }
}

impl Widget for Notices {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let code_width = self
            .notices
            .iter()
            .map(|n| wire(&n.code).len())
            .max()
            .unwrap_or(0);

        let lines: Vec<Line> = self
            .notices
            .iter()
            .map(|notice| {
                let (glyph, glyph_style) = severity_marks(notice.severity);
                Line::from(vec![
                    Span::styled(glyph, glyph_style),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:<code_width$}  ", wire(&notice.code)),
                        tokens::muted_style(),
                    ),
                    Span::styled(notice.message.clone(), tokens::normal_style()),
                ])
            })
            .collect();

        Paragraph::new(lines).render(area, buf);
    }
}

/// A tool-level failure, as a modal or status line shows it.
pub struct ErrorPanel {
    error: ToolError,
}

impl ErrorPanel {
    pub fn new(error: ToolError) -> Self {
        Self { error }
    }
}

impl Widget for ErrorPanel {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let lines = vec![
            Line::from(vec![
                Span::styled(
                    Health::Offline.glyph(),
                    Style::default().fg(Health::Offline.color()),
                ),
                Span::raw(" "),
                // Goal 7: the code is the datum the user must see, so it carries weight.
                Span::styled(wire(&self.error.code), tokens::strong_style()),
            ]),
            Line::from(vec![Span::styled(
                self.error.message.clone(),
                tokens::normal_style(),
            )]),
            Line::from(vec![
                Span::styled("retryable   ", tokens::muted_style()),
                // `retryable` exists so an agent decides without pattern-matching prose;
                // the screen states it just as plainly.
                Span::styled(
                    if self.error.retryable { "yes" } else { "no" },
                    tokens::faint_style(),
                ),
            ]),
        ];

        Paragraph::new(lines).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};
    // Only the preview frames enumerate the closed vocabularies; importing them at module
    // scope left them unused in a build without this feature.
    use wqm_common::envelope::{ErrorCode, NoticeCode};

    /// Every member of §4.4's closed ten. Listed rather than iterated because the
    /// vocabulary is not enumerable at runtime; the "All Codes" frame is what makes a
    /// missing member visible.
    const ALL_NOTICES: [NoticeCode; 10] = [
        NoticeCode::FilterEmptied,
        NoticeCode::CorpusEmpty,
        NoticeCode::DegradedResult,
        NoticeCode::IndexLag,
        NoticeCode::StaleSource,
        NoticeCode::PartialSource,
        NoticeCode::CapabilityOff,
        NoticeCode::BudgetTruncated,
        NoticeCode::SchemaBudgetTruncated,
        NoticeCode::ProtocolDowngraded,
    ];

    /// Every member of §4.3's closed seven.
    const ALL_ERRORS: [ErrorCode; 7] = [
        ErrorCode::InvalidArgument,
        ErrorCode::GrammarParse,
        ErrorCode::GrammarUnsupported,
        ErrorCode::UnknownReference,
        ErrorCode::Refused,
        ErrorCode::BackendUnavailable,
        ErrorCode::Internal,
    ];

    fn notice(code: NoticeCode, severity: Severity, message: &str) -> Notice {
        Notice {
            code,
            severity,
            message: message.to_string(),
            details: serde_json::Value::Null,
        }
    }

    struct AllNoticeCodes;
    impl Ingredient for AllNoticeCodes {
        fn group(&self) -> &str {
            "Notices"
        }
        fn name(&self) -> &str {
            "All Codes"
        }
        fn source(&self) -> &str {
            "wqm_common::envelope::NoticeCode"
        }
        fn description(&self) -> &str {
            "All ten §4.4 codes at once — the width test: schema_budget_truncated is the longest"
        }
        fn props(&self) -> &[PropInfo] {
            &[PropInfo {
                name: "notices",
                ty: "Vec<Notice>",
                description: "The always-present channel; frequently empty, never absent",
            }]
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let notices = ALL_NOTICES
                .iter()
                .map(|code| {
                    // Alternating severity so both markers appear in one frame.
                    let severity = if wire(code).len() % 2 == 0 {
                        Severity::Info
                    } else {
                        Severity::Warn
                    };
                    notice(*code, severity, "one human-readable sentence")
                })
                .collect();
            Notices::new(notices).render(area, buf);
        }
    }

    struct SeverityContrast;
    impl Ingredient for SeverityContrast {
        fn group(&self) -> &str {
            "Notices"
        }
        fn name(&self) -> &str {
            "Info vs Warn"
        }
        fn source(&self) -> &str {
            "wqm_common::envelope::Severity"
        }
        fn description(&self) -> &str {
            "Does warn read as louder than info without the status line out-shouting the content?"
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            Notices::new(vec![
                notice(
                    NoticeCode::CorpusEmpty,
                    Severity::Info,
                    "no document in this collection matches",
                ),
                notice(
                    NoticeCode::IndexLag,
                    Severity::Warn,
                    "the index is 96s behind its sources",
                ),
            ])
            .render(area, buf);
        }
    }

    struct Empty;
    impl Ingredient for Empty {
        fn group(&self) -> &str {
            "Notices"
        }
        fn name(&self) -> &str {
            "Empty"
        }
        fn source(&self) -> &str {
            "wqm_common::envelope"
        }
        fn description(&self) -> &str {
            "The common case: the channel is present and empty. It must occupy no chrome."
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            Notices::new(Vec::new()).render(area, buf);
        }
    }

    struct AllErrorCodes;
    impl Ingredient for AllErrorCodes {
        fn group(&self) -> &str {
            "Tool Error"
        }
        fn name(&self) -> &str {
            "All Codes"
        }
        fn source(&self) -> &str {
            "wqm_common::envelope::ErrorCode"
        }
        fn description(&self) -> &str {
            "All seven §4.3 codes stacked, so no spelling overflows the panel"
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let lines: Vec<Line> = ALL_ERRORS
                .iter()
                .map(|code| {
                    Line::from(vec![
                        Span::styled(
                            Health::Offline.glyph(),
                            Style::default().fg(Health::Offline.color()),
                        ),
                        Span::raw(" "),
                        Span::styled(wire(code), tokens::normal_style()),
                    ])
                })
                .collect();
            Paragraph::new(lines).render(area, buf);
        }
    }

    struct RefusedError;
    impl Ingredient for RefusedError {
        fn group(&self) -> &str {
            "Tool Error"
        }
        fn name(&self) -> &str {
            "Refused"
        }
        fn source(&self) -> &str {
            "wqm_common::envelope::ToolError"
        }
        fn description(&self) -> &str {
            "A write refused by a gate — code carries weight, message stays a plain sentence"
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            ErrorPanel::new(ToolError {
                code: ErrorCode::Refused,
                message: "the rules collection is not writable from this surface".to_string(),
                details: serde_json::Value::Null,
                retryable: false,
            })
            .render(area, buf);
        }
    }

    struct BackendUnavailableError;
    impl Ingredient for BackendUnavailableError {
        fn group(&self) -> &str {
            "Tool Error"
        }
        fn name(&self) -> &str {
            "Backend Unavailable"
        }
        fn source(&self) -> &str {
            "wqm_common::envelope::ErrorCode"
        }
        fn description(&self) -> &str {
            "Retryable, and never raised by `status` — unreachability there is data, not an error"
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            ErrorPanel::new(ToolError {
                code: ErrorCode::BackendUnavailable,
                message: "the vector backend did not answer".to_string(),
                details: serde_json::Value::Null,
                retryable: true,
            })
            .render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(AllNoticeCodes),
            Box::new(SeverityContrast),
            Box::new(Empty),
            Box::new(AllErrorCodes),
            Box::new(RefusedError),
            Box::new(BackendUnavailableError),
        ]
    }
}
