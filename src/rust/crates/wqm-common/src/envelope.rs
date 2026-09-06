//! N12 -- the response envelope every MCP tool answers in (MCP-SURFACE.md §3.1),
//! carrying §4.3's error vocabulary and §4.4's notice vocabulary. Seeded at
//! `P04-GT001-WO011` with the sealed seven-key shape; the per-tool `data`
//! shapes live with the tools that produce them.
//!
//! **Seven keys, always all seven, on every tool, on success and on every
//! tool-level failure.** That is the sealed shape, and it is the cure for AS-F023:
//! a key that appears only under some conditions is a key an agent that never met
//! those conditions does not know exists. So `notices` and `defaults_applied` are
//! arrays that are frequently empty and never absent, and `data`/`error` are
//! present-and-null rather than missing.
//!
//! The type system is what keeps that true here. [`Envelope`] has no public
//! constructor that can omit a key -- `ok`/`data`/`error` are set together by
//! [`Envelope::success`] or [`Envelope::failure`], so the impossible combinations
//! (`ok:true` with an error, `ok:false` with data) are not expressible rather than
//! merely discouraged. This is the DP-8 shape: prevention, not discipline.
//!
//! The two closed vocabularies (§4.3's seven error codes, §4.4's ten notice codes)
//! are declared here **in full**, including members no tool in this build can yet
//! raise. They are closed sets in a sealed document: declaring them completely
//! costs nothing and means a later slice adds a *call site*, not a string. A
//! variant with no producer yet is honest -- an invented string later would not be.

use serde::Serialize;

/// The sealed seven-key envelope (§3.1).
#[derive(Debug, Clone, Serialize)]
pub struct Envelope {
    ok: bool,
    tool: &'static str,
    elapsed_ms: u64,
    defaults_applied: Vec<AppliedDefault>,
    notices: Vec<Notice>,
    data: Option<serde_json::Value>,
    error: Option<ToolError>,
}

impl Envelope {
    /// A successful answer. `error` is `null` by construction.
    pub fn success(tool: &'static str, elapsed_ms: u64, data: serde_json::Value) -> Self {
        Envelope {
            ok: true,
            tool,
            elapsed_ms,
            defaults_applied: Vec::new(),
            notices: Vec::new(),
            data: Some(data),
            error: None,
        }
    }

    /// A tool-level failure. `data` is `null` by construction.
    ///
    /// Protocol-level failures (`invalid_argument`) have no envelope at all --
    /// they ride the JSON-RPC error's `data` member (§3.1, §4.1) -- which is why
    /// that path does not come through here.
    pub fn failure(tool: &'static str, elapsed_ms: u64, error: ToolError) -> Self {
        Envelope {
            ok: false,
            tool,
            elapsed_ms,
            defaults_applied: Vec::new(),
            notices: Vec::new(),
            data: None,
            error: Some(error),
        }
    }

    /// Record a default this call applied on the caller's behalf (§3.1: L-2 is
    /// bound by the envelope, not by one tool's data).
    #[must_use]
    pub fn with_default(mut self, applied: AppliedDefault) -> Self {
        self.defaults_applied.push(applied);
        self
    }

    /// Add a notice. The channel is always present; this fills it.
    #[must_use]
    pub fn with_notice(mut self, notice: Notice) -> Self {
        self.notices.push(notice);
        self
    }

    /// Whether this envelope reports success -- the flag a transport needs to set
    /// MCP's `isError`, without re-deriving it from the other keys.
    pub fn is_ok(&self) -> bool {
        self.ok
    }
}

/// One row of `defaults_applied`: the three-column disclosure §3.1 mandates,
/// plus the parameter it applies to.
#[derive(Debug, Clone, Serialize)]
pub struct AppliedDefault {
    /// The parameter that was defaulted.
    pub parameter: &'static str,
    /// The value it took.
    pub value: serde_json::Value,
    /// Why, in the caller's terms -- never an implementation note.
    pub reason: String,
    /// How to override it. The wire key is `override`, which is a Rust keyword,
    /// so the field is named for Rust and renamed for the wire -- the sealed
    /// spelling wins on the wire, always.
    #[serde(rename = "override")]
    pub override_with: String,
}

/// `notices[]` elements are `{code, severity, message, details}` (§4.4).
#[derive(Debug, Clone, Serialize)]
pub struct Notice {
    /// Which of the ten closed members this is.
    pub code: NoticeCode,
    /// `info` or `warn`.
    pub severity: Severity,
    /// One human-readable sentence.
    pub message: String,
    /// The code-specific detail map (§4.4's per-row shapes).
    pub details: serde_json::Value,
}

/// §4.4's closed ten-member notice vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NoticeCode {
    /// Zero results, and dropping the predicates would match more than zero.
    FilterEmptied,
    /// Zero results and the corpus genuinely has no match.
    CorpusEmpty,
    /// An answer was produced by a fallback or partial path.
    DegradedResult,
    /// The index is behind its sources.
    IndexLag,
    /// Results include sources known to be duplicated or superseded.
    StaleSource,
    /// A source named in `FROM` could not be searched.
    PartialSource,
    /// A clause this build can execute found nothing to add.
    CapabilityOff,
    /// The byte budget cut a result set below the requested `limit`.
    BudgetTruncated,
    /// The byte budget cut a `status {schemas:[…]}` set.
    SchemaBudgetTruncated,
    /// The client asked for a protocol version this server does not serve.
    ProtocolDowngraded,
}

/// Notice severity (§4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Worth knowing.
    Info,
    /// Something is not as the caller probably assumed.
    Warn,
}

/// `error` is `{code, message, details, retryable}` (§4.3).
#[derive(Debug, Clone, Serialize)]
pub struct ToolError {
    /// Which of the seven closed members this is.
    pub code: ErrorCode,
    /// A single human-readable sentence -- never a backend debug representation
    /// (§4.3: the AS-F022 cure).
    pub message: String,
    /// The code-specific detail map.
    pub details: serde_json::Value,
    /// Whether retrying could succeed, so an agent decides without pattern
    /// matching on prose.
    pub retryable: bool,
}

/// §4.3's closed seven-member error vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// An argument failed validation. Carried at protocol level (`-32602`).
    InvalidArgument,
    /// `q` did not parse.
    GrammarParse,
    /// `q` parsed, but names a clause this build cannot execute.
    GrammarUnsupported,
    /// An addressing argument names something that does not exist.
    UnknownReference,
    /// A write was refused by a gate.
    Refused,
    /// A backend leg required for this call is down.
    ///
    /// **Never raised by `status`**, and the exclusion is load-bearing (§4.3):
    /// `status` exists to answer when the backend is down, so a `status` that
    /// failed here could not report the one condition it exists to report.
    /// Unreachability is data, not an error.
    BackendUnavailable,
    /// An unexpected failure. Carries a correlation id and nothing else.
    Internal,
}
