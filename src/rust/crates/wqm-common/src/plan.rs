//! N56's plan, as the response carries it back (MCP-SURFACE.md §3.4).
//!
//! # Why the types are here and not in the planner
//!
//! `CONTRACTS.md`:2346 puts the plan TYPES in `wqm-common` and lets the planner
//! run wherever the executor runs. The reason is visible the moment a client
//! embeds the read pipeline: the bin serializes a plan it did not build, so the
//! shape has to be reachable from the floor rather than from the kernel crate that
//! happens to construct it.
//!
//! # This is the ECHO side
//!
//! `plan@v1` (the input override) and `plan_echo@v1` (what comes back) are sibling
//! schemas, not one schema with optional keys -- §3.4 is explicit that the "one
//! schema plus three annotations" reading fails under `additionalProperties:false`.
//! What is declared here is the **echo**: every one of the ten keys present on
//! every response, `mode`/`object` server-set, and no key omitted. The override
//! side has no type yet because this build accepts no `plan` parameter (§2.2's
//! schema in this build carries `q` and `limit` only).
//!
//! # The closed vocabularies are declared in full
//!
//! [`LegMethod`], [`ObjectKind`] and [`SourceKind`] carry every member the sealed
//! algebra names, including members this build can never emit. That is the
//! [`crate::envelope`] precedent and the same argument holds: these are closed sets
//! in a sealed document, so declaring them completely means a later slice adds a
//! *call site* rather than a string.

use serde::Serialize;

/// The retrieval regime (§1.6: `SEMANTIC` default, `TEXT` literal, `REGEX`).
/// Server-set on the echo, and rejected on input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Hybrid dense + sparse retrieval.
    Semantic,
    /// Literal substring / phrase matching.
    Text,
    /// Regular-expression matching.
    Regex,
}

/// What a result *is* (§3.4's `object`). Server-set on the echo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    /// A retrieval chunk of a document.
    Chunk,
    /// One file line, the grep-shaped result row.
    Line,
    /// A whole document.
    Document,
    /// A code symbol.
    Symbol,
    /// One behavioural rule.
    Rule,
    /// One scratchpad note.
    Note,
    /// A tag.
    Tag,
    /// A graph relation.
    Relation,
    /// A classification topic.
    Topic,
}

/// Where a plan reads from (§2.2a's `source.kind`).
///
/// Two members -- [`SourceKind::Rules`] and [`SourceKind::Scratchpad`] -- are
/// spelled identically to N8 collection names, and that agreement is asserted by a
/// test rather than left to the eye: one vocabulary rendered twice is exactly the
/// pair that drifts. The literals themselves are produced by serde's rename
/// derivation, so this module re-spells nothing N8 owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// One project.
    Project,
    /// A resolved group of projects.
    Group,
    /// One reference library.
    Library,
    /// The behavioural-rules collection.
    Rules,
    /// The scratchpad collection.
    Scratchpad,
    /// The code-relationship graph.
    Graph,
    /// Every canonical collection.
    All,
}

/// One entry of the plan's `sources` array.
#[derive(Debug, Clone, Serialize)]
pub struct Source {
    /// Which kind of source this is.
    pub kind: SourceKind,
    /// Its name, or `null` where the kind has no name (`rules`, `scratchpad`).
    pub name: Option<String>,
}

/// A retrieval leg's method -- the closed five of §2.2a's `leg.method`.
///
/// [`LegMethod::Trigram`] is the FTS5 literal-text leg. The name is the sealed
/// one, and this workspace's FTS5 table is declared with SQLite's `trigram`
/// tokenizer so the declaration and the index agree; a plan announcing `trigram`
/// over a word-token index would be the "declared feature that is something else"
/// class in the one field an agent reads to learn what ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegMethod {
    /// Dense vector retrieval.
    Dense,
    /// Sparse (BM25/IDF) vector retrieval.
    Sparse,
    /// FTS5 trigram literal matching.
    Trigram,
    /// Regular-expression scanning.
    Regex,
    /// Graph traversal.
    Graph,
}

/// A leg's parameters. All four keys are present on the echo, `null` where the
/// method does not use them.
#[derive(Debug, Clone, Serialize)]
pub struct LegParams {
    /// Embedding model, for the vector legs.
    pub model: Option<String>,
    /// Index name, for the lexical legs.
    pub index: Option<String>,
    /// Traversal depth, for the graph leg.
    pub hops: Option<u32>,
    /// Candidate-set bound.
    pub candidates: Option<u32>,
}

impl LegParams {
    /// Params with every slot empty -- the base a method fills only what it uses.
    pub const fn none() -> Self {
        LegParams {
            model: None,
            index: None,
            hops: None,
            candidates: None,
        }
    }
}

/// One retrieval leg of the plan.
#[derive(Debug, Clone, Serialize)]
pub struct Leg {
    /// How this leg retrieves.
    pub method: LegMethod,
    /// What it was parameterized with.
    pub params: LegParams,
}

/// One applied predicate (§3.4's `filter_echo`).
#[derive(Debug, Clone, Serialize)]
pub struct Filter {
    /// The field the predicate is over.
    pub field: String,
    /// The operator, in the surface's spelling.
    pub op: String,
    /// The compared value.
    pub value: serde_json::Value,
    /// Whether the predicate was negated.
    pub negate: bool,
    /// Server-set: whether the engine actually applied it.
    pub applied: bool,
}

/// The fusion step (§3.4's `fuse_echo`). Present only when there is more than one
/// leg to fuse -- see [`Plan::fuse`].
#[derive(Debug, Clone, Serialize)]
pub struct Fuse {
    /// The fusion method.
    pub method: String,
    /// RRF's `k` constant.
    pub k: Option<u32>,
    /// Per-leg weights.
    pub weights: Option<serde_json::Value>,
}

/// The executed plan, as §3.4 returns it: ten keys, every one present.
///
/// The struct has no optional *keys* -- `fuse`, `expand` and `rerank` are
/// `Option` **values** that serialize to `null`, which is the surface's stated
/// convention ("present and null … never absent"). Rust's `Option` and JSON's
/// `null` line up here exactly, so the shape cannot lose a key by construction.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    /// Server-set retrieval regime.
    pub mode: Mode,
    /// Server-set result object kind.
    pub object: ObjectKind,
    /// What was read from.
    pub sources: Vec<Source>,
    /// Whether recall broadening was refused.
    pub strict: bool,
    /// The legs that ran.
    pub legs: Vec<Leg>,
    /// The predicates, each with its server-set `applied` flag.
    pub filters: Vec<Filter>,
    /// The fusion step, or `null`.
    ///
    /// **`null` on a single-leg plan is structural, not an omission.** N4's
    /// contract takes `legs: Vec<RankedList>` and requires N17 dense scores
    /// (`CONTRACTS.md`:1012-1014), so one leg never reaches fusion at all. A
    /// single-leg plan that announced `{"method":"rrf"}` would be describing a
    /// step that did not happen.
    pub fuse: Option<Fuse>,
    /// The graph-expansion step, or `null`.
    pub expand: Option<serde_json::Value>,
    /// The rerank step, or `null`.
    pub rerank: Option<serde_json::Value>,
    /// The result bound that was in force.
    pub limit: u32,
}
