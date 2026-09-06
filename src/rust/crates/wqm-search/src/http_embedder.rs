//! `C-adp-N17-http` -- the ONE HTTP embedding provider, declared.
//!
//! Transcribed from the sealed fill fragment
//! `P04-build/T1-kernel/fill/anchor/Alg-N17-http.md` (`P04-GT002-WO001`).
//! Vocabulary only: nothing here opens a socket, signs a request, parses a
//! response, or classifies an error.
//!
//! # The connection-model insight, which is why there is only one of these
//!
//! NEXUSES Appendix A.2 collapses "local server" and "remote vendor" into the
//! same implementation: an embeddings endpoint is an HTTP endpoint whatever is
//! behind it, so **one provider parameterized by base URL + protocol dialect**
//! covers Ollama, llama.cpp's `llama-server`, LM Studio, HF TEI, OpenAI, Azure
//! and the rest. Without that collapse the backend variety ARCH §7 commits to
//! (Chris-locked S2.19) would cost a crate per vendor. The dialect is therefore
//! **data on this struct, not a control-flow fork in any caller** -- which is
//! the property [`EmbeddingDialect`] exists to make expressible.
//!
//! # Why it is the floor of the provider family
//!
//! Pure Rust, no native library, no feature gate: this adapter is compiled into
//! every build and is the guaranteed floor beneath the native and proxy siblings
//! (Appendix A.2). ARCH §9.1 homes it in `wqm-search` alongside the port it will
//! satisfy (`Alg-N17-http`: "struct in `wqm-search`, day-1").
//!
//! # What is ABSENT here, rather than stubbed
//!
//! `P04-GT002` is the shared kernel: it declares types and the invariants a type
//! can carry. Everything that *acts* arrives with N17's own slice, and none of
//! it is scaffolded here --
//!
//! - **The `Embedder` impl.** `embed`, `output_dim`, `probe`,
//!   `max_input_bytes` and `metrics_label` (`A-if-N17`) are the port's five
//!   contract methods; this struct implements none of them, because a method
//!   that returns a placeholder is worse than a method that does not exist --
//!   the first compiles at every call site.
//! - **The HTTP client dependency.** §9.1 records that `wqm-search` will carry
//!   one. It is not added at this work order: a declaration needs no transport,
//!   and the acceptance test that `wqm-search`'s default closure contains no
//!   `ort`/`fastembed` is cheapest to keep true by not linking anything yet.
//! - **`max_input_bytes`.** The value is finite and per-endpoint, and it drives
//!   upstream chunk splitting -- so it is a fact about a live backend, obtained
//!   by the slice that can talk to one, not a constant to guess here.
//! - **Batching under the I4 byte budget**, and the N9 error classification
//!   (`A-error-taxonomy`) that decides retryable from permanent.
//! - **Dialect wire shapes.** Which JSON body each dialect sends, where its
//!   auth goes, how its response parses: contract-test material for N17's
//!   slice, one mock server per dialect (`Alg-N17-http` Acceptance).
//!
//! Sources: contracts N17 (§1.2 adapter list, the HTTP provider row; Provides
//! invariants Sc-59/Sc-63/I4/I7/F16; Supplied-by-N28 `Secret` injection;
//! config-selected); NEXUSES Appendix A.1 (the v0.1 `OpenAiCompatibleProvider`
//! precedent, factory dispatch with typed `InitializationError`, the
//! `metrics_label` vocabulary, the finite-`max_input_bytes` rationale) and A.2
//! (the connection-model table); ARCH §4.2 (the external-embedder branch --
//! in-process direct, nothing to warm), §6.2 N17 r18 + §7 (the day-one dialect
//! set, Chris-locked S2.19), §9.1 (`wqm-search` row and its HTTP client dep).

use wqm_common::secret::Secret;

/// The wire protocol an embeddings endpoint speaks.
///
/// **This is the day-one set, and it is CLOSED by decision** -- ARCH §7's
/// day-one-providers row, Chris-locked at S2.19. Four vendors ship APIs whose
/// request and response shapes are not OpenAI-shaped, and everything else in the
/// family is reached through the OpenAI-compatible dialect at a different base
/// URL. So the axis has exactly five values, and adding a sixth is a design
/// decision that reopens S2.19 rather than a mechanical extension.
///
/// That is why there is deliberately **no `#[non_exhaustive]`**. The attribute
/// says "expect more variants, handle the ones you do not know" -- the opposite
/// of what S2.19 decided, and it would force a wildcard arm into every
/// downstream `match`, which is precisely where a newly-added dialect would then
/// be silently swallowed. The closure is asserted from another crate, where an
/// exhaustive `match` without a wildcard only compiles if the enum is open to
/// no one (`tests/http_embedder.rs`).
///
/// **It carries no strings.** The `metrics_label` vocabulary the fragment
/// names (`openai`, `azure_openai`, `lmstudio`, ...) is an observability label
/// set, not a member of this enum: labels are finer than dialects (three of
/// those four speak the same dialect at different base URLs) and literal
/// identifier strings are N8's single-producer remit (`wqm-common::names`).
/// This type declares *which protocols exist*; what a metric calls one arrives
/// with `metrics_label()`, in N17's slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingDialect {
    /// The `/v1/embeddings`-shaped request-response used by OpenAI itself and by
    /// every server that emulates it -- Azure OpenAI, Ollama, `llama-server`,
    /// LM Studio, HF TEI. The common case: these differ by base URL and by auth
    /// placement, not by wire shape.
    OpenAiCompatible,
    /// Cohere's embed API. Distinct wire shape, not OpenAI-emulating.
    Cohere,
    /// Google Gemini's embed API. Distinct wire shape.
    Gemini,
    /// Voyage AI's embed API. Distinct wire shape.
    Voyage,
    /// Amazon Bedrock's runtime invoke API. Distinct wire shape, and distinct
    /// auth: SigV4 rather than a bearer token. The injected credential is still
    /// a [`Secret`] -- how each dialect places it on the wire is N17's slice.
    Bedrock,
}

impl EmbeddingDialect {
    /// Every day-one dialect, in declaration order.
    ///
    /// Lets a consumer -- and the closure test -- enumerate the set without
    /// re-spelling it, the same shape `wqm_common::names::Collection::ALL` uses
    /// for the other closed vocabulary in this workspace.
    pub const ALL: [EmbeddingDialect; 5] = [
        EmbeddingDialect::OpenAiCompatible,
        EmbeddingDialect::Cohere,
        EmbeddingDialect::Gemini,
        EmbeddingDialect::Voyage,
        EmbeddingDialect::Bedrock,
    ];
}

/// An HTTP embedding provider, bound at construction to one endpoint, one
/// dialect, one model and one credential.
///
/// # Model binding is per-INSTANCE
///
/// `A-if-N17` fixes this: an `Embedder` is bound to one model when it is built,
/// so the port carries no model parameter and the caller obtains the right
/// instance instead (N45's `select_query_model` at query time, N28's wiring at
/// ingest time). The `model` field is that binding, which is why it is a
/// constructor argument and not a method argument.
///
/// # The credential arrives injected, and is never fetched
///
/// R7 (dependency inversion): **N26 resolves, N28 injects, this adapter
/// receives.** Nothing in this type reads `env::var`, opens a keystore, or holds
/// a resolver -- only the resolved VALUE crosses the edge. That is what makes
/// the secret's provenance a build-time fact about the composition root rather
/// than a runtime behaviour hidden inside an adapter.
///
/// Secret discipline is I7/F16: the key exists only as the injected [`Secret`],
/// and it must never reach `metrics_label()`, an error, a log line, or a
/// serialized request beyond the auth header itself. Two structural properties
/// carry that here, rather than a convention anyone has to remember --
///
/// - the field is private and there is no accessor returning bytes, so the only
///   way to raw credential bytes is [`Secret::expose`], whose name a single grep
///   enumerates;
/// - the derived `Debug` cannot leak, because `Secret`'s own `Debug` redacts.
///   The unit test plants a marker key and looks for the marker, not for the
///   redaction token.
///
/// ```
/// use wqm_common::secret::Secret;
/// use wqm_search::http_embedder::{EmbeddingDialect, HttpEmbedder};
///
/// let adapter = HttpEmbedder::new(
///     "https://api.openai.com/v1".to_owned(),
///     EmbeddingDialect::OpenAiCompatible,
///     "text-embedding-3-small".to_owned(),
///     Secret::new(b"sk-planted".to_vec()),
/// );
///
/// assert_eq!(adapter.base_url(), "https://api.openai.com/v1");
/// assert!(!format!("{adapter:?}").contains("sk-planted"));
/// ```
///
/// # No `Default`, and that is the point
///
/// CR-007's first defect is a default live endpoint: a type that can be built
/// with no arguments is a process that can talk to somewhere nobody named. Every
/// field here is supplied by the composition root or the value does not exist.
/// The absence is asserted rather than merely intended:
///
/// ```compile_fail
/// // CR-007: no default endpoint, no default model, no default credential.
/// // If this ever compiles, a `Default` impl has been added and an unnamed
/// // endpoint became reachable.
/// let _ = wqm_search::http_embedder::HttpEmbedder::default();
/// ```
///
/// The credential likewise cannot be read back out as bytes -- [`secret`] hands
/// out a `&Secret`, which redacts at every formatting site, and not a slice:
///
/// [`secret`]: HttpEmbedder::secret
///
/// ```compile_fail
/// use wqm_common::secret::Secret;
/// use wqm_search::http_embedder::{EmbeddingDialect, HttpEmbedder};
///
/// let adapter = HttpEmbedder::new(
///     "http://localhost:11434/v1".to_owned(),
///     EmbeddingDialect::OpenAiCompatible,
///     "nomic-embed-text".to_owned(),
///     Secret::new(b"sk-planted".to_vec()),
/// );
/// // The field is private, so there is no field access at all; I7's escape
/// // is named `expose` and reached through `secret()`. No type annotation
/// // here on purpose -- this line must fail for privacy and nothing else.
/// let _leaked = adapter.secret;
/// ```
#[derive(Debug)]
pub struct HttpEmbedder {
    /// The endpoint's base URL, stored verbatim.
    ///
    /// A `String` rather than a parsed URL type **because no invariant on this
    /// adapter depends on the URL's structure.** `TransportAddress` types its
    /// TCP payload as a `SocketAddr` for the opposite reason -- N48 must ask
    /// "is this loopback?" -- and there is no equivalent question here: nothing
    /// in `Alg-N17-http`'s Invariants inspects scheme, host or path. Whether a
    /// malformed URL is refused at construction or surfaces as a typed
    /// unreachable from `probe()` is N17's slice to decide, and typing it now
    /// would pre-empt that decision with a dependency.
    base_url: String,
    /// Which wire protocol this endpoint speaks.
    dialect: EmbeddingDialect,
    /// The model this instance is bound to, in the endpoint's own spelling.
    model: String,
    /// The N28-injected credential, owned.
    ///
    /// Owned rather than borrowed: an adapter outlives the wiring that built it
    /// and is shared across tasks, so a lifetime here would tie every surface to
    /// the composition root's stack frame.
    secret: Secret,
}

impl HttpEmbedder {
    /// Bind an endpoint, a dialect, a model and a credential into one provider.
    ///
    /// **This constructor stores and does nothing else.** No URL validation, no
    /// reachability check, no credential check -- `probe()` is the port's
    /// declared place for reachability and credentials, returning typed N9
    /// failures (unreachable, unauthorized, wrong model), and it arrives with
    /// N17's slice. Doing any of it here would put a network call inside a
    /// constructor, where a caller has no typed error to receive it in.
    #[must_use]
    pub fn new(
        base_url: String,
        dialect: EmbeddingDialect,
        model: String,
        secret: Secret,
    ) -> Self {
        Self {
            base_url,
            dialect,
            model,
            secret,
        }
    }

    /// The endpoint this instance was bound to.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The wire protocol this instance was bound to.
    #[must_use]
    pub fn dialect(&self) -> EmbeddingDialect {
        self.dialect
    }

    /// The model this instance was bound to.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The injected credential, still inside its newtype.
    ///
    /// Returns `&Secret` and never `&[u8]`: handing out the newtype preserves
    /// redaction at every formatting site downstream and keeps
    /// [`Secret::expose`] the single greppable escape (I7). The auth-header
    /// construction that will call `expose` belongs to N17's slice.
    #[must_use]
    pub fn secret(&self) -> &Secret {
        &self.secret
    }
}
