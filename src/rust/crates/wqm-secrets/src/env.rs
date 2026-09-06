//! The environment-variable adapter of N26's `SecretResolver` port
//! (`C-adp-N26-env`).
//!
//! Fill fragment:
//! `project-notes/wqm-0.2/P04-build/T4-glue/fill/anchor/Alg-N26-env.md`.

/// Resolves a credential by reading the process environment.
///
/// # Why it exists
///
/// It is the day-1 source for `QDRANT_API_KEY` -- the Qdrant Cloud credential --
/// and the fallback path for keys no keystore holds. Its purpose is
/// CONSOLIDATION: today each surface performs its own `env::var` read, so there
/// is no one place where an env-sourced credential can be redacted, zeroized, or
/// even counted. This adapter replaces those scattered reads with a single
/// resolver sitting behind the port, so the environment becomes one source among
/// several rather than a habit spread across the codebase.
///
/// # Carrier
///
/// A UNIT struct, and the choice is not cosmetic. The fragment writes the body as
/// `pub struct EnvResolver { /* */ }` -- deliberately empty, because the adapter
/// holds no state: it reads `std::env` at call time, so there is nothing to
/// configure, cache, or hand in. A unit struct says exactly that, and it is
/// constructible as the bare expression `EnvResolver` -- which is what N28's
/// wiring needs when it assembles the resolver chain, with no constructor to call
/// and no builder to keep in step. An empty braced struct would carry the same
/// (zero) state while requiring `EnvResolver {}` at every construction site and
/// leaving a brace pair that invites a future field to be dropped in without the
/// statelessness above being re-argued.
///
/// # Invariants
///
/// These are the fragment's, recorded here because the type is where a reader
/// meets them; they become enforceable when the `impl` lands.
///
/// - Resolution yields a `Secret` VALUE (`A-secret`, homed in `wqm-common`): the
///   raw `String` handed back by `env::var` is moved into the zeroizing newtype
///   immediately, so no plaintext copy lingers.
/// - The env-key SPELLING is N8-owned, never inlined here: the name comes from
///   `EnvVar::QdrantApiKey` in `wqm-common/src/names/env.rs`, whose `key()`
///   returns `"QDRANT_API_KEY"`. Renaming it at N8 must recompile with no edit in
///   this file. (`wqm-common` is not yet a declared dependency of this crate --
///   it arrives with the first row that names a type from it; see this crate's
///   `Cargo.toml`.)
/// - This adapter is the ONLY sanctioned `env::var` read for a credential; a
///   planted read elsewhere is a CI-guard failure (PRD F-33).
/// - It is interchangeable behind the port with the Keychain and
///   EnvironmentFile adapters (`selection = strategy-dispatch`), so nothing may
///   depend on *which* adapter answered.
/// - The environment is a resolution SOURCE, not a config-file plaintext default
///   for keystore-resident keys: the embedder key's at-rest home is the keystore
///   (SEC-3).
///
/// # What is missing, and why
///
/// There is no `impl SecretResolver for EnvResolver` here. The port trait
/// `A-if-N26` is `P04-GT002-WO068`, outside this GT's scope, and `SecretId` --
/// the argument `resolve` takes -- is declared nowhere in the mesh yet. The
/// charter (§2.2) therefore ships this crate's structs as vocabulary alone: an
/// `impl` is behavior, and behavior written against an undeclared type would be
/// invention rather than transcription. The mapping the fragment specifies --
/// `id -> N8 env-key -> std::env::var`, a present var becoming a `Secret` and a
/// missing or non-Unicode one a typed N9 decline that lets N28's chain fall
/// through -- arrives with WO068.
///
/// # Sources
///
/// - contracts N26 (env adapter; strategy-dispatch; requires N8; build path
///   greenfield) and the N9 typed-decline classification in the same document.
/// - ARCH §6.2, the N26 row (`QDRANT_API_KEY`; never an env-var default for
///   keystore keys).
/// - PRD F-33 (one path; redaction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvResolver;
