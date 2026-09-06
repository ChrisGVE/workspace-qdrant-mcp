//! `Secret` — the one in-memory value type a resolved credential is carried in.
//!
//! Transcribed from the sealed fill fragment
//! `P04-build/T4-glue/fill/anchor/A-secret.md` (`P04-GT002-WO075`, concrete
//! `C-ty-secret`). Vocabulary only: nothing here reads an environment variable,
//! opens a keystore, or resolves anything.
//!
//! # Why the floor crate holds it
//!
//! Both the kernel (N7 config, N17 the embed provider) and the glue (N26 the
//! resolver, N44 the auth header, N46 provisioning, N49 the conn-token) name this
//! type. Homing it in `wqm-common` — the bottom of the DAG — lets all six share
//! ONE declaration with no crate cycle, while the resolver that MINTS a `Secret`
//! stays in `wqm-secrets`. That is FP-2 made structural: there is no second
//! credential type to drift against.
//!
//! # In-memory altitude, and what that forbids
//!
//! A `Secret` never serializes — not to disk, a log, a metric, or the wire. That
//! is why this module derives no `Serialize`, no `PartialEq`, no `Deref`, no
//! `AsRef<[u8]>`, and no `From<String>`. Each of those is a read path, and every
//! read path is a place credential bytes can leave without anyone naming it.
//! [`Secret::expose`] is the single escape, and it is *named* that way so a CI
//! sweep can grep its callers and bound exactly where raw bytes flow.
//! At-rest storage of the underlying credential is N46's provisioning concern;
//! this type carries no store handle (contracts N26: "never returned to the
//! kernel as a store handle -- only the resolved value crosses the edge").
//!
//! Sources: contracts N26 *Provides*; ARCH §9.1 (the `Secret` TYPE homed in
//! `wqm-common`, closure `wqm-secrets -> {wqm-common}`); PRD F-08 (AC1 redact +
//! zeroize, FP-2 single home); Design_Principles I7 + FP-2. Owner nexus N26; the
//! law it makes concrete is `A-inv-secret-redaction`.

use core::fmt;

use zeroize::{ZeroizeOnDrop, Zeroizing};

/// What `Debug` writes in place of the bytes.
///
/// A constant rather than an inline literal because the tests assert the exact
/// spelling the fragment gives, and a redaction token that two sites spell
/// differently is a redaction one of them can lose.
const DEBUG_REDACTION: &str = "Secret(<redacted>)";

/// What `Display` writes in place of the bytes.
const DISPLAY_REDACTION: &str = "<redacted>";

/// A resolved credential value -- zeroized on drop, redacted in `Debug` and
/// `Display`.
///
/// Carries a VALUE, never a store handle or a live resolver: it is moved and
/// injected across the kernel edge (R7), not held open. It has no concurrency
/// story of its own -- an immutable value, shared by `clone` or by an `Arc` at
/// the injection site when a surface needs it in several tasks. Any failure to
/// obtain one is the resolver's typed N9 error, not a `Secret` state.
///
/// The backing bytes are wiped when the value drops, which is the I7 property
/// travelling WITH the value instead of being re-applied at each site:
///
/// ```
/// use wqm_common::secret::Secret;
///
/// let key = Secret::new(b"hunter2".to_vec());
/// assert_eq!(key.expose(), b"hunter2");
/// assert_eq!(format!("{key:?}"), "Secret(<redacted>)");
/// assert_eq!(format!("{key}"), "<redacted>");
/// ```
///
/// `expose()` is the only read path -- the bytes are not reachable by indexing
/// or by deref, so a leak cannot happen without the greppable name appearing:
///
/// ```compile_fail
/// use wqm_common::secret::Secret;
///
/// let key = Secret::new(b"hunter2".to_vec());
/// let _leaked = key[0]; // no Index, no Deref: `Secret` cannot be indexed
/// ```
#[derive(Clone)]
pub struct Secret(Zeroizing<Vec<u8>>);

impl Secret {
    /// Wrap resolved credential bytes.
    ///
    /// By convention this is called by N26's resolver chain
    /// (`resolve(SecretId) -> Result<Secret, N9>`); the constructor is public
    /// because `wqm-secrets` lives in another crate, not because arbitrary
    /// callers should be minting credentials.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(Zeroizing::new(bytes))
    }

    /// The ONLY read path to the credential bytes.
    ///
    /// Named `expose` rather than `as_bytes` or `get` so that every site where
    /// raw credential bytes escape the newtype is enumerable by a single grep --
    /// which is what bounds the blast radius of I7 to a reviewable list.
    #[must_use]
    pub fn expose(&self) -> &[u8] {
        &self.0
    }
}

/// Wiped on drop by the field's own type; the marker states the property so it
/// can be named in a bound rather than only described in prose.
impl ZeroizeOnDrop for Secret {}

/// Never the inner bytes -- I7 holds at every formatting site, including the
/// derived `Debug` of any struct that happens to contain a `Secret`.
impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(DEBUG_REDACTION)
    }
}

/// Redacted for the same reason as `Debug`: a credential interpolated into a
/// message is a credential in a log.
impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(DISPLAY_REDACTION)
    }
}
