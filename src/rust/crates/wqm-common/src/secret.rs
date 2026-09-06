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
//! # The absences, held by the compiler rather than by the paragraph above
//!
//! A derive added in a later slice would satisfy that prose and break the
//! invariant silently, so each named absence has a doctest that must fail to
//! compile. Each is written as a trait bound rather than a use site, because a
//! bound is decided by the impl set alone -- which is the thing being asserted.
//!
//! Every block carries its expected error code (`compile_fail,E0277`), and what
//! that code is worth has to be stated exactly, because it is easy to overtrust:
//! a bare `compile_fail` passes on ANY error, including a mistyped path or a
//! renamed module, so the code says which failure was intended. **rustdoc does
//! not verify it.** Measured on rustc 1.98.0, stable and nightly alike: a block
//! pinned to a code the snippet does not raise -- even a code that does not
//! exist -- still passes. The pin is therefore a review anchor and the record of
//! what was proven, not a gate. The proof itself is out-of-band: each snippet was
//! compiled on its own against this crate and confirmed to raise that code and
//! nothing else, which is the step a reviewer must repeat when one of these is
//! edited.
//!
//! No `Serialize` -- the read path onto disk, a log, a metric, or the wire:
//!
//! ```compile_fail,E0277
//! use wqm_common::secret::Secret;
//!
//! fn assert_serialize<T: serde::Serialize>() {}
//! assert_serialize::<Secret>();
//! ```
//!
//! No `PartialEq` -- comparison reads the bytes, one answer at a time:
//!
//! ```compile_fail,E0277
//! use wqm_common::secret::Secret;
//!
//! fn assert_partial_eq<T: PartialEq>() {}
//! assert_partial_eq::<Secret>();
//! ```
//!
//! No `Deref` -- coercion would hand out the bytes implicitly, at any site
//! expecting a slice, with no named call to grep:
//!
//! ```compile_fail,E0277
//! use wqm_common::secret::Secret;
//!
//! fn assert_deref<T: core::ops::Deref>() {}
//! assert_deref::<Secret>();
//! ```
//!
//! No `AsRef<[u8]>` -- the same unnamed read path, reached through the generic
//! parameter most byte-taking APIs advertise:
//!
//! ```compile_fail,E0277
//! use wqm_common::secret::Secret;
//!
//! fn assert_as_ref<T: AsRef<[u8]>>() {}
//! assert_as_ref::<Secret>();
//! ```
//!
//! No `From<String>` -- an infallible `.into()` is a mint with no named
//! construction site to review:
//!
//! ```compile_fail,E0277
//! use wqm_common::secret::Secret;
//!
//! fn assert_from_string<T: From<String>>() {}
//! assert_from_string::<Secret>();
//! ```
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
/// ```compile_fail,E0608
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
///
/// `ZeroizeOnDrop` is an EMPTY marker trait, so writing this impl by hand asserts
/// nothing on its own -- it would keep compiling, and every `T: ZeroizeOnDrop`
/// bound on `Secret` would keep passing, if the field below became a bare
/// `Vec<u8>` and stopped wiping. The `const` block underneath is what makes the
/// impl honest: it names the FIELD, so the property this marker advertises is the
/// property the field actually has.
impl ZeroizeOnDrop for Secret {}

/// The honesty check for the impl above. Never executed: a function body is
/// type-checked whether or not anything calls it, and that type-check IS the
/// assertion. `&secret.0` has whatever type the field has, so replacing
/// `Zeroizing<Vec<u8>>` with a type that does not wipe fails the bound (E0277)
/// and the crate stops building.
const _: fn(&Secret) = |secret| {
    fn assert_zeroize_on_drop<T: ZeroizeOnDrop>(_: &T) {}

    assert_zeroize_on_drop(&secret.0);
};

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
