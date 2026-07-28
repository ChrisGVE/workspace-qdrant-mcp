//! The deployment namespace (N8) -- the ONE knob, and the write refusal built on it.
//!
//! `PROJECT_LOGISTICS.md` decided on 2026-07-01 that v0.2 runs as a parallel `-v2`
//! system **alongside** production, never in place: production is the daily-driver
//! code-intelligence surface, so an in-place rebuild would remove it and make
//! regression judgement impossible.
//!
//! Two rules follow, and this module is where both are made true:
//!
//! 1. **One knob, not N hardcoded literals.** A single deployment suffix derives
//!    directory names, collection names and the service label. Cutover is then
//!    "flip the knob to empty and migrate", not a manual rename sweep.
//! 2. **The write-guard / blast-radius rule.** On a shared Qdrant, isolation is by
//!    collection NAME only -- so v0.2 may write only to `-v2` collections and must
//!    refuse any write to a non-`-v2` one. The migration is the single boundary
//!    crossing: it reads production read-only and writes `-v2`.
//!
//! # The knob
//!
//! [`DEPLOYMENT_SUFFIX`] is spelled exactly once, in the `deployment_suffix!`
//! macro below, because `concat!` needs a literal to build the deployed names at
//! compile time. Cutover is a one-token edit of that macro body to `""`.
//!
//! Ports are still NOT derived here, but no longer for the reason this comment
//! first gave. `P04-GT001-WO010` measured the checklist's premise -- "configurable
//! ⇒ the `-v2` config dir just declares different ones, no code change" -- against
//! v0.1 at ref `46f5df66b` and **refuted it**: the serving side is overridable, but
//! several consumers hardcode the address outright (`http://127.0.0.1:6337/metrics`
//! in two CLI/TUI sites, `DEFAULT_GRPC_ADDR = "127.0.0.1:50051"`), and every default
//! is re-spelled per site. Worse, two defaulted daemons collide on the control port
//! (`7799`) -- the bind whose entire purpose is single-writer mutual exclusion.
//!
//! So ports are a knob concern, not a config-file concern, and the *values* are
//! Chris's to set. This workspace serves nothing yet (all three bins decline), so
//! nothing is decided prematurely: the requirement and its evidence live in
//! `SCAFFOLD.md` §5, routed to N7 (`P04-GT013`), which is where the first port is
//! bound and therefore where the derivation must exist before it binds.
//!
//! # How the refusal is enforced (DP-8)
//!
//! DP-8 asks whether an invariant is prevented structurally or merely by
//! discipline. A free function `check_writable(name)` that a caller must remember
//! to call is runtime-only discipline -- one forgotten call and a slice writes to
//! production.
//!
//! So the refusal is **structural**, using the module-privacy funnel the
//! architecture already relies on for N59's storage lock: [`WriteTarget`] wraps a
//! private field and has no public constructor other than the two below, both of
//! which either produce a suffixed name or refuse. A write API typed as
//! `WriteTarget` therefore *cannot be handed* a bare production collection name --
//! not "must not be", cannot be. The check is unreachable to bypass from outside
//! this module.
//!
//! # Provisional home
//!
//! CHARTER §5A.3: the chokepoint and the refusal are harness concerns and belong
//! in `P04-GT001`. Whether the CONSTANT ultimately lives here or in the N7 config
//! nexus / `wqm-conventions` is `P04-GT002`'s to settle; it may re-home the value
//! without changing this module's contract.

use core::fmt;

use super::collections::Collection;

/// The deployment suffix, spelled ONCE. `concat!` requires a literal, which is
/// why this is a macro rather than a `const`. Cutover = change the body to `""`.
macro_rules! deployment_suffix {
    () => {
        "-v2"
    };
}

/// The suffix every deployed artefact name carries while v0.2 runs alongside
/// production. Empty after cutover.
pub const DEPLOYMENT_SUFFIX: &str = deployment_suffix!();

/// Whether this build is a parallel (`-v2`) deployment rather than the primary.
/// After cutover the suffix is empty and this is `false`, which is what turns the
/// write refusal below into a no-op without deleting the machinery.
pub const IS_PARALLEL_DEPLOYMENT: bool = !DEPLOYMENT_SUFFIX.is_empty();

/// The product's base name, spelled ONCE, for the same `concat!` reason as the
/// suffix above: every deployed name below is built from it at compile time.
macro_rules! dir_base {
    () => {
        "workspace-qdrant"
    };
}

/// The base name the XDG-style directories are derived from.
const DIR_BASE: &str = dir_base!();

/// The deployed config/cache/data directory name -- `workspace-qdrant-v2` while
/// running in parallel. N7 joins this to the platform's XDG roots.
pub const DEPLOYMENT_DIR: &str = concat!(dir_base!(), deployment_suffix!());

/// The deployed service label (launchd / systemd unit name), derived from the
/// same knob so the parallel daemon never collides with production's.
pub const SERVICE_LABEL: &str = concat!("com.", dir_base!(), ".memexd", deployment_suffix!());

impl Collection {
    /// The DEPLOYED collection name -- the logical name plus the deployment
    /// suffix. This is the name that reaches Qdrant.
    ///
    /// Prefer this over [`Collection::name`] everywhere a real collection is
    /// addressed; `name()` is the logical identity (what the collection *is*),
    /// this is where it lives in *this* deployment.
    pub const fn deployed_name(self) -> &'static str {
        match self {
            Collection::Projects => concat!("projects", deployment_suffix!()),
            Collection::Libraries => concat!("libraries", deployment_suffix!()),
            Collection::Rules => concat!("rules", deployment_suffix!()),
            Collection::Scratchpad => concat!("scratchpad", deployment_suffix!()),
        }
    }
}

/// Why a write target was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteGuardError {
    /// The name does not carry the deployment suffix, so it addresses a
    /// collection outside this deployment -- production, on a shared Qdrant.
    OutsideDeployment {
        /// The name that was refused.
        name: String,
        /// The suffix it lacked.
        required_suffix: &'static str,
    },
}

impl fmt::Display for WriteGuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteGuardError::OutsideDeployment {
                name,
                required_suffix,
            } => write!(
                f,
                "refusing to write to collection `{name}`: it does not carry the \
                 deployment suffix `{required_suffix}`, so it belongs to another \
                 deployment. v0.2 writes only to its own collections; reading \
                 across the boundary is the migration's job and is read-only."
            ),
        }
    }
}

impl core::error::Error for WriteGuardError {}

/// A collection name that has been **proven** to belong to this deployment.
///
/// The inner field is private and there is no other constructor, so a value of
/// this type cannot exist for a collection outside the deployment. A write path
/// typed as `WriteTarget` is therefore structurally incapable of addressing
/// production (DP-8: prevention, not discipline).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteTarget(&'static str);

impl WriteTarget {
    /// The infallible path: a canonical collection is always addressable, and
    /// always through its deployed name.
    pub const fn of(collection: Collection) -> Self {
        WriteTarget(collection.deployed_name())
    }

    /// The fallible path, for a name that arrived as data (config, a request, a
    /// migration manifest) rather than as a [`Collection`].
    ///
    /// Refuses anything without the deployment suffix. When the suffix is empty
    /// (post-cutover) every name is in-deployment and this always succeeds --
    /// the guard retires with the parallel period rather than being deleted.
    pub fn try_from_name(name: &str) -> Result<Self, WriteGuardError> {
        if !name.ends_with(DEPLOYMENT_SUFFIX) {
            return Err(WriteGuardError::OutsideDeployment {
                name: name.to_owned(),
                required_suffix: DEPLOYMENT_SUFFIX,
            });
        }
        // Resolve back to a canonical collection so the stored value stays
        // 'static and cannot drift from the registry.
        for c in Collection::ALL {
            if c.deployed_name() == name {
                return Ok(WriteTarget(c.deployed_name()));
            }
        }
        Err(WriteGuardError::OutsideDeployment {
            name: name.to_owned(),
            required_suffix: DEPLOYMENT_SUFFIX,
        })
    }

    /// The wire name to send to Qdrant.
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

impl fmt::Display for WriteTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// The directory base, exposed so N7 can assert it derives from the same knob
/// rather than re-spelling it.
pub const fn dir_base() -> &'static str {
    DIR_BASE
}
