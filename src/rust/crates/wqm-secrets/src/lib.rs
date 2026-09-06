//! wqm-secrets -- N26 credential resolution.
//!
//! N26 is the nexus that answers one question for the whole system: *given the
//! identity of a credential, where do its bytes come from?* Today the answer is
//! scattered -- each surface reads its own `env::var` -- and the greenfield build
//! consolidates it behind ONE port, `SecretResolver` (`A-if-N26`), with a small
//! set of interchangeable adapters selected by strategy-dispatch: the process
//! environment, the OS keychain, and an environment file.
//!
//! What this crate will own: the port trait, its adapters, and the N9 decline
//! classification a resolution failure maps to. What it deliberately does NOT own
//! is the `Secret` value type itself -- that is `A-secret`, homed in `wqm-common`,
//! because every crate that *carries* a resolved credential needs the zeroizing
//! newtype while only this crate needs the machinery that produces one.
//!
//! ARCH rev15 §9.1 places the crate at stratum 1 with a wqm-closure of exactly
//! `{wqm-common}`.
//!
//! # What ships today
//!
//! `P04-GT002-WO057` declares this crate and its first row, `C-adp-N26-env`: the
//! [`EnvResolver`](env::EnvResolver) type in [`mod@env`]. Vocabulary only -- zero
//! behavior. The port trait `A-if-N26` is `P04-GT002-WO068` and is outside this
//! GT's scope, and by the charter (§2.2) the adapters therefore ship WITHOUT their
//! `impl SecretResolver` blocks: `SecretId` is declared nowhere in the mesh yet,
//! and an `impl` is behavior. The struct is the declaration; the behavior follows
//! its port.
//!
//! `mesh/program.db` is the tracker.

pub mod env;
