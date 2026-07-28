//! CR-007 defect (c): a test must establish its own environment, not inherit it.
//! **STRUCTURAL.**
//!
//! The cure is that this file never reads the ambient environment. There is no
//! `std::env::var` here, so there is no path by which an exported `XDG_*`,
//! `QDRANT_URL`, `WQM_LOG_DIR` or `OTEL_*` can reach a value a test observes. Every
//! sensitive key is *written* from a directory this type owns — written
//! unconditionally, not written-if-absent, which is the difference between
//! overriding an inherited value and consulting it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use wqm_common::names::EnvVar;

/// Keys this harness overrides that N8 does NOT own.
///
/// The configuration keys proper come from `EnvVar::ALL` below rather than being
/// re-spelled here — N8 is their single producer, and `guard_name_registry.py`
/// caught the first draft doing exactly what FP-2 forbids. Sourcing them has a
/// second payoff: a key added to N8 is overridden by this harness automatically,
/// with no edit here to forget.
///
/// These remaining families are process-environment conventions, not wqm
/// configuration, so N8 has no row for them. `P00-GT002` §4.3 measured seven tests
/// that "encode the machine they were written on"; CR-007 §1a records that two of
/// the seven failed for a different reason (a sandboxed `HOME` hid the FastEmbed
/// cache), so this is the set the defect actually names.
const NON_REGISTRY: &[&str] = &[
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    "WQM_LOG_DIR",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "OTEL_TRACES_EXPORTER",
];

/// Every key the harness takes responsibility for: N8's registry plus the
/// process-environment families above.
fn overridden_keys_owned() -> Vec<&'static str> {
    EnvVar::ALL
        .iter()
        .map(|v| v.key())
        .chain(NON_REGISTRY.iter().copied())
        .collect()
}

/// An environment a test owns outright.
///
/// Holds a temporary root and the exact key/value set derived from it. Consumers ask
/// this type what the environment *is*; they cannot ask it what the surrounding
/// shell said, because it never looked.
#[derive(Debug)]
pub struct HermeticEnv {
    root: PathBuf,
    vars: BTreeMap<String, String>,
    /// Dropping this removes the temporary tree.
    _guard: TempRoot,
}

impl HermeticEnv {
    /// Build an environment rooted in a fresh temporary directory.
    ///
    /// `label` only makes the directory recognisable while a test is running; it does
    /// not participate in isolation, which comes from the unique suffix.
    pub fn new(label: &str) -> std::io::Result<Self> {
        let guard = TempRoot::create(label)?;
        let root = guard.path().to_path_buf();

        let db_key = EnvVar::DatabasePath.key();
        let mut vars = BTreeMap::new();
        for key in overridden_keys_owned() {
            // Classified by SHAPE rather than by name, so adding a key to N8 does not
            // require a new match arm here.
            let value = if key.starts_with("XDG_") {
                dir(&root, &key.to_ascii_lowercase())
            } else if key == "WQM_LOG_DIR" {
                dir(&root, "logs")
            } else if key == db_key {
                dir(&root, "state") + "/state.db"
            } else {
                // Everything else is blanked rather than pointed somewhere. A test
                // that needs a store gets one from `StoreEndpoint`, and an empty
                // value means a stray reader fails loudly instead of silently
                // reaching whatever the developer's shell had set.
                String::new()
            };
            vars.insert(key.to_string(), value);
        }

        for value in vars.values() {
            if value.starts_with(root.to_string_lossy().as_ref()) && !value.ends_with("state.db") {
                std::fs::create_dir_all(value)?;
            }
        }

        Ok(Self {
            root,
            vars,
            _guard: guard,
        })
    }

    /// The temporary root every derived path sits under.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The full key/value set, for a caller that spawns a child process.
    ///
    /// Returned whole rather than as a lookup, so a caller cannot pick a subset and
    /// leave one sensitive key inherited.
    pub fn vars(&self) -> &BTreeMap<String, String> {
        &self.vars
    }

    /// The keys this harness takes responsibility for.
    pub fn overridden_keys() -> Vec<&'static str> {
        overridden_keys_owned()
    }
}

fn dir(root: &Path, name: &str) -> String {
    root.join(name).to_string_lossy().into_owned()
}

/// A temporary directory that removes itself.
///
/// Hand-rolled rather than pulling `tempfile`: the need is one directory with a
/// unique name and a `Drop`, and the harness's dependency surface is itself part of
/// what a test harness should keep small.
#[derive(Debug)]
struct TempRoot(PathBuf);

impl TempRoot {
    fn create(label: &str) -> std::io::Result<Self> {
        // Uniqueness from pid + a monotonic counter: no randomness needed, and two
        // roots within one process cannot collide.
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let safe: String = label
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let path = std::env::temp_dir().join(format!("wqm-test-{safe}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        // Best effort: a leaked temp dir is untidy, but panicking in Drop during a
        // test failure would replace the real diagnosis with this one.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_sensitive_key_is_written_not_defaulted() {
        let env = HermeticEnv::new("keys").expect("env");
        for key in HermeticEnv::overridden_keys() {
            assert!(
                env.vars().contains_key(key),
                "{key} is named as inherited-and-dangerous but is not overridden"
            );
        }
    }

    #[test]
    fn directory_keys_live_under_the_owned_root() {
        let env = HermeticEnv::new("paths").expect("env");
        let root = env.root().to_string_lossy().into_owned();
        for (key, value) in env.vars() {
            if key.starts_with("XDG_") || key == "WQM_LOG_DIR" || key == EnvVar::DatabasePath.key()
            {
                assert!(
                    value.starts_with(&root),
                    "{key}={value} escapes the harness root {root}"
                );
            }
        }
    }

    /// The defect is inheritance, so the test that matters is that an exported value
    /// does not survive into the harness's view.
    #[test]
    fn an_exported_value_is_overridden_rather_than_consulted() {
        // Edition 2021: `set_var` is safe here. The point is precisely to prove that
        // what the process environment says does not reach `HermeticEnv`.
        std::env::set_var(EnvVar::QdrantUrl.key(), "http://production.invalid:6333");
        std::env::set_var("XDG_CONFIG_HOME", "/somewhere/else");
        let env = HermeticEnv::new("inherit").expect("env");
        assert_eq!(env.vars()[EnvVar::QdrantUrl.key()], "");
        assert!(env.vars()["XDG_CONFIG_HOME"].starts_with(&*env.root().to_string_lossy()));
        std::env::remove_var(EnvVar::QdrantUrl.key());
        std::env::remove_var("XDG_CONFIG_HOME");
    }

    /// The point of sourcing from N8 rather than re-spelling: a key added there is
    /// covered here with no edit. Assert the coupling, not just today's list.
    #[test]
    fn every_n8_registry_key_is_covered() {
        let env = HermeticEnv::new("n8").expect("env");
        for v in EnvVar::ALL {
            assert!(
                env.vars().contains_key(v.key()),
                "N8 key {} is not overridden by the harness",
                v.key()
            );
        }
    }

    #[test]
    fn two_environments_do_not_share_a_root() {
        let a = HermeticEnv::new("iso").expect("a");
        let b = HermeticEnv::new("iso").expect("b");
        assert_ne!(a.root(), b.root());
    }

    #[test]
    fn the_root_is_removed_on_drop() {
        let path = {
            let env = HermeticEnv::new("drop").expect("env");
            env.root().to_path_buf()
        };
        assert!(!path.exists(), "{} survived Drop", path.display());
    }
}
