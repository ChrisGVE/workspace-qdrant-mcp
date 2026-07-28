//! Deployment-namespace tests (N8): the one knob, and the write refusal.
//!
//! `PROJECT_LOGISTICS.md`'s parallel-deployment model turns on two properties, and
//! both are asserted here rather than trusted: every deployed name derives from a
//! single suffix, and a write cannot address a collection outside the deployment.

use wqm_common::names::{
    Collection, WriteGuardError, WriteTarget, DEPLOYMENT_DIR, DEPLOYMENT_SUFFIX,
    IS_PARALLEL_DEPLOYMENT, SERVICE_LABEL,
};

// ---------------------------------------------------------------- the one knob

/// The whole point of a single knob: every derived name ends with it. If a name
/// family is added that does NOT derive from the suffix, this fails -- which is
/// the "one knob, not N hardcoded literals" rule made falsifiable rather than
/// merely stated.
#[test]
fn every_derived_name_carries_the_deployment_suffix() {
    let mut derived: Vec<&str> = vec![DEPLOYMENT_DIR, SERVICE_LABEL];
    derived.extend(Collection::ALL.iter().map(|c| c.deployed_name()));

    for name in derived {
        assert!(
            name.ends_with(DEPLOYMENT_SUFFIX),
            "`{name}` does not derive from the deployment suffix `{DEPLOYMENT_SUFFIX}` \
             -- a second knob has appeared"
        );
    }
}

#[test]
fn deployed_names_are_the_logical_names_plus_the_suffix() {
    for c in Collection::ALL {
        assert_eq!(
            c.deployed_name(),
            format!("{}{}", c.name(), DEPLOYMENT_SUFFIX),
            "deployed name must be exactly the logical name plus the suffix"
        );
    }
}

/// The logical identity is NOT the deployed location. Keeping them distinct is
/// what lets the migration read production while writing `-v2`.
/// This build must be a parallel deployment. Asserted at COMPILE time rather than
/// in a test body: the value is a `const`, so a runtime `assert!` on it is dead
/// code (clippy says so), while a const-eval assert fails the build outright --
/// which is the right severity for "am I about to run in production's namespace".
/// After cutover this line is the one that must be deliberately removed.
const _: () = assert!(
    IS_PARALLEL_DEPLOYMENT,
    "the deployment suffix is empty: this build would address production directly"
);

#[test]
fn logical_and_deployed_names_are_distinct_while_running_in_parallel() {
    for c in Collection::ALL {
        assert_ne!(c.name(), c.deployed_name());
    }
}

#[test]
fn deployed_names_are_unique() {
    let names: Vec<&str> = Collection::ALL.iter().map(|c| c.deployed_name()).collect();
    let unique: std::collections::HashSet<_> = names.iter().collect();
    assert_eq!(
        unique.len(),
        names.len(),
        "deployed names collide: {names:?}"
    );
}

#[test]
fn the_dir_and_service_label_do_not_collide_with_production() {
    assert_eq!(DEPLOYMENT_DIR, "workspace-qdrant-v2");
    assert_eq!(SERVICE_LABEL, "com.workspace-qdrant.memexd-v2");
}

// ------------------------------------------------------------ the write refusal

#[test]
fn a_canonical_collection_always_yields_its_deployed_name() {
    for c in Collection::ALL {
        assert_eq!(WriteTarget::of(c).as_str(), c.deployed_name());
    }
}

/// The blast-radius rule: on a shared Qdrant, isolation is by collection name
/// only, so a bare production name must be refused.
#[test]
fn a_bare_production_collection_name_is_refused() {
    for c in Collection::ALL {
        let err = WriteTarget::try_from_name(c.name())
            .expect_err("a bare production name must never produce a WriteTarget");
        match err {
            WriteGuardError::OutsideDeployment {
                ref name,
                required_suffix,
            } => {
                assert_eq!(name, c.name());
                assert_eq!(required_suffix, DEPLOYMENT_SUFFIX);
            }
        }
    }
}

#[test]
fn a_suffixed_canonical_name_is_accepted() {
    for c in Collection::ALL {
        let target = WriteTarget::try_from_name(c.deployed_name())
            .expect("a suffixed canonical name must be accepted");
        assert_eq!(target.as_str(), c.deployed_name());
    }
}

/// Carrying the suffix is necessary but not sufficient -- the name must also
/// resolve to a canonical collection, so a typo cannot conjure a write target.
#[test]
fn a_suffixed_but_unknown_name_is_still_refused() {
    for bogus in ["projekts-v2", "-v2", "images-v2", "scratchpadd-v2"] {
        assert!(
            WriteTarget::try_from_name(bogus).is_err(),
            "`{bogus}` carries the suffix but is not a canonical collection"
        );
    }
}

/// A near-miss must not slip through on a substring match.
#[test]
fn the_suffix_must_be_a_true_suffix() {
    for bogus in ["projects-v2-shadow", "v2-projects", "projects-v20"] {
        assert!(
            WriteTarget::try_from_name(bogus).is_err(),
            "`{bogus}` must not be accepted"
        );
    }
}

/// The refusal explains itself. An agent (and a human) must be able to tell WHY
/// a write was refused, which is the failure-visibility rule applied to the
/// guard's own error.
#[test]
fn the_refusal_names_the_collection_and_the_required_suffix() {
    let err = WriteTarget::try_from_name("projects").unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("projects"),
        "must name the collection: {text}"
    );
    assert!(
        text.contains(DEPLOYMENT_SUFFIX),
        "must name the required suffix: {text}"
    );
    assert!(
        text.contains("refusing"),
        "must read as a refusal, not a report: {text}"
    );
}
