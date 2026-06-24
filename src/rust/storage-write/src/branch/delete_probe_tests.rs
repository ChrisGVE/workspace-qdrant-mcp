//! Truth-table and probe integration tests for `branch::delete` (AC-F9.1).
//!
//! These are pure unit tests (no DB, no async) that verify every row of the
//! `delete_decision` truth table, plus one integration test that runs
//! `probe_branch` against a real git2 repo. Separated from the SQL-integration
//! tests in `delete_tests.rs` to keep each file within the codesize budget.

use super::*;

// ---------------------------------------------------------------------------
// AC-F9.1: truth table -- every row of delete_decision
// (types re-exported from super::probe via delete.rs pub use)
// ---------------------------------------------------------------------------

#[test]
fn truth_table_proceed_requires_both_signals() {
    let probe = GitBranchProbe {
        for_each_ref_empty: true,
        reflog_has_delete: true,
        git2_error: false,
        reflog_unavailable: false,
    };
    assert_eq!(delete_decision(probe), DeleteAction::Proceed);
}

#[test]
fn truth_table_keep_when_ref_present() {
    let probe = GitBranchProbe {
        for_each_ref_empty: false,
        reflog_has_delete: true,
        git2_error: false,
        reflog_unavailable: false,
    };
    assert_eq!(delete_decision(probe), DeleteAction::Keep);
}

#[test]
fn truth_table_keep_when_ref_present_no_reflog() {
    let probe = GitBranchProbe {
        for_each_ref_empty: false,
        reflog_has_delete: false,
        git2_error: false,
        reflog_unavailable: false,
    };
    assert_eq!(delete_decision(probe), DeleteAction::Keep);
}

#[test]
fn truth_table_defer_on_git2_error() {
    let probe = GitBranchProbe {
        for_each_ref_empty: true,
        reflog_has_delete: true,
        git2_error: true,
        reflog_unavailable: false,
    };
    assert_eq!(delete_decision(probe), DeleteAction::Defer);
}

#[test]
fn truth_table_defer_on_reflog_unavailable() {
    let probe = GitBranchProbe {
        for_each_ref_empty: true,
        reflog_has_delete: false,
        git2_error: false,
        reflog_unavailable: true,
    };
    assert_eq!(delete_decision(probe), DeleteAction::Defer);
}

#[test]
fn truth_table_defer_on_empty_ref_no_reflog_confirmation() {
    let probe = GitBranchProbe {
        for_each_ref_empty: true,
        reflog_has_delete: false,
        git2_error: false,
        reflog_unavailable: false,
    };
    assert_eq!(delete_decision(probe), DeleteAction::Defer);
}

// AC-F9.1: transient NotFound MUST map to DEFER (SEED F06).
#[test]
fn truth_table_transient_not_found_maps_to_defer() {
    let probe = GitBranchProbe {
        for_each_ref_empty: true,
        reflog_has_delete: false,
        git2_error: true, // NotFound captured as git2_error
        reflog_unavailable: false,
    };
    assert_eq!(
        delete_decision(probe),
        DeleteAction::Defer,
        "transient NotFound must map to DEFER, not Proceed (AC-F9.1 SEED F06)"
    );
}

// AC-F9.1: integration -- probe_branch on a nonexistent branch -> DEFER.
#[test]
fn probe_branch_notfound_defers() {
    use super::super::super::probe::probe_branch;

    let dir = tempfile::TempDir::new().expect("tempdir");
    let repo = git2::Repository::init(dir.path()).expect("git init");
    let sig = git2::Signature::now("Test", "t@t.com").expect("sig");
    let tree_id = {
        let mut index = repo.index().expect("index");
        index.write_tree().expect("write_tree")
    };
    let tree = repo.find_tree(tree_id).expect("tree");
    repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[])
        .expect("commit");

    let probe = probe_branch(dir.path(), "nonexistent-branch");
    assert_eq!(
        delete_decision(probe),
        DeleteAction::Defer,
        "probe for nonexistent branch -> NotFound -> git2_error -> DEFER"
    );
}
