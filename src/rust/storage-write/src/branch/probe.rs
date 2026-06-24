//! Git2-backed deletion truth-table probe (arch §4.3, AC-F9.1 / DR GP-3).
//!
//! File: `wqm-storage-write/src/branch/probe.rs`
//! Location: `src/rust/storage-write/src/branch/` (write-crate branch layer)
//! Context: Owns the GP-4 truth-table types and the git2-backed `probe_branch`
//!   function.  Split from `delete.rs` to keep each file within the arch §9
//!   line-budget.  The decision logic is pure (`delete_decision`) so every row
//!   of the truth table is unit-testable without a real git repository.
//!
//!   Uses git2 exclusively — `std::process::Command` / shell git is FORBIDDEN
//!   throughout this crate (arch §7.6, §6 "no shell git").
//!
//! Neighbors: [`super::delete`] (calls `probe_branch` + `delete_decision`),
//!   [`super::steps`] (the SQL step helpers called after a Proceed decision).

use std::path::Path;

/// The result of the GP-4 deletion truth-table probe (arch §4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteAction {
    /// Positively confirmed deleted: for-each-ref empty + reflog delete event.
    Proceed,
    /// Branch present in topology: do NOT delete.
    Keep,
    /// Ambiguous / read error / transient NotFound: defer to next reconcile run.
    Defer,
}

/// Input to the pure `delete_decision` function (AC-F9.1).
///
/// Produced by `probe_branch`; separated so the decision logic is unit-testable
/// without touching a git repository.
#[derive(Debug, Clone)]
pub struct GitBranchProbe {
    /// True if `for-each-ref` (or equivalent) returned no entry for the branch.
    pub for_each_ref_empty: bool,
    /// True if the reflog contains a delete event for the branch.
    pub reflog_has_delete: bool,
    /// True if the git2 call returned a transient error (NotFound-ambiguous, I/O, Auth).
    pub git2_error: bool,
    /// True if the reflog was unreachable or git dir was not found.
    pub reflog_unavailable: bool,
}

/// Pure decision function: maps a `GitBranchProbe` to a `DeleteAction`.
///
/// MUST NOT have side effects — the probe struct contains all inputs.
/// Every row of the truth table is independently testable (AC-F9.1).
pub fn delete_decision(probe: GitBranchProbe) -> DeleteAction {
    // Any ambiguity or error -> DEFER first (GP-4 positive-absence principle).
    if probe.git2_error || probe.reflog_unavailable {
        return DeleteAction::Defer;
    }
    // Branch still present -> Keep.
    if !probe.for_each_ref_empty {
        return DeleteAction::Keep;
    }
    // for-each-ref empty but no reflog confirmation -> DEFER (could be read error).
    if !probe.reflog_has_delete {
        return DeleteAction::Defer;
    }
    // Positively confirmed: ref gone AND reflog recorded the deletion.
    DeleteAction::Proceed
}

/// Probe the git repository at `repo_path` for the branch named `branch_name`.
///
/// Uses git2 exclusively (no `std::process::Command`/shell git — arch §7.6/§6).
/// Returns a `GitBranchProbe` capturing what the repo reports; errors are captured
/// in the probe rather than propagated so `delete_decision` remains purely structural.
///
/// `branch_name` is the short local branch name (e.g. `"main"`, `"feat/x"`).
pub fn probe_branch(repo_path: impl AsRef<Path>, branch_name: &str) -> GitBranchProbe {
    let repo = match git2::Repository::open(repo_path.as_ref()) {
        Ok(r) => r,
        Err(_) => {
            return GitBranchProbe {
                for_each_ref_empty: false,
                reflog_has_delete: false,
                git2_error: true,
                reflog_unavailable: true,
            };
        }
    };

    // Check whether the branch ref exists. A git2::ErrorCode::NotFound on a local
    // branch means "ref absent" but the same code on a remote-tracking ref may be a
    // network transient — we treat ALL NotFound as git2_error so the caller always
    // DEFERs on ambiguity (GP-4).
    let ref_name = format!("refs/heads/{}", branch_name);
    let for_each_ref_empty = match repo.find_reference(&ref_name) {
        Ok(_) => false, // branch present
        Err(e) if e.code() == git2::ErrorCode::NotFound => {
            // NotFound is ambiguous (transient vs genuine) -> flag git2_error.
            return GitBranchProbe {
                for_each_ref_empty: true,
                reflog_has_delete: false,
                git2_error: true,
                reflog_unavailable: false,
            };
        }
        Err(_) => {
            return GitBranchProbe {
                for_each_ref_empty: false,
                reflog_has_delete: false,
                git2_error: true,
                reflog_unavailable: false,
            };
        }
    };

    // Ref is present: branch still exists -> Keep path.
    if !for_each_ref_empty {
        return GitBranchProbe {
            for_each_ref_empty: false,
            reflog_has_delete: false,
            git2_error: false,
            reflog_unavailable: false,
        };
    }

    // Ref absent: check the reflog for a recorded delete event.
    let (reflog_has_delete, reflog_unavailable) =
        check_reflog_for_delete(&repo, &ref_name, branch_name);

    GitBranchProbe {
        for_each_ref_empty,
        reflog_has_delete,
        git2_error: false,
        reflog_unavailable,
    }
}

/// Scan the branch reflog for a delete event. Falls back to HEAD reflog.
///
/// Returns `(has_delete, unavailable)`.
fn check_reflog_for_delete(
    repo: &git2::Repository,
    ref_name: &str,
    branch_name: &str,
) -> (bool, bool) {
    match repo.reflog(ref_name) {
        Ok(reflog) => {
            // A non-empty reflog for an absent branch means the branch was live and
            // is now gone — treat as a confirmed delete event. git2 does not expose
            // a semantic "delete" marker, so non-empty is the best heuristic.
            let has_delete = !reflog.is_empty();
            (has_delete, false)
        }
        Err(_) => {
            // Reflog absent (shallow clone, GC, new repo) — scan HEAD reflog as fallback.
            let head_has_delete = check_head_reflog_for_branch(repo, branch_name);
            (head_has_delete, !head_has_delete)
        }
    }
}

/// Scan the HEAD reflog for an entry mentioning `branch_name`.
/// Fallback when the branch-specific reflog is absent.
fn check_head_reflog_for_branch(repo: &git2::Repository, branch_name: &str) -> bool {
    let Ok(reflog) = repo.reflog("HEAD") else {
        return false;
    };
    for entry in reflog.iter() {
        if let Some(summary) = entry.message() {
            if summary.contains(branch_name) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // AC-F9.1: pure truth-table unit tests (no git repo needed)
    // -----------------------------------------------------------------------

    #[test]
    fn proceed_requires_both_signals() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: true,
            git2_error: false,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Proceed);
    }

    #[test]
    fn keep_when_ref_present() {
        let probe = GitBranchProbe {
            for_each_ref_empty: false,
            reflog_has_delete: true,
            git2_error: false,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Keep);
    }

    #[test]
    fn keep_when_ref_present_no_reflog() {
        let probe = GitBranchProbe {
            for_each_ref_empty: false,
            reflog_has_delete: false,
            git2_error: false,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Keep);
    }

    #[test]
    fn defer_on_git2_error() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: true,
            git2_error: true,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Defer);
    }

    #[test]
    fn defer_on_reflog_unavailable() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: false,
            git2_error: false,
            reflog_unavailable: true,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Defer);
    }

    #[test]
    fn defer_on_empty_ref_no_reflog_confirmation() {
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
    fn transient_not_found_maps_to_defer() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: false,
            git2_error: true, // NotFound captured here
            reflog_unavailable: false,
        };
        assert_eq!(
            delete_decision(probe),
            DeleteAction::Defer,
            "transient NotFound must DEFER, not Proceed (AC-F9.1 SEED F06)"
        );
    }

    // AC-F9.1: integration — probe_branch on a nonexistent branch -> DEFER.
    #[test]
    fn probe_branch_notfound_defers() {
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
            "nonexistent branch -> NotFound -> git2_error -> DEFER"
        );
    }
}
