use std::path::{Path, PathBuf};

use super::watcher_types::GitEventType;

/// Resolve the actual .git directory for a project.
///
/// Handles:
/// - Standard repos: `.git/` is a directory
/// - Worktrees: `.git` is a file containing `gitdir: /path/to/actual/.git/worktrees/name`
pub fn resolve_git_dir(project_root: &Path) -> Option<PathBuf> {
    let git_path = project_root.join(".git");

    if git_path.is_dir() {
        return Some(git_path);
    }

    if git_path.is_file() {
        // Worktree: .git is a file with `gitdir: <path>`
        if let Ok(content) = std::fs::read_to_string(&git_path) {
            let content = content.trim();
            if let Some(gitdir) = content.strip_prefix("gitdir: ") {
                let resolved = if Path::new(gitdir).is_absolute() {
                    PathBuf::from(gitdir)
                } else {
                    project_root.join(gitdir)
                };
                if resolved.exists() {
                    return Some(resolved);
                }
            }
        }
    }

    None
}

/// Resolve the common git directory (for worktrees, this is the main repo's .git).
///
/// In a worktree, `<gitdir>/commondir` contains the path to the shared git directory.
pub fn resolve_common_dir(git_dir: &Path) -> PathBuf {
    let commondir_file = git_dir.join("commondir");
    if let Ok(content) = std::fs::read_to_string(&commondir_file) {
        let common_path = content.trim();
        if Path::new(common_path).is_absolute() {
            PathBuf::from(common_path)
        } else {
            git_dir.join(common_path)
        }
    } else {
        // Not a worktree -- git_dir is the common dir
        git_dir.to_path_buf()
    }
}

/// Read the current branch name from .git/HEAD
pub fn read_current_branch(git_dir: &Path) -> Option<String> {
    let head_path = git_dir.join("HEAD");
    let content = std::fs::read_to_string(&head_path).ok()?;
    let content = content.trim();

    // HEAD contains "ref: refs/heads/<branch>" or a detached commit SHA
    if let Some(ref_path) = content.strip_prefix("ref: refs/heads/") {
        Some(ref_path.to_string())
    } else {
        // Detached HEAD
        None
    }
}

/// Parse the last entry of the reflog to determine what happened.
///
/// Reflog format: `<old-sha> <new-sha> <author> <timestamp> \t<operation>`
///
/// Returns: (old_sha, new_sha, event_type, old_branch_for_switch)
pub fn parse_reflog_last_entry(
    git_dir: &Path,
) -> Option<(String, String, GitEventType, Option<String>)> {
    let reflog_path = git_dir.join("logs").join("HEAD");

    let content = std::fs::read_to_string(&reflog_path).ok()?;
    let last_line = content.lines().last()?;

    parse_reflog_line(last_line)
}

/// Parse a single reflog line into its components.
///
/// Format: `<old-sha> <new-sha> Author Name <email> <timestamp> <tz>\t<operation description>`
pub fn parse_reflog_line(line: &str) -> Option<(String, String, GitEventType, Option<String>)> {
    let (metadata, operation) = line.split_once('\t')?;

    let parts: Vec<&str> = metadata.splitn(3, ' ').collect();
    if parts.len() < 3 {
        return None;
    }

    let old_sha = parts[0].to_string();
    let new_sha = parts[1].to_string();

    // Validate SHAs (should be 40 hex chars)
    if old_sha.len() != 40 || new_sha.len() != 40 {
        return None;
    }

    let operation = operation.trim();

    let (event_type, old_branch) = classify_reflog_operation(operation);

    Some((old_sha, new_sha, event_type, old_branch))
}

/// Classify a reflog operation string into a GitEventType.
///
/// Common reflog patterns:
/// - `checkout: moving from <old> to <new>` -> BranchSwitch
/// - `commit: <message>` / `commit (amend): <message>` / `commit (initial): <message>` -> Commit
/// - `merge <branch>: <details>` -> Merge
/// - `pull: <details>` / `pull --rebase: <details>` -> Pull
/// - `rebase (start):` / `rebase (continue):` / `rebase (finish):` -> Rebase
/// - `reset: moving to <ref>` -> Reset
fn classify_reflog_operation(operation: &str) -> (GitEventType, Option<String>) {
    let op_lower = operation.to_lowercase();

    if op_lower.starts_with("checkout: moving from ") {
        let rest = &operation["checkout: moving from ".len()..];
        let old_branch = rest.split(" to ").next().map(|s| s.to_string());
        return (GitEventType::BranchSwitch, old_branch);
    }

    if op_lower.starts_with("commit")
        && (op_lower.starts_with("commit:")
            || op_lower.starts_with("commit (amend):")
            || op_lower.starts_with("commit (initial):")
            || op_lower.starts_with("commit (merge):"))
    {
        return (GitEventType::Commit, None);
    }

    if op_lower.starts_with("merge ") {
        return (GitEventType::Merge, None);
    }

    if op_lower.starts_with("pull") {
        return (GitEventType::Pull, None);
    }

    if op_lower.starts_with("rebase") {
        return (GitEventType::Rebase, None);
    }

    if op_lower.starts_with("reset:") {
        return (GitEventType::Reset, None);
    }

    if op_lower.contains("stash") {
        return (GitEventType::Stash, None);
    }

    (GitEventType::Unknown, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_branch_switch() {
        let (event_type, old_branch) =
            classify_reflog_operation("checkout: moving from main to feature/new-stuff");
        assert_eq!(event_type, GitEventType::BranchSwitch);
        assert_eq!(old_branch.as_deref(), Some("main"));
    }

    #[test]
    fn test_classify_commit() {
        let (event_type, _) = classify_reflog_operation("commit: fix authentication bug");
        assert_eq!(event_type, GitEventType::Commit);

        let (event_type, _) = classify_reflog_operation("commit (amend): fix typo");
        assert_eq!(event_type, GitEventType::Commit);

        let (event_type, _) = classify_reflog_operation("commit (initial): initial commit");
        assert_eq!(event_type, GitEventType::Commit);

        let (event_type, _) = classify_reflog_operation("commit (merge): Merge branch 'dev'");
        assert_eq!(event_type, GitEventType::Commit);
    }

    #[test]
    fn test_classify_merge() {
        let (event_type, _) = classify_reflog_operation("merge feature/auth: Fast-forward");
        assert_eq!(event_type, GitEventType::Merge);
    }

    #[test]
    fn test_classify_pull() {
        let (event_type, _) = classify_reflog_operation("pull: Fast-forward");
        assert_eq!(event_type, GitEventType::Pull);

        let (event_type, _) = classify_reflog_operation("pull --rebase: checkout abc123");
        assert_eq!(event_type, GitEventType::Pull);
    }

    #[test]
    fn test_classify_rebase() {
        let (event_type, _) = classify_reflog_operation("rebase (start): checkout origin/main");
        assert_eq!(event_type, GitEventType::Rebase);

        let (event_type, _) =
            classify_reflog_operation("rebase (finish): returning to refs/heads/feature");
        assert_eq!(event_type, GitEventType::Rebase);
    }

    #[test]
    fn test_classify_reset() {
        let (event_type, _) = classify_reflog_operation("reset: moving to HEAD~1");
        assert_eq!(event_type, GitEventType::Reset);
    }

    #[test]
    fn test_classify_unknown() {
        let (event_type, _) = classify_reflog_operation("some-unknown-operation");
        assert_eq!(event_type, GitEventType::Unknown);
    }

    #[test]
    fn test_parse_reflog_line_commit() {
        let line = "abc1234567890123456789012345678901234567 def1234567890123456789012345678901234567 John Doe <john@example.com> 1708300000 +0000\tcommit: add new feature";
        let result = parse_reflog_line(line);
        assert!(result.is_some());
        let (old_sha, new_sha, event_type, _) = result.unwrap();
        assert_eq!(old_sha, "abc1234567890123456789012345678901234567");
        assert_eq!(new_sha, "def1234567890123456789012345678901234567");
        assert_eq!(event_type, GitEventType::Commit);
    }

    #[test]
    fn test_parse_reflog_line_branch_switch() {
        let line = "abc1234567890123456789012345678901234567 def1234567890123456789012345678901234567 John Doe <john@example.com> 1708300000 +0000\tcheckout: moving from main to feature/auth";
        let result = parse_reflog_line(line);
        assert!(result.is_some());
        let (_, _, event_type, old_branch) = result.unwrap();
        assert_eq!(event_type, GitEventType::BranchSwitch);
        assert_eq!(old_branch.as_deref(), Some("main"));
    }

    #[test]
    fn test_parse_reflog_line_invalid() {
        let result = parse_reflog_line("no tab in this line");
        assert!(result.is_none());

        let result = parse_reflog_line("abc def Author <email> 12345 +0000\tcommit: test");
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_git_dir_standard() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();

        let result = resolve_git_dir(temp_dir.path());
        assert_eq!(result, Some(git_dir));
    }

    #[test]
    fn test_resolve_git_dir_worktree() {
        let temp_dir = tempfile::tempdir().unwrap();

        let actual_git_dir = temp_dir.path().join("actual_git");
        std::fs::create_dir(&actual_git_dir).unwrap();

        let worktree_dir = temp_dir.path().join("worktree");
        std::fs::create_dir(&worktree_dir).unwrap();

        let git_file = worktree_dir.join(".git");
        std::fs::write(&git_file, format!("gitdir: {}", actual_git_dir.display())).unwrap();

        let result = resolve_git_dir(&worktree_dir);
        assert_eq!(result, Some(actual_git_dir));
    }

    #[test]
    fn test_resolve_git_dir_not_git() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = resolve_git_dir(temp_dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_read_current_branch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path();

        std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        assert_eq!(read_current_branch(git_dir), Some("main".to_string()));

        std::fs::write(
            git_dir.join("HEAD"),
            "abc1234567890123456789012345678901234567\n",
        )
        .unwrap();
        assert_eq!(read_current_branch(git_dir), None);
    }

    #[test]
    fn test_parse_reflog_last_entry() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path();

        let logs_dir = git_dir.join("logs");
        std::fs::create_dir_all(&logs_dir).unwrap();

        let reflog_content = "0000000000000000000000000000000000000000 abc1234567890123456789012345678901234567 John <john@x.com> 1708300000 +0000\tcommit (initial): init\nabc1234567890123456789012345678901234567 def1234567890123456789012345678901234567 John <john@x.com> 1708301000 +0000\tcommit: second commit\n";
        std::fs::write(logs_dir.join("HEAD"), reflog_content).unwrap();

        let result = parse_reflog_last_entry(git_dir);
        assert!(result.is_some());
        let (old_sha, new_sha, event_type, _) = result.unwrap();
        assert_eq!(old_sha, "abc1234567890123456789012345678901234567");
        assert_eq!(new_sha, "def1234567890123456789012345678901234567");
        assert_eq!(event_type, GitEventType::Commit);
    }

    #[test]
    fn test_parse_reflog_last_entry_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = parse_reflog_last_entry(temp_dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_common_dir_standard() {
        let temp_dir = tempfile::tempdir().unwrap();
        let git_dir = temp_dir.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();

        let result = resolve_common_dir(&git_dir);
        assert_eq!(result, git_dir);
    }

    #[test]
    fn test_resolve_common_dir_worktree() {
        let temp_dir = tempfile::tempdir().unwrap();

        let main_git_dir = temp_dir.path().join("main_repo").join(".git");
        std::fs::create_dir_all(&main_git_dir).unwrap();

        let worktree_git_dir = main_git_dir.join("worktrees").join("wt1");
        std::fs::create_dir_all(&worktree_git_dir).unwrap();

        std::fs::write(worktree_git_dir.join("commondir"), "../..").unwrap();

        let result = resolve_common_dir(&worktree_git_dir);
        assert_eq!(
            result.canonicalize().unwrap(),
            main_git_dir.canonicalize().unwrap()
        );
    }
}
