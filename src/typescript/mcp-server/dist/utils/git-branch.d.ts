/**
 * Git branch detection utilities.
 *
 * Reads .git/HEAD directly — no subprocess, no shell invocation.
 */
/**
 * Detect the current git branch for a project root.
 *
 * Algorithm:
 *   1. Locate .git/HEAD by walking up from projectRoot.
 *   2. If the file contains "ref: refs/heads/<name>", return <name>.
 *   3. If the file contains a raw SHA (detached HEAD), return the first 8 chars.
 *   4. If not a git repository or the file is unreadable, return "default".
 */
export declare function detectCurrentBranch(projectRoot: string): string;
//# sourceMappingURL=git-branch.d.ts.map