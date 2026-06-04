/**
 * Git branch detection utilities.
 *
 * Reads .git/HEAD directly — no subprocess, no shell invocation.
 */
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { findGitRoot } from './git-utils.js';
/**
 * Detect the current git branch for a project root.
 *
 * Algorithm:
 *   1. Locate .git/HEAD by walking up from projectRoot.
 *   2. If the file contains "ref: refs/heads/<name>", return <name>.
 *   3. If the file contains a raw SHA (detached HEAD), return the first 8 chars.
 *   4. If not a git repository or the file is unreadable, return "default".
 */
export function detectCurrentBranch(projectRoot) {
    const gitRoot = findGitRoot(projectRoot) ?? projectRoot;
    const headPath = join(gitRoot, '.git', 'HEAD');
    if (!existsSync(headPath)) {
        return 'default';
    }
    try {
        const content = readFileSync(headPath, 'utf-8').trim();
        // Symbolic ref — normal branch checkout
        const refMatch = content.match(/^ref: refs\/heads\/(.+)$/);
        if (refMatch?.[1]) {
            return refMatch[1];
        }
        // Detached HEAD — bare SHA (40 hex chars)
        if (/^[0-9a-f]{40}$/i.test(content)) {
            return content.slice(0, 8);
        }
        // Unrecognised format
        return 'default';
    }
    catch {
        return 'default';
    }
}
//# sourceMappingURL=git-branch.js.map