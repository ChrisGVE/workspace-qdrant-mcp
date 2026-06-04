/**
 * Git utility functions for repository detection and remote URL parsing.
 */
/**
 * Check if a path is a git repository
 */
export declare function isGitRepository(path: string): boolean;
/**
 * Find git repository root by walking up from a path.
 * Returns the path containing .git, or null if not found.
 */
export declare function findGitRoot(startPath: string): string | null;
/**
 * Get git remote URL for a repository
 *
 * Note: This is for informational purposes only.
 * The daemon computes the authoritative project_id.
 */
export declare function getGitRemoteUrl(repoPath: string): string | null;
//# sourceMappingURL=git-utils.d.ts.map