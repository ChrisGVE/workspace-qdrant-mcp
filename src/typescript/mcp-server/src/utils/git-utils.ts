/**
 * Git utility functions for repository detection and remote URL parsing.
 */

import { existsSync, statSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Check if a path is a git repository
 */
export function isGitRepository(path: string): boolean {
  const gitDir = join(path, '.git');
  try {
    return existsSync(gitDir) && statSync(gitDir).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Get git remote URL for a repository
 *
 * Note: This is for informational purposes only.
 * The daemon computes the authoritative project_id.
 */
export function getGitRemoteUrl(repoPath: string): string | null {
  const gitConfigPath = join(repoPath, '.git', 'config');

  try {
    const config = readFileSync(gitConfigPath, 'utf-8');

    // Parse remote "origin" URL
    const remoteMatch = config.match(/\[remote "origin"\][^\[]*url = (.+)/m);
    if (remoteMatch?.[1]) {
      return remoteMatch[1].trim();
    }
  } catch {
    // File doesn't exist or not readable
  }

  return null;
}
