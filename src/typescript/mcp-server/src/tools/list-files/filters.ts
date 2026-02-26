/**
 * File filtering utilities for the list tool.
 *
 * Provides glob-based filtering and folder counting helpers.
 */

import type { TrackedFileEntry } from '../../clients/tracked-files-queries/index.js';
import type { FolderNode } from '../list-files-types.js';

// ── Folder counting ───────────────────────────────────────────────────────

export function countFolders(node: FolderNode): number {
  let count = 0;
  for (const child of node.children.values()) {
    count += 1 + countFolders(child);
  }
  return count;
}

// ── Glob filtering ────────────────────────────────────────────────────────

/**
 * Simple glob filter on relative paths.
 * Supports * (any non-/ chars) and ** (any path segment including /).
 */
export function filterByGlob(files: TrackedFileEntry[], pattern: string): TrackedFileEntry[] {
  const regex = globToRegex(pattern);
  return files.filter((f) => regex.test(f.relativePath));
}

export function globToRegex(pattern: string): RegExp {
  let result = '';
  let i = 0;

  while (i < pattern.length) {
    const c = pattern.charAt(i);

    if (c === '*' && i + 1 < pattern.length && pattern.charAt(i + 1) === '*') {
      // ** matches anything including /
      result += '.*';
      i += 2;
      // Skip trailing /
      if (i < pattern.length && pattern.charAt(i) === '/') i++;
    } else if (c === '*') {
      // * matches anything except /
      result += '[^/]*';
      i++;
    } else if (c === '?') {
      result += '[^/]';
      i++;
    } else if ('.+^${}()|[]\\'.includes(c)) {
      result += '\\' + c;
      i++;
    } else {
      result += c;
      i++;
    }
  }

  return new RegExp(`^${result}$`);
}
