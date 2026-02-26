/**
 * Directory-based component detection fallback.
 *
 * Uses top-level directories as components when no workspace
 * definition files (Cargo.toml, package.json) are found.
 */

import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type { ComponentMap } from './types.js';

// ── Directory fallback ───────────────────────────────────────────────────

const IGNORED_DIRS = new Set([
  '.git', '.github', '.vscode', '.idea',
  'node_modules', 'target', 'dist', 'build',
  '.taskmaster', '.claude', '.serena', 'tmp',
]);

/**
 * Fallback: use top-level directories as components.
 *
 * Only includes directories that likely contain source code
 * (skips hidden dirs, build output, etc.).
 */
export function detectFromDirectories(projectPath: string): ComponentMap {
  const components: ComponentMap = new Map();

  try {
    const entries = readdirSync(projectPath);
    for (const entry of entries) {
      if (entry.startsWith('.') || IGNORED_DIRS.has(entry)) continue;

      const fullPath = join(projectPath, entry);
      try {
        if (statSync(fullPath).isDirectory()) {
          components.set(entry, {
            id: entry,
            basePath: entry,
            patterns: [`${entry}/**`],
            source: 'directory',
          });
        }
      } catch {
        // Skip entries we can't stat
      }
    }
  } catch {
    // Can't read project directory
  }

  return components;
}
