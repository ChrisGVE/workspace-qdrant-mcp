/**
 * npm/yarn workspace component detection.
 *
 * Parses package.json workspaces to derive dot-separated
 * hierarchical component names.
 *
 * Example:
 *   package.json workspace "packages/ui" → component "packages.ui"
 */

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type { ComponentMap } from './types.js';
import { pathToComponentId } from './helpers.js';

// ── npm workspace ────────────────────────────────────────────────────────

/**
 * Parse package.json workspaces into components.
 *
 * Handles both array format (`"workspaces": ["packages/*"]`)
 * and object format (`"workspaces": { "packages": ["pkg-a"] }`).
 */
export function detectNpmWorkspace(projectPath: string): ComponentMap {
  const components: ComponentMap = new Map();

  const pkgPath = join(projectPath, 'package.json');
  if (!existsSync(pkgPath)) return components;

  let pkg: Record<string, unknown>;
  try {
    pkg = JSON.parse(readFileSync(pkgPath, 'utf-8')) as Record<string, unknown>;
  } catch {
    return components;
  }

  let workspacePaths: string[] = [];

  if (Array.isArray(pkg['workspaces'])) {
    workspacePaths = pkg['workspaces'] as string[];
  } else if (pkg['workspaces'] && typeof pkg['workspaces'] === 'object') {
    const ws = pkg['workspaces'] as Record<string, unknown>;
    if (Array.isArray(ws['packages'])) {
      workspacePaths = ws['packages'] as string[];
    }
  }

  for (const wsPath of workspacePaths) {
    // Resolve globs: "packages/*" → list actual subdirectories
    if (wsPath.includes('*')) {
      const baseDir = wsPath.replace(/\/?\*.*$/, '');
      const fullBase = join(projectPath, baseDir);
      if (!existsSync(fullBase)) continue;

      try {
        const entries = readdirSync(fullBase);
        for (const entry of entries) {
          const entryPath = join(fullBase, entry);
          try {
            if (statSync(entryPath).isDirectory()) {
              const relPath = baseDir ? `${baseDir}/${entry}` : entry;
              const id = pathToComponentId(relPath);
              components.set(id, {
                id,
                basePath: relPath,
                patterns: [`${relPath}/**`],
                source: 'npm',
              });
            }
          } catch {
            // Skip entries we can't stat
          }
        }
      } catch {
        // Skip if we can't read directory
      }
    } else {
      // Direct path
      const id = pathToComponentId(wsPath);
      components.set(id, {
        id,
        basePath: wsPath,
        patterns: [`${wsPath}/**`],
        source: 'npm',
      });
    }
  }

  return components;
}
