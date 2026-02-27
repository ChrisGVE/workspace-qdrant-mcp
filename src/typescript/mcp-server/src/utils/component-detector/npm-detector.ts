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
/** Extract workspace paths from package.json workspaces field. */
function parseWorkspacePaths(pkg: Record<string, unknown>): string[] {
  if (Array.isArray(pkg['workspaces'])) return pkg['workspaces'] as string[];
  if (pkg['workspaces'] && typeof pkg['workspaces'] === 'object') {
    const ws = pkg['workspaces'] as Record<string, unknown>;
    if (Array.isArray(ws['packages'])) return ws['packages'] as string[];
  }
  return [];
}

/** Resolve a glob workspace path (e.g. "packages/*") into components. */
function resolveGlobWorkspace(
  projectPath: string, wsPath: string, components: ComponentMap,
): void {
  const baseDir = wsPath.replace(/\/?\*.*$/, '');
  const fullBase = join(projectPath, baseDir);
  if (!existsSync(fullBase)) return;

  try {
    for (const entry of readdirSync(fullBase)) {
      try {
        if (statSync(join(fullBase, entry)).isDirectory()) {
          const relPath = baseDir ? `${baseDir}/${entry}` : entry;
          const id = pathToComponentId(relPath);
          components.set(id, { id, basePath: relPath, patterns: [`${relPath}/**`], source: 'npm' });
        }
      } catch { /* skip entries we can't stat */ }
    }
  } catch { /* skip if we can't read directory */ }
}

export function detectNpmWorkspace(projectPath: string): ComponentMap {
  const components: ComponentMap = new Map();
  const pkgPath = join(projectPath, 'package.json');
  if (!existsSync(pkgPath)) return components;

  let pkg: Record<string, unknown>;
  try {
    pkg = JSON.parse(readFileSync(pkgPath, 'utf-8')) as Record<string, unknown>;
  } catch { return components; }

  for (const wsPath of parseWorkspacePaths(pkg)) {
    if (wsPath.includes('*')) {
      resolveGlobWorkspace(projectPath, wsPath, components);
    } else {
      const id = pathToComponentId(wsPath);
      components.set(id, { id, basePath: wsPath, patterns: [`${wsPath}/**`], source: 'npm' });
    }
  }
  return components;
}
