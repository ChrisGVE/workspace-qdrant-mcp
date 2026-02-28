/**
 * Pure helper functions for component matching and ID conversion.
 */

import type { ComponentInfo, ComponentMap } from './types.js';

// ── Helpers ──────────────────────────────────────────────────────────────

/**
 * Convert a path to a dot-separated component ID.
 *
 * "daemon/core"        → "daemon.core"
 * "cli"                → "cli"
 * "src/typescript/mcp" → "src.typescript.mcp"
 */
export function pathToComponentId(path: string): string {
  return path
    .replace(/\/+$/, '')   // trim trailing slashes
    .replace(/^\/+/, '')   // trim leading slashes
    .replace(/\//g, '.');  // replace / with .
}

/**
 * Check if a relative file path matches a component.
 *
 * A file matches if its relativePath starts with the component's basePath + "/".
 */
export function fileMatchesComponent(
  relativePath: string,
  component: ComponentInfo,
): boolean {
  const base = component.basePath;
  return relativePath === base ||
    relativePath.startsWith(base + '/');
}

/**
 * Check if a component ID matches a filter (exact or prefix).
 *
 * "daemon.core" matches filter "daemon.core" (exact)
 * "daemon.core" matches filter "daemon" (prefix)
 * "daemon.core" does NOT match filter "cli" (no match)
 */
export function componentMatchesFilter(
  componentId: string,
  filter: string,
): boolean {
  return componentId === filter ||
    componentId.startsWith(filter + '.');
}

/**
 * Assign a component to a file based on its relative path.
 *
 * Returns the most specific (longest basePath) matching component,
 * or undefined if no component matches.
 */
export function assignComponent(
  relativePath: string,
  components: ComponentMap,
): ComponentInfo | undefined {
  let bestMatch: ComponentInfo | undefined;
  let bestLength = -1;

  for (const component of components.values()) {
    if (fileMatchesComponent(relativePath, component)) {
      if (component.basePath.length > bestLength) {
        bestMatch = component;
        bestLength = component.basePath.length;
      }
    }
  }

  return bestMatch;
}
