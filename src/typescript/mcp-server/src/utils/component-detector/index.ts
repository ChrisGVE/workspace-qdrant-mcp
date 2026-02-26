/**
 * Component auto-detection from workspace definition files.
 *
 * Parses Cargo.toml [workspace] members and package.json workspaces
 * to derive dot-separated hierarchical component names.
 *
 * Examples:
 *   Cargo.toml member "daemon/core"  → component "daemon.core"
 *   Cargo.toml member "cli"          → component "cli"
 *   package.json workspace "packages/ui" → component "packages.ui"
 */

import { detectCargoWorkspace } from './cargo-detector.js';
import { detectNpmWorkspace } from './npm-detector.js';
import { detectFromDirectories } from './directory-detector.js';
import type { ComponentMap } from './types.js';

// ── Re-exports ────────────────────────────────────────────────────────────

export type { ComponentInfo, ComponentMap } from './types.js';
export { parseCargoMembers } from './cargo-detector.js';
export {
  pathToComponentId,
  fileMatchesComponent,
  componentMatchesFilter,
  assignComponent,
} from './helpers.js';

// ── Detection ─────────────────────────────────────────────────────────────

/**
 * Detect project components from workspace definition files.
 *
 * Tries Cargo.toml first, then package.json, then falls back to
 * top-level directory heuristic.
 */
export function detectComponents(projectPath: string): ComponentMap {
  const components = new Map<string, import('./types.js').ComponentInfo>();

  // Try Cargo workspace first (searches up to 2 levels for Cargo.toml with [workspace])
  const cargoComponents = detectCargoWorkspace(projectPath);
  if (cargoComponents.size > 0) {
    for (const [id, info] of cargoComponents) {
      components.set(id, info);
    }
  }

  // Try npm/yarn workspaces
  const npmComponents = detectNpmWorkspace(projectPath);
  if (npmComponents.size > 0) {
    for (const [id, info] of npmComponents) {
      // Don't overwrite Cargo-detected components
      if (!components.has(id)) {
        components.set(id, info);
      }
    }
  }

  // Fallback: top-level directories if nothing detected
  if (components.size === 0) {
    const dirComponents = detectFromDirectories(projectPath);
    for (const [id, info] of dirComponents) {
      components.set(id, info);
    }
  }

  return components;
}
