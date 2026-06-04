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
import type { ComponentMap } from './types.js';
export type { ComponentInfo, ComponentMap } from './types.js';
export { parseCargoMembers } from './cargo-detector.js';
export { pathToComponentId, fileMatchesComponent, componentMatchesFilter, assignComponent, } from './helpers.js';
/**
 * Detect project components from workspace definition files.
 *
 * Tries Cargo.toml first, then package.json, then falls back to
 * top-level directory heuristic.
 */
export declare function detectComponents(projectPath: string): ComponentMap;
//# sourceMappingURL=index.d.ts.map