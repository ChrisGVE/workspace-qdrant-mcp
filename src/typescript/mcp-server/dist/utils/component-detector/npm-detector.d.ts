/**
 * npm/yarn workspace component detection.
 *
 * Parses package.json workspaces to derive dot-separated
 * hierarchical component names.
 *
 * Example:
 *   package.json workspace "packages/ui" → component "packages.ui"
 */
import type { ComponentMap } from './types.js';
export declare function detectNpmWorkspace(projectPath: string): ComponentMap;
//# sourceMappingURL=npm-detector.d.ts.map