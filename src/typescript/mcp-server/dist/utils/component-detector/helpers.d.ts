/**
 * Pure helper functions for component matching and ID conversion.
 */
import type { ComponentInfo, ComponentMap } from './types.js';
/**
 * Convert a path to a dot-separated component ID.
 *
 * "daemon/core"        → "daemon.core"
 * "cli"                → "cli"
 * "src/typescript/mcp" → "src.typescript.mcp"
 */
export declare function pathToComponentId(path: string): string;
/**
 * Check if a relative file path matches a component.
 *
 * A file matches if its relativePath starts with the component's basePath + "/".
 */
export declare function fileMatchesComponent(relativePath: string, component: ComponentInfo): boolean;
/**
 * Check if a component ID matches a filter (exact or prefix).
 *
 * "daemon.core" matches filter "daemon.core" (exact)
 * "daemon.core" matches filter "daemon" (prefix)
 * "daemon.core" does NOT match filter "cli" (no match)
 */
export declare function componentMatchesFilter(componentId: string, filter: string): boolean;
/**
 * Assign a component to a file based on its relative path.
 *
 * Returns the most specific (longest basePath) matching component,
 * or undefined if no component matches.
 */
export declare function assignComponent(relativePath: string, components: ComponentMap): ComponentInfo | undefined;
//# sourceMappingURL=helpers.d.ts.map