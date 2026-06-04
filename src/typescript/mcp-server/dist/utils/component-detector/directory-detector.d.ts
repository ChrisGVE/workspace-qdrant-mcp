/**
 * Directory-based component detection fallback.
 *
 * Uses top-level directories as components when no workspace
 * definition files (Cargo.toml, package.json) are found.
 */
import type { ComponentMap } from './types.js';
/**
 * Fallback: use top-level directories as components.
 *
 * Only includes directories that likely contain source code
 * (skips hidden dirs, build output, etc.).
 */
export declare function detectFromDirectories(projectPath: string): ComponentMap;
//# sourceMappingURL=directory-detector.d.ts.map