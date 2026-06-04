/**
 * Cargo workspace component detection.
 *
 * Parses Cargo.toml [workspace] members to derive dot-separated
 * hierarchical component names.
 *
 * Example:
 *   Cargo.toml member "daemon/core" → component "daemon.core"
 */
import type { ComponentMap } from './types.js';
export declare function detectCargoWorkspace(projectPath: string): ComponentMap;
/**
 * Extract workspace members from Cargo.toml content.
 *
 * Parses the `members = [...]` array from a [workspace] section.
 * Handles multi-line arrays and inline comments.
 */
export declare function parseCargoMembers(content: string): string[];
//# sourceMappingURL=cargo-detector.d.ts.map