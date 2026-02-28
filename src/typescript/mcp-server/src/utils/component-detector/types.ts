/**
 * Types for component auto-detection.
 */

// ── Types ────────────────────────────────────────────────────────────────

export interface ComponentInfo {
  /** Dot-separated component ID, e.g. "daemon.core" */
  id: string;
  /** Base directory relative to project root, e.g. "daemon/core" */
  basePath: string;
  /** Glob patterns matching files in this component */
  patterns: string[];
  /** Detection source */
  source: 'cargo' | 'npm' | 'directory';
}

export type ComponentMap = Map<string, ComponentInfo>;
