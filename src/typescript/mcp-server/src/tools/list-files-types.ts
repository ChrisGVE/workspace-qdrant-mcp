/**
 * Types and constants for the list MCP tool.
 */

// ── Constants ────────────────────────────────────────────────────────────

export const DEFAULT_DEPTH = 3;
export const MAX_DEPTH = 10;
export const DEFAULT_LIMIT = 200;
export const MAX_LIMIT = 500;

// ── Input types ──────────────────────────────────────────────────────────

export type ListFormat = 'tree' | 'summary' | 'flat';

export interface ListOptions {
  path?: string;
  depth?: number;
  format?: ListFormat;
  fileType?: string;
  language?: string;
  extension?: string;
  pattern?: string;
  includeTests?: boolean;
  limit?: number;
  projectId?: string;
  /** Filter by component (dot-separated ID or prefix, e.g. "daemon" or "daemon.core") */
  component?: string;
}

// ── Internal tree types ──────────────────────────────────────────────────

export interface FolderNode {
  name: string;
  children: Map<string, FolderNode>;
  files: FileLeaf[];
  /** If set, this folder is a submodule root — do not expand children */
  submodule?: SubmoduleMarker;
  /** Total file count in this subtree (computed during tree build) */
  totalFiles: number;
}

export interface FileLeaf {
  name: string;
  extension: string | null;
  language: string | null;
  isTest: boolean;
}

export interface SubmoduleMarker {
  repoName: string;
}

// ── Output types ─────────────────────────────────────────────────────────

export interface ComponentSummary {
  id: string;
  basePath: string;
  source: 'cargo' | 'npm' | 'directory';
}

export interface ListStats {
  files: number;
  folders: number;
  languages: string[];
  truncated: boolean;
  totalMatching: number;
  /** Detected project components (when available) */
  components?: ComponentSummary[];
}

export interface ListResponse {
  success: boolean;
  projectPath: string | null;
  basePath: string;
  format: ListFormat;
  listing: string;
  stats: ListStats;
  message?: string;
}
