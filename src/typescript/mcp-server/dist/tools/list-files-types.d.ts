/**
 * Types and constants for the list MCP tool.
 */
export declare const DEFAULT_DEPTH = 3;
export declare const MAX_DEPTH = 10;
export declare const DEFAULT_LIMIT = 200;
export declare const MAX_LIMIT = 500;
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
    /** Opaque pagination cursor from a previous response's next_token */
    cursor?: string;
    /** Page size for cursor-based pagination (falls back to limit) */
    pageSize?: number;
    /**
     * Filter by branch name.
     * Pass "*" to list files across all branches.
     * Defaults to the session's current branch when not provided.
     */
    branch?: string;
}
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
    /** Opaque cursor for fetching the next page; absent when no more pages */
    next_token?: string;
}
//# sourceMappingURL=list-files-types.d.ts.map