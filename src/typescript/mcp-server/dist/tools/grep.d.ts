/**
 * Grep tool implementation for FTS5-based code search
 *
 * Provides exact substring and regex search across indexed code files with:
 * - Pattern matching (exact or regex)
 * - Path glob filtering (e.g., "**\/*.rs")
 * - Context lines before/after matches
 * - Tenant/branch scoping
 *
 * Uses daemon's TextSearchService via gRPC.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { ProjectDetector } from '../utils/project-detector.js';
export interface GrepOptions {
    pattern: string;
    regex?: boolean;
    caseSensitive?: boolean;
    pathGlob?: string;
    scope?: 'project' | 'all';
    contextLines?: number;
    maxResults?: number;
    branch?: string;
    projectId?: string;
}
export interface GrepMatch {
    file: string;
    line: number;
    content: string;
    context_before: string[];
    context_after: string[];
}
export interface GrepResponse {
    success: boolean;
    matches: GrepMatch[];
    total_matches: number;
    truncated: boolean;
    latency_ms: number;
    message?: string;
}
/**
 * Grep tool for FTS5-based code search
 */
export declare class GrepTool {
    private readonly daemonClient;
    private readonly projectDetector;
    constructor(daemonClient: DaemonClient, projectDetector: ProjectDetector);
    /**
     * Search code using FTS5 trigram index
     */
    grep(options: GrepOptions): Promise<GrepResponse>;
    private executeSearch;
    /**
     * Resolve project ID from current working directory
     */
    private resolveProjectId;
}
//# sourceMappingURL=grep.d.ts.map