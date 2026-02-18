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
import type { TextSearchMatch } from '../clients/grpc-types.js';

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
export class GrepTool {
  private readonly daemonClient: DaemonClient;
  private readonly projectDetector: ProjectDetector;

  constructor(daemonClient: DaemonClient, projectDetector: ProjectDetector) {
    this.daemonClient = daemonClient;
    this.projectDetector = projectDetector;
  }

  /**
   * Search code using FTS5 trigram index
   */
  async grep(options: GrepOptions): Promise<GrepResponse> {
    const {
      pattern,
      regex = false,
      caseSensitive = true,
      pathGlob,
      scope = 'project',
      contextLines = 0,
      maxResults = 1000,
      branch,
      projectId,
    } = options;

    if (!pattern) {
      return {
        success: false,
        matches: [],
        total_matches: 0,
        truncated: false,
        latency_ms: 0,
        message: 'Search pattern is required',
      };
    }

    const startTime = Date.now();

    // Resolve tenant_id based on scope
    let tenantId: string | undefined;
    if (scope === 'project') {
      tenantId = projectId ?? (await this.resolveProjectId());
      if (!tenantId) {
        return {
          success: false,
          matches: [],
          total_matches: 0,
          truncated: false,
          latency_ms: Date.now() - startTime,
          message: 'Could not detect project ID. Use scope "all" or provide projectId.',
        };
      }
    }

    try {
      // Build request conditionally to satisfy exactOptionalPropertyTypes
      const request: {
        pattern: string;
        regex: boolean;
        case_sensitive: boolean;
        context_lines: number;
        max_results: number;
        tenant_id?: string;
        branch?: string;
        path_glob?: string;
      } = {
        pattern,
        regex,
        case_sensitive: caseSensitive,
        context_lines: contextLines,
        max_results: maxResults,
      };
      if (tenantId) request.tenant_id = tenantId;
      if (branch) request.branch = branch;
      if (pathGlob) request.path_glob = pathGlob;

      const response = await this.daemonClient.textSearch(request);

      const matches: GrepMatch[] = response.matches.map((m: TextSearchMatch) => ({
        file: m.file_path,
        line: m.line_number,
        content: m.content,
        context_before: m.context_before ?? [],
        context_after: m.context_after ?? [],
      }));

      return {
        success: true,
        matches,
        total_matches: response.total_matches,
        truncated: response.truncated,
        latency_ms: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        matches: [],
        total_matches: 0,
        truncated: false,
        latency_ms: Date.now() - startTime,
        message: `Grep failed: ${error instanceof Error ? error.message : 'unknown error'}`,
      };
    }
  }

  /**
   * Resolve project ID from current working directory
   */
  private async resolveProjectId(): Promise<string | undefined> {
    const cwd = process.cwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
    return projectInfo?.projectId;
  }
}
