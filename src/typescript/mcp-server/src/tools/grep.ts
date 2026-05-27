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

import { randomUUID } from 'node:crypto';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { getEffectiveCwd } from '../utils/request-context.js';
import type { TextSearchMatch } from '../clients/grpc-types.js';
import { finishToolEvent, logSearchEvent } from '../clients/search-event-queries.js';
import { SERVER_VERSION as MCP_SERVER_VERSION } from '../server-types.js';

/**
 * Conservative proxy for the size of the files that contain a grep match.
 * Used to compute `bytes_in` for token-economy instrumentation when the
 * daemon has not yet been extended to report per-match file sizes.
 *
 * Spec `docs/specs/20-token-economy-instrumentation.md` §3.2 calls for
 * "sum of file sizes for each unique `file_path`, capped at FILE_PROBE_CAP".
 * Until that probe lands (step 1 of the spec's implementation sequence),
 * we approximate per-unique-file at 8 KiB — large enough that
 * `savings_ratio` lands in a plausible range without overclaiming, small
 * enough that the metric won't dwarf real per-search byte counts.
 */
export const GREP_BYTES_IN_PER_FILE_PROXY = 8192;

/**
 * Pure helper: compute `bytes_out` / `bytes_in` proxies for a grep
 * response. `bytes_out` is the on-the-wire content cost the agent
 * actually paid; `bytes_in` is what they would have paid to load each
 * referenced file without the tool (proxied — see constant above).
 */
export function computeGrepEconomy(matches: GrepMatch[]): {
  bytesOut: number;
  bytesIn: number;
} {
  let bytesOut = 0;
  const uniqueFiles = new Set<string>();
  for (const m of matches) {
    bytesOut += m.content.length;
    for (const line of m.context_before) bytesOut += line.length;
    for (const line of m.context_after) bytesOut += line.length;
    uniqueFiles.add(m.file);
  }
  const bytesIn = Math.max(bytesOut, uniqueFiles.size * GREP_BYTES_IN_PER_FILE_PROXY);
  return { bytesOut, bytesIn };
}

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

/** Build the text search request object for the daemon. */
function buildGrepRequest(
  pattern: string,
  regex: boolean,
  caseSensitive: boolean,
  contextLines: number,
  maxResults: number,
  tenantId: string | undefined,
  branch: string | undefined,
  pathGlob: string | undefined
): {
  pattern: string;
  regex: boolean;
  case_sensitive: boolean;
  context_lines: number;
  max_results: number;
  tenant_id?: string;
  branch?: string;
  path_glob?: string;
} {
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
  return request;
}

/** Map daemon TextSearchMatch array to GrepMatch array. */
function mapGrepMatches(matches: TextSearchMatch[]): GrepMatch[] {
  return matches.map((m: TextSearchMatch) => ({
    file: m.file_path,
    line: m.line_number,
    content: m.content,
    context_before: m.context_before ?? [],
    context_after: m.context_after ?? [],
  }));
}

/** Build an empty failure GrepResponse. */
function grepError(message: string, latency_ms: number): GrepResponse {
  return { success: false, matches: [], total_matches: 0, truncated: false, latency_ms, message };
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

    if (!pattern) return grepError('Search pattern is required', 0);

    const startTime = Date.now();
    const eventId = randomUUID();

    let tenantId: string | undefined;
    if (scope === 'project') {
      tenantId = projectId ?? (await this.resolveProjectId());
      if (!tenantId) {
        // Log a quick event so the rejection still shows up in
        // followup / escalation analyses even though we never reach
        // the daemon.
        this.logGrepStart(eventId, pattern, maxResults, undefined);
        finishToolEvent(this.daemonClient, eventId, {
          resultCount: 0,
          latencyMs: Date.now() - startTime,
          bytesIn: 0,
          bytesOut: 0,
          toolVersion: MCP_SERVER_VERSION,
          outcome: 'unresolved_tenant',
        });
        return grepError(
          'Could not detect project ID. Use scope "all" or provide projectId.',
          Date.now() - startTime
        );
      }
    }

    this.logGrepStart(eventId, pattern, maxResults, tenantId);

    return this.executeSearch(
      pattern,
      regex,
      caseSensitive,
      contextLines,
      maxResults,
      tenantId,
      branch,
      pathGlob,
      startTime,
      eventId
    );
  }

  /** Log the pre-execution event for a grep call. */
  private logGrepStart(
    eventId: string,
    pattern: string,
    maxResults: number,
    tenantId: string | undefined
  ): void {
    // Cap the queryText so a pathological regex doesn't bloat the
    // search_events row.
    const truncatedPattern = pattern.length > 500 ? pattern.slice(0, 500) : pattern;
    logSearchEvent(this.daemonClient, {
      id: eventId,
      actor: 'claude',
      tool: 'mcp_qdrant',
      op: 'grep',
      queryText: truncatedPattern,
      topK: maxResults,
      projectId: tenantId,
    });
  }

  private async executeSearch(
    pattern: string,
    regex: boolean,
    caseSensitive: boolean,
    contextLines: number,
    maxResults: number,
    tenantId: string | undefined,
    branch: string | undefined,
    pathGlob: string | undefined,
    startTime: number,
    eventId: string
  ): Promise<GrepResponse> {
    try {
      const request = buildGrepRequest(
        pattern,
        regex,
        caseSensitive,
        contextLines,
        maxResults,
        tenantId,
        branch,
        pathGlob
      );
      const response = await this.daemonClient.textSearch(request);
      const matches = mapGrepMatches(response.matches);
      const economy = computeGrepEconomy(matches);
      const latencyMs = Date.now() - startTime;
      finishToolEvent(this.daemonClient, eventId, {
        resultCount: matches.length,
        latencyMs,
        bytesIn: economy.bytesIn,
        bytesOut: economy.bytesOut,
        toolVersion: MCP_SERVER_VERSION,
      });
      return {
        success: true,
        matches,
        total_matches: response.total_matches,
        truncated: response.truncated,
        latency_ms: latencyMs,
      };
    } catch (error) {
      const latencyMs = Date.now() - startTime;
      finishToolEvent(this.daemonClient, eventId, {
        resultCount: 0,
        latencyMs,
        bytesIn: 0,
        bytesOut: 0,
        toolVersion: MCP_SERVER_VERSION,
        outcome: 'error',
      });
      return grepError(
        `Grep failed: ${error instanceof Error ? error.message : 'unknown error'}`,
        latencyMs
      );
    }
  }

  /**
   * Resolve project ID from current working directory
   */
  private async resolveProjectId(): Promise<string | undefined> {
    const cwd = getEffectiveCwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
    return projectInfo?.projectId;
  }
}
