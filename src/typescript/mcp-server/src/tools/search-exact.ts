/**
 * FTS5 exact/substring search via daemon's TextSearchService.
 */

import { randomUUID } from 'node:crypto';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { SearchOptions, SearchResult, SearchResponse } from './search-types.js';
import { PROJECTS_COLLECTION } from './search-types.js';

/** Resolve the tenant ID for an exact search from options and project detector. */
async function resolveExactSearchTenant(
  options: SearchOptions,
  projectDetector: ProjectDetector
): Promise<string | undefined> {
  if (options.scope === 'all') return undefined;
  if (options.projectId) return options.projectId;
  const projectInfo = await projectDetector.getProjectInfo(process.cwd(), false);
  return projectInfo?.projectId;
}

/** Map daemon text search matches to SearchResult array. */
function mapExactResults(
  matches: Array<{
    file_path: string;
    line_number: number;
    content: string;
    tenant_id?: string;
    branch?: string;
    context_before?: string[];
    context_after?: string[];
  }>
): SearchResult[] {
  return matches.map((m, idx) => ({
    id: `${m.file_path}:${m.line_number}`,
    score: 1.0 - idx * 0.001,
    collection: PROJECTS_COLLECTION,
    content: m.content,
    metadata: {
      file_path: m.file_path,
      line_number: m.line_number,
      tenant_id: m.tenant_id,
      branch: m.branch,
      context_before: m.context_before,
      context_after: m.context_after,
      _search_type: 'exact',
    },
  }));
}

/** Build the text search request from search options. */
function buildExactSearchRequest(
  options: SearchOptions,
  tenantId: string | undefined
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
    pattern: options.query,
    regex: false,
    case_sensitive: true,
    context_lines: options.contextLines ?? 0,
    max_results: options.limit ?? 100,
  };
  if (tenantId) request.tenant_id = tenantId;
  if (options.branch) request.branch = options.branch;
  if (options.pathGlob) request.path_glob = options.pathGlob;
  return request;
}

/**
 * Execute FTS5 exact/substring search via daemon's TextSearchService.
 * Maps TextSearchResponse to the standard SearchResponse format.
 */
export async function searchExact(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  projectDetector: ProjectDetector,
  options: SearchOptions
): Promise<SearchResponse> {
  const startTime = Date.now();
  const eventId = randomUUID();
  const tenantId = await resolveExactSearchTenant(options, projectDetector);

  stateManager.logSearchEvent({
    id: eventId,
    projectId: tenantId,
    actor: 'claude',
    tool: 'mcp_qdrant',
    op: 'search_exact',
    queryText: options.query,
  });

  try {
    const request = buildExactSearchRequest(options, tenantId);
    const response = await daemonClient.textSearch(request);
    const results = mapExactResults(response.matches);

    stateManager.updateSearchEvent(eventId, {
      resultCount: results.length,
      latencyMs: Date.now() - startTime,
    });
    return {
      results,
      total: response.total_matches,
      query: options.query,
      mode: 'keyword',
      scope: options.scope ?? 'project',
      collections_searched: [PROJECTS_COLLECTION],
    };
  } catch (error) {
    stateManager.updateSearchEvent(eventId, { resultCount: 0, latencyMs: Date.now() - startTime });
    return {
      results: [],
      total: 0,
      query: options.query,
      mode: 'keyword',
      scope: options.scope ?? 'project',
      collections_searched: [],
      status: 'uncertain',
      status_reason: `Exact search failed: ${error instanceof Error ? error.message : 'unknown error'}`,
    };
  }
}
