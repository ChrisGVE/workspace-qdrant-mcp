/**
 * FTS5 exact/substring search via daemon's TextSearchService.
 */

import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { SearchOptions, SearchResult, SearchResponse } from './search-types.js';
import { PROJECTS_COLLECTION } from './search-types.js';
import { attachIndexingProgress } from './search-helpers.js';
import {
  applyEffectiveBranch,
  resolveEffectiveBranch,
  resolveProjectIdentity,
} from './branch-scope.js';

/**
 * Resolution outcome for exact-search tenant scoping.
 *
 * Closes F-004: project-scope exact search MUST refuse to run when no
 * tenant can be resolved — without a tenant the daemon's FTS query
 * builder drops the `fm.tenant_id = ?` clause and broadens to every
 * tenant in the FTS index.
 */
type ExactSearchTenantResolution =
  | { kind: 'tenant'; tenantId: string; projectPath?: string | undefined }
  | { kind: 'unscoped' } // explicit `scope: 'all'` — caller asked for global FTS
  | { kind: 'unresolved' }; // project-scope but no tenant could be found

async function resolveExactSearchTenant(
  options: SearchOptions,
  projectDetector: ProjectDetector,
  stateManager: SqliteStateManager
): Promise<ExactSearchTenantResolution> {
  if (options.scope === 'all') return { kind: 'unscoped' };
  const identity = await resolveProjectIdentity(projectDetector, options.projectId);
  if (identity.projectId) {
    return {
      kind: 'tenant',
      tenantId: identity.projectId,
      projectPath:
        identity.projectPath ??
        stateManager.getProjectById(identity.projectId).data?.project_path,
    };
  }
  return { kind: 'unresolved' };
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
  // `branch: "*"` is the documented "any branch" opt-out (see
  // buildSearchOptions / search-filters.ts buildBranchCondition). The
  // daemon FTS query builder has no "*" concept — it would filter
  // `fm.branch = '*'` literally and match nothing — so drop the filter
  // entirely, matching the Qdrant path's behaviour.
  if (options.branch && options.branch !== '*') request.branch = options.branch;
  if (options.pathGlob) request.path_glob = options.pathGlob;
  return request;
}

/** Build the response returned when project-scope exact search has no
 * resolvable tenant. Closes F-004 (no broadening to all FTS tenants). */
function unresolvedTenantResponse(options: SearchOptions): SearchResponse {
  return {
    results: [],
    total: 0,
    query: options.query,
    mode: 'keyword',
    scope: options.scope ?? 'project',
    collections_searched: [],
    status: 'uncertain',
    status_reason:
      'Project scope requested but no project could be resolved. ' +
      'Pass `projectId` explicitly, run from a registered project directory, ' +
      'or set `scope: "all"` to search across every indexed tenant.',
  };
}

/**
 * Execute FTS5 exact/substring search via daemon's TextSearchService.
 * Maps TextSearchResponse to the standard SearchResponse format.
 */
export async function searchExact(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  projectDetector: ProjectDetector,
  options: SearchOptions,
  eventId: string
): Promise<SearchResponse> {
  const startTime = Date.now();
  const resolution = await resolveExactSearchTenant(options, projectDetector, stateManager);

  if (resolution.kind === 'unresolved') {
    // F-004: refuse to broaden to every tenant in the FTS index. The
    // pre-fix code path omitted `tenant_id` from the daemon request and
    // the Rust query builder then dropped its `fm.tenant_id = ?`
    // clause, returning cross-tenant matches.
    stateManager.logSearchEvent({
      id: eventId,
      actor: options.telemetryActor ?? 'claude',
      tool: 'mcp_qdrant',
      op: 'search_exact',
      queryText: options.query,
    });
    stateManager.updateSearchEvent(eventId, {
      resultCount: 0,
      latencyMs: Date.now() - startTime,
    });
    return unresolvedTenantResponse(options);
  }

  const tenantId = resolution.kind === 'tenant' ? resolution.tenantId : undefined;
  const effectiveBranch = resolveEffectiveBranch({
    explicitBranch: options.branch,
    scope: options.scope ?? 'project',
    projectId: tenantId,
    projectPath: resolution.kind === 'tenant' ? resolution.projectPath : undefined,
  });
  const effectiveOptions = applyEffectiveBranch(options, effectiveBranch);
  stateManager.logSearchEvent({
    id: eventId,
    projectId: tenantId,
    actor: options.telemetryActor ?? 'claude',
    tool: 'mcp_qdrant',
    op: 'search_exact',
    queryText: effectiveOptions.query,
    filters:
      effectiveBranch && effectiveBranch !== '*'
        ? JSON.stringify({ branch: effectiveBranch })
        : undefined,
  });

  return executeAndLogSearch(
    daemonClient,
    stateManager,
    effectiveOptions,
    tenantId,
    eventId,
    startTime
  );
}

async function executeAndLogSearch(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  options: SearchOptions,
  tenantId: string | undefined,
  eventId: string,
  startTime: number
): Promise<SearchResponse> {
  try {
    const request = buildExactSearchRequest(options, tenantId);
    const response = await daemonClient.textSearch(request);
    const results = mapExactResults(response.matches);

    stateManager.updateSearchEvent(eventId, {
      resultCount: results.length,
      latencyMs: Date.now() - startTime,
    });
    const successResponse: SearchResponse = {
      results,
      total: response.total_matches,
      query: options.query,
      mode: 'keyword',
      scope: options.scope ?? 'project',
      collections_searched: [PROJECTS_COLLECTION],
    };
    await attachIndexingProgress(
      successResponse,
      daemonClient,
      successResponse.scope,
      tenantId
    );
    return successResponse;
  } catch (error) {
    stateManager.updateSearchEvent(eventId, { resultCount: 0, latencyMs: Date.now() - startTime });
    // Don't attach indexing on the error path: the daemon just failed
    // a different RPC, so the cached probe is unlikely to be fresh.
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
