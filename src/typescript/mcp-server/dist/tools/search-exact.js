/**
 * FTS5 exact/substring search via daemon's TextSearchService.
 */
import { randomUUID } from 'node:crypto';
import { PROJECTS_COLLECTION } from './search-types.js';
async function resolveExactSearchTenant(options, projectDetector) {
    if (options.scope === 'all')
        return { kind: 'unscoped' };
    if (options.projectId)
        return { kind: 'tenant', tenantId: options.projectId };
    const projectInfo = await projectDetector.getProjectInfo(process.cwd(), false);
    if (projectInfo?.projectId) {
        return { kind: 'tenant', tenantId: projectInfo.projectId };
    }
    return { kind: 'unresolved' };
}
/** Map daemon text search matches to SearchResult array. */
function mapExactResults(matches) {
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
function buildExactSearchRequest(options, tenantId) {
    const request = {
        pattern: options.query,
        regex: false,
        case_sensitive: true,
        context_lines: options.contextLines ?? 0,
        max_results: options.limit ?? 100,
    };
    if (tenantId)
        request.tenant_id = tenantId;
    if (options.branch)
        request.branch = options.branch;
    if (options.pathGlob)
        request.path_glob = options.pathGlob;
    return request;
}
/** Build the response returned when project-scope exact search has no
 * resolvable tenant. Closes F-004 (no broadening to all FTS tenants). */
function unresolvedTenantResponse(options) {
    return {
        results: [],
        total: 0,
        query: options.query,
        mode: 'keyword',
        scope: options.scope ?? 'project',
        collections_searched: [],
        status: 'uncertain',
        status_reason: 'Project scope requested but no project could be resolved. ' +
            'Pass `projectId` explicitly, run from a registered project directory, ' +
            'or set `scope: "all"` to search across every indexed tenant.',
    };
}
/**
 * Execute FTS5 exact/substring search via daemon's TextSearchService.
 * Maps TextSearchResponse to the standard SearchResponse format.
 */
export async function searchExact(daemonClient, stateManager, projectDetector, options) {
    const startTime = Date.now();
    const eventId = randomUUID();
    const resolution = await resolveExactSearchTenant(options, projectDetector);
    if (resolution.kind === 'unresolved') {
        // F-004: refuse to broaden to every tenant in the FTS index. The
        // pre-fix code path omitted `tenant_id` from the daemon request and
        // the Rust query builder then dropped its `fm.tenant_id = ?`
        // clause, returning cross-tenant matches.
        stateManager.logSearchEvent({
            id: eventId,
            actor: 'claude',
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
    stateManager.logSearchEvent({
        id: eventId,
        projectId: tenantId,
        actor: 'claude',
        tool: 'mcp_qdrant',
        op: 'search_exact',
        queryText: options.query,
    });
    return executeAndLogSearch(daemonClient, stateManager, options, tenantId, eventId, startTime);
}
async function executeAndLogSearch(daemonClient, stateManager, options, tenantId, eventId, startTime) {
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
    }
    catch (error) {
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
//# sourceMappingURL=search-exact.js.map