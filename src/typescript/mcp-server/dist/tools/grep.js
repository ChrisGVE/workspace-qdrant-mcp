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
/** Build the text search request object for the daemon. */
function buildGrepRequest(pattern, regex, caseSensitive, contextLines, maxResults, tenantId, branch, pathGlob) {
    const request = {
        pattern,
        regex,
        case_sensitive: caseSensitive,
        context_lines: contextLines,
        max_results: maxResults,
    };
    if (tenantId)
        request.tenant_id = tenantId;
    if (branch)
        request.branch = branch;
    if (pathGlob)
        request.path_glob = pathGlob;
    return request;
}
/** Map daemon TextSearchMatch array to GrepMatch array. */
function mapGrepMatches(matches) {
    return matches.map((m) => ({
        file: m.file_path,
        line: m.line_number,
        content: m.content,
        context_before: m.context_before ?? [],
        context_after: m.context_after ?? [],
    }));
}
/** Build an empty failure GrepResponse. */
function grepError(message, latency_ms) {
    return { success: false, matches: [], total_matches: 0, truncated: false, latency_ms, message };
}
/**
 * Grep tool for FTS5-based code search
 */
export class GrepTool {
    daemonClient;
    projectDetector;
    constructor(daemonClient, projectDetector) {
        this.daemonClient = daemonClient;
        this.projectDetector = projectDetector;
    }
    /**
     * Search code using FTS5 trigram index
     */
    async grep(options) {
        const { pattern, regex = false, caseSensitive = true, pathGlob, scope = 'project', contextLines = 0, maxResults = 1000, branch, projectId, } = options;
        if (!pattern)
            return grepError('Search pattern is required', 0);
        const startTime = Date.now();
        let tenantId;
        if (scope === 'project') {
            tenantId = projectId ?? (await this.resolveProjectId());
            if (!tenantId) {
                return grepError('Could not detect project ID. Use scope "all" or provide projectId.', Date.now() - startTime);
            }
        }
        return this.executeSearch(pattern, regex, caseSensitive, contextLines, maxResults, tenantId, branch, pathGlob, startTime);
    }
    async executeSearch(pattern, regex, caseSensitive, contextLines, maxResults, tenantId, branch, pathGlob, startTime) {
        try {
            const request = buildGrepRequest(pattern, regex, caseSensitive, contextLines, maxResults, tenantId, branch, pathGlob);
            const response = await this.daemonClient.textSearch(request);
            return {
                success: true,
                matches: mapGrepMatches(response.matches),
                total_matches: response.total_matches,
                truncated: response.truncated,
                latency_ms: Date.now() - startTime,
            };
        }
        catch (error) {
            return grepError(`Grep failed: ${error instanceof Error ? error.message : 'unknown error'}`, Date.now() - startTime);
        }
    }
    /**
     * Resolve project ID from current working directory
     */
    async resolveProjectId() {
        const cwd = process.cwd();
        const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
        return projectInfo?.projectId;
    }
}
//# sourceMappingURL=grep.js.map