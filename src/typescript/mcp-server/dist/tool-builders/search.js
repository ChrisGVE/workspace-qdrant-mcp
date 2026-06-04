/**
 * Search tool argument builder — parse raw MCP tool arguments into SearchOptions
 */
// ── Option group extractors ───────────────────────────────────────────────
function extractScopeOptions(args, options) {
    const collection = args?.['collection'];
    if (collection)
        options.collection = collection;
    const mode = args?.['mode'];
    if (mode === 'hybrid' || mode === 'semantic' || mode === 'keyword')
        options.mode = mode;
    const scope = args?.['scope'];
    if (scope === 'project' || scope === 'group' || scope === 'all')
        options.scope = scope;
    const limit = args?.['limit'];
    if (limit !== undefined)
        options.limit = limit;
    const scoreThreshold = args?.['scoreThreshold'];
    if (scoreThreshold !== undefined)
        options.scoreThreshold = scoreThreshold;
}
function extractIdentifierOptions(args, options, defaultBranch) {
    const projectId = args?.['projectId'];
    if (projectId)
        options.projectId = projectId;
    const libraryName = args?.['libraryName'];
    if (libraryName)
        options.libraryName = libraryName;
    const libraryPath = args?.['libraryPath'];
    if (libraryPath)
        options.libraryPath = libraryPath;
    const branch = args?.['branch'];
    if (branch === '*') {
        // Explicit wildcard — cross-branch search, no filter applied
    }
    else if (branch) {
        options.branch = branch;
    }
    else if (defaultBranch && defaultBranch !== 'default') {
        // Fall back to the session's current branch when not explicitly provided.
        // Skip when the sentinel value "default" is set — that indicates the
        // session is not inside a git repository and no branch filter should apply.
        options.branch = defaultBranch;
    }
    const fileType = args?.['fileType'];
    if (fileType)
        options.fileType = fileType;
    const includeLibraries = args?.['includeLibraries'];
    if (includeLibraries !== undefined)
        options.includeLibraries = includeLibraries;
}
function extractFilterOptions(args, options) {
    const tag = args?.['tag'];
    if (tag)
        options.tag = tag;
    const tags = args?.['tags'];
    if (tags && tags.length > 0)
        options.tags = tags;
    const pathGlob = args?.['pathGlob'];
    if (pathGlob)
        options.pathGlob = pathGlob;
    const component = args?.['component'];
    if (component)
        options.component = component;
}
function extractOutputOptions(args, options) {
    const exact = args?.['exact'];
    if (exact !== undefined)
        options.exact = exact;
    const contextLines = args?.['contextLines'];
    if (contextLines !== undefined)
        options.contextLines = contextLines;
    const includeGraphContext = args?.['includeGraphContext'];
    if (includeGraphContext !== undefined)
        options.includeGraphContext = includeGraphContext;
    const diverse = args?.['diverse'];
    if (diverse !== undefined)
        options.diverse = diverse;
}
/**
 * Build search options from raw tool arguments.
 *
 * @param args           Raw MCP tool arguments.
 * @param defaultBranch  Session's current branch, used when the caller does
 *                       not explicitly pass a `branch` argument. Pass `null`
 *                       or omit to skip the default. Pass the string `"*"` as
 *                       the `branch` argument to bypass filtering entirely.
 */
export function buildSearchOptions(args, defaultBranch) {
    const options = { query: args?.['query'] ?? '' };
    extractScopeOptions(args, options);
    extractIdentifierOptions(args, options, defaultBranch);
    extractFilterOptions(args, options);
    extractOutputOptions(args, options);
    return options;
}
//# sourceMappingURL=search.js.map