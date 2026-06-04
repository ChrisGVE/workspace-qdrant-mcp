/**
 * Grep tool argument builder — parse raw MCP tool arguments into GrepOptions
 */
/** Build grep options from raw tool arguments */
export function buildGrepOptions(args) {
    const pattern = args?.['pattern'];
    if (!pattern) {
        throw new Error('Pattern is required for grep operation');
    }
    const options = { pattern };
    const regex = args?.['regex'];
    if (regex !== undefined)
        options.regex = regex;
    const caseSensitive = args?.['caseSensitive'];
    if (caseSensitive !== undefined)
        options.caseSensitive = caseSensitive;
    const pathGlob = args?.['pathGlob'];
    if (pathGlob)
        options.pathGlob = pathGlob;
    const scope = args?.['scope'];
    if (scope === 'project' || scope === 'all')
        options.scope = scope;
    const contextLines = args?.['contextLines'];
    if (contextLines !== undefined)
        options.contextLines = contextLines;
    const maxResults = args?.['maxResults'];
    if (maxResults !== undefined)
        options.maxResults = maxResults;
    const branch = args?.['branch'];
    if (branch)
        options.branch = branch;
    const projectId = args?.['projectId'];
    if (projectId)
        options.projectId = projectId;
    return options;
}
//# sourceMappingURL=grep.js.map