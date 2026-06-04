/**
 * List tool argument builder — parse raw MCP tool arguments into ListOptions
 */
/**
 * Build list options from raw tool arguments.
 *
 * @param args           Raw MCP tool arguments.
 * @param defaultBranch  Session's current branch, used when the caller does
 *                       not explicitly pass a `branch` argument. Pass `null`
 *                       or omit to skip the default. Pass the string `"*"` as
 *                       the `branch` argument to list files across all branches.
 */
export function buildListOptions(args, defaultBranch) {
    const options = {};
    const path = args?.['path'];
    if (path)
        options.path = path;
    const depth = args?.['depth'];
    if (depth !== undefined)
        options.depth = depth;
    const format = args?.['format'];
    if (format === 'tree' || format === 'summary' || format === 'flat')
        options.format = format;
    const fileType = args?.['fileType'];
    if (fileType)
        options.fileType = fileType;
    const language = args?.['language'];
    if (language)
        options.language = language;
    const extension = args?.['extension'];
    if (extension)
        options.extension = extension;
    const pattern = args?.['pattern'];
    if (pattern)
        options.pattern = pattern;
    const includeTests = args?.['includeTests'];
    if (includeTests !== undefined)
        options.includeTests = includeTests;
    const limit = args?.['limit'];
    if (limit !== undefined)
        options.limit = limit;
    const projectId = args?.['projectId'];
    if (projectId)
        options.projectId = projectId;
    const component = args?.['component'];
    if (component)
        options.component = component;
    const branch = args?.['branch'];
    if (branch === '*') {
        // Explicit wildcard — cross-branch listing, no filter applied
        options.branch = '*';
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
    return options;
}
//# sourceMappingURL=list.js.map