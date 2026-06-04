/**
 * Rules tool argument builder — parse raw MCP tool arguments into RuleOptions
 */
/** Build rule options from raw tool arguments */
export function buildRuleOptions(args) {
    const action = args?.['action'];
    if (action !== 'add' && action !== 'update' && action !== 'remove' && action !== 'list') {
        throw new Error(`Invalid rules action: ${action}`);
    }
    const options = { action };
    const content = args?.['content'];
    if (content)
        options.content = content;
    const label = args?.['label'];
    if (label)
        options.label = label;
    const scope = args?.['scope'];
    if (scope === 'global' || scope === 'project')
        options.scope = scope;
    const projectId = args?.['projectId'];
    if (projectId)
        options.projectId = projectId;
    const title = args?.['title'];
    if (title)
        options.title = title;
    const tags = args?.['tags'];
    if (tags)
        options.tags = tags;
    const priority = args?.['priority'];
    if (priority !== undefined)
        options.priority = priority;
    const limit = args?.['limit'];
    if (limit !== undefined)
        options.limit = limit;
    return options;
}
//# sourceMappingURL=rules.js.map