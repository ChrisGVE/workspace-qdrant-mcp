/**
 * Search tool argument builder — parse raw MCP tool arguments into SearchOptions
 */
export type SearchOptions = {
    query: string;
    collection?: string;
    mode?: 'hybrid' | 'semantic' | 'keyword';
    scope?: 'project' | 'group' | 'all';
    limit?: number;
    scoreThreshold?: number;
    projectId?: string;
    libraryName?: string;
    libraryPath?: string;
    branch?: string;
    fileType?: string;
    includeLibraries?: boolean;
    tag?: string;
    tags?: string[];
    pathGlob?: string;
    component?: string;
    exact?: boolean;
    contextLines?: number;
    includeGraphContext?: boolean;
    diverse?: boolean;
};
/**
 * Build search options from raw tool arguments.
 *
 * @param args           Raw MCP tool arguments.
 * @param defaultBranch  Session's current branch, used when the caller does
 *                       not explicitly pass a `branch` argument. Pass `null`
 *                       or omit to skip the default. Pass the string `"*"` as
 *                       the `branch` argument to bypass filtering entirely.
 */
export declare function buildSearchOptions(args: Record<string, unknown> | undefined, defaultBranch?: string | null): SearchOptions;
//# sourceMappingURL=search.d.ts.map