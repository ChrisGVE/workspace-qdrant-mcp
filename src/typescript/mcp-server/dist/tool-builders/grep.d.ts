/**
 * Grep tool argument builder — parse raw MCP tool arguments into GrepOptions
 */
export type GrepOptions = {
    pattern: string;
    regex?: boolean;
    caseSensitive?: boolean;
    pathGlob?: string;
    scope?: 'project' | 'all';
    contextLines?: number;
    maxResults?: number;
    branch?: string;
    projectId?: string;
};
/** Build grep options from raw tool arguments */
export declare function buildGrepOptions(args: Record<string, unknown> | undefined): GrepOptions;
//# sourceMappingURL=grep.d.ts.map