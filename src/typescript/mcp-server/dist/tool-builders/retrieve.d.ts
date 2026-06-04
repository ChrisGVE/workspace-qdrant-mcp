/**
 * Retrieve tool argument builder — parse raw MCP tool arguments into RetrieveOptions
 */
export type RetrieveOptions = {
    documentId?: string;
    collection?: 'projects' | 'libraries' | 'rules' | 'scratchpad';
    filter?: Record<string, string>;
    limit?: number;
    offset?: number;
    projectId?: string;
    libraryName?: string;
};
/** Build retrieve options from raw tool arguments */
export declare function buildRetrieveOptions(args: Record<string, unknown> | undefined): RetrieveOptions;
//# sourceMappingURL=retrieve.d.ts.map