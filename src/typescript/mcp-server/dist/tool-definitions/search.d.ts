/**
 * MCP tool schema definition for the 'search' tool
 */
export declare const searchToolDefinition: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            query: {
                type: string;
                description: string;
            };
            collection: {
                type: string;
                enum: string[];
                description: string;
            };
            mode: {
                type: string;
                enum: string[];
                description: string;
            };
            scope: {
                type: string;
                enum: string[];
                description: string;
            };
            limit: {
                type: string;
                description: string;
            };
            projectId: {
                type: string;
                description: string;
            };
            libraryName: {
                type: string;
                description: string;
            };
            libraryPath: {
                type: string;
                description: string;
            };
            branch: {
                type: string;
                description: string;
            };
            fileType: {
                type: string;
                description: string;
            };
            scoreThreshold: {
                type: string;
                description: string;
            };
            includeLibraries: {
                type: string;
                description: string;
            };
            tag: {
                type: string;
                description: string;
            };
            tags: {
                type: string;
                items: {
                    type: string;
                };
                description: string;
            };
            pathGlob: {
                type: string;
                description: string;
            };
            component: {
                type: string;
                description: string;
            };
            exact: {
                type: string;
                description: string;
            };
            contextLines: {
                type: string;
                description: string;
            };
            includeGraphContext: {
                type: string;
                description: string;
            };
            diverse: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=search.d.ts.map