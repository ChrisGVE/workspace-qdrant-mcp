/**
 * MCP tool schema definition for the 'grep' tool
 */
export declare const grepToolDefinition: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            pattern: {
                type: string;
                description: string;
            };
            regex: {
                type: string;
                description: string;
            };
            caseSensitive: {
                type: string;
                description: string;
            };
            pathGlob: {
                type: string;
                description: string;
            };
            scope: {
                type: string;
                enum: string[];
                description: string;
            };
            contextLines: {
                type: string;
                description: string;
            };
            maxResults: {
                type: string;
                description: string;
            };
            branch: {
                type: string;
                description: string;
            };
            projectId: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=grep.d.ts.map