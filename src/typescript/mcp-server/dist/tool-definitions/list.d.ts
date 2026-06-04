/**
 * MCP tool schema definition for the 'list' tool
 */
export declare const listToolDefinition: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            path: {
                type: string;
                description: string;
            };
            depth: {
                type: string;
                description: string;
            };
            format: {
                type: string;
                enum: string[];
                description: string;
            };
            fileType: {
                type: string;
                description: string;
            };
            language: {
                type: string;
                description: string;
            };
            extension: {
                type: string;
                description: string;
            };
            pattern: {
                type: string;
                description: string;
            };
            includeTests: {
                type: string;
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
            component: {
                type: string;
                description: string;
            };
            branch: {
                type: string;
                description: string;
            };
        };
    };
};
//# sourceMappingURL=list.d.ts.map