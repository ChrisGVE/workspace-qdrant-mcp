/**
 * MCP tool schema definition for the 'store' tool
 */
export declare const storeToolDefinition: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            type: {
                type: string;
                enum: string[];
                description: string;
            };
            content: {
                type: string;
                description: string;
            };
            libraryName: {
                type: string;
                description: string;
            };
            forProject: {
                type: string;
                description: string;
            };
            path: {
                type: string;
                description: string;
            };
            name: {
                type: string;
                description: string;
            };
            title: {
                type: string;
                description: string;
            };
            url: {
                type: string;
                description: string;
            };
            filePath: {
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
            sourceType: {
                type: string;
                enum: string[];
                description: string;
            };
            metadata: {
                type: string;
                additionalProperties: {
                    type: string;
                };
                description: string;
            };
        };
    };
};
//# sourceMappingURL=store.d.ts.map