/**
 * MCP tool schema definition for the 'retrieve' tool
 */
export declare const retrieveToolDefinition: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            documentId: {
                type: string;
                description: string;
            };
            collection: {
                type: string;
                enum: string[];
                description: string;
            };
            filter: {
                type: string;
                additionalProperties: {
                    type: string;
                };
                description: string;
            };
            limit: {
                type: string;
                description: string;
            };
            offset: {
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
        };
    };
};
//# sourceMappingURL=retrieve.d.ts.map