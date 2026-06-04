/**
 * Tool dispatcher — executes a named MCP tool and returns its result.
 *
 * Extracted from WorkspaceQdrantMcpServer.handleToolCall to keep server.ts
 * within the 300-line file-size limit.
 */
import type { SessionState } from './server-types.js';
import type { ServerComponents } from './server-factory.js';
export type ToolResult = {
    content: Array<{
        type: string;
        text: string;
    }>;
    isError?: boolean;
};
export declare function dispatchToolCall(toolName: string, args: Record<string, unknown> | undefined, components: ServerComponents, sessionState: SessionState): Promise<ToolResult>;
//# sourceMappingURL=tool-dispatcher.d.ts.map