/**
 * Claude Agent SDK integration for session hooks and rule injection.
 *
 * Wraps the MCP server with Agent SDK to enable:
 * - SessionStart hook for rule fetching and injection
 * - SessionEnd hook for cleanup
 * - systemPrompt injection for rules
 */
/** Run the agent with optional initial prompt. */
export declare function runAgent(prompt?: string): Promise<void>;
/** Agent entry point. */
export declare function main(): Promise<void>;
//# sourceMappingURL=agent.d.ts.map