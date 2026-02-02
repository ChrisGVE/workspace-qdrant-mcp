/**
 * Type definitions for @anthropic-ai/claude-agent-sdk
 *
 * Based on the SDK v0.2.x documentation.
 * These types supplement the SDK's built-in types.
 */

declare module '@anthropic-ai/claude-agent-sdk' {
  /**
   * Hook event types
   */
  export type HookEvent =
    | 'PreToolUse'
    | 'PostToolUse'
    | 'PostToolUseFailure'
    | 'Notification'
    | 'UserPromptSubmit'
    | 'SessionStart'
    | 'SessionEnd'
    | 'Stop'
    | 'SubagentStart'
    | 'SubagentStop'
    | 'PreCompact'
    | 'PermissionRequest';

  /**
   * Base hook input shared by all hook types
   */
  export interface BaseHookInput {
    hook_event_name: HookEvent;
  }

  /**
   * SessionStart hook input
   */
  export interface SessionStartHookInput extends BaseHookInput {
    hook_event_name: 'SessionStart';
    source: 'startup' | 'resume' | 'clear' | 'compact';
  }

  /**
   * SessionEnd hook input
   */
  export interface SessionEndHookInput extends BaseHookInput {
    hook_event_name: 'SessionEnd';
    reason: string;
  }

  /**
   * Hook callback function type
   */
  export type HookCallback = (
    input: BaseHookInput | SessionStartHookInput | SessionEndHookInput,
    toolUseId: string | undefined,
    options: { signal: AbortSignal }
  ) => Promise<HookJSONOutput>;

  /**
   * Hook output with optional system message
   */
  export interface HookJSONOutput {
    systemMessage?: string;
    hookSpecificOutput?: {
      hookEventName: string;
      updatedPrompt?: string;
    };
    [key: string]: unknown;
  }

  /**
   * Hook matcher for registering callbacks
   */
  export interface HookMatcher {
    hooks: HookCallback[];
    toolNames?: string[];
  }

  /**
   * System prompt preset configuration
   */
  export interface SystemPromptPreset {
    type: 'preset';
    preset: 'claude_code';
    append?: string;
  }

  /**
   * MCP server configuration - stdio transport
   */
  export interface McpServerStdio {
    command: string;
    args?: string[];
    env?: Record<string, string>;
  }

  /**
   * MCP server configuration - HTTP transport
   */
  export interface McpServerHttp {
    type: 'http';
    url: string;
  }

  /**
   * MCP server configuration union
   */
  export type McpServerConfig = McpServerStdio | McpServerHttp;

  /**
   * Agent options for query function
   */
  export interface ClaudeAgentOptions {
    /**
     * Custom system prompt (string or preset)
     */
    systemPrompt?: string | SystemPromptPreset;

    /**
     * Model to use
     */
    model?: string;

    /**
     * Session ID to resume
     */
    resume?: string;

    /**
     * Hook configurations by event type
     */
    hooks?: Partial<Record<HookEvent, HookMatcher[]>>;

    /**
     * MCP server configurations
     */
    mcpServers?: Record<string, McpServerConfig>;

    /**
     * Allowed tools (supports wildcards)
     */
    allowedTools?: string[];
  }

  /**
   * Query options wrapper
   */
  export interface QueryOptions {
    prompt: string;
    options?: ClaudeAgentOptions;
  }

  /**
   * Text content block
   */
  export interface TextBlock {
    type: 'text';
    text: string;
  }

  /**
   * Tool use content block
   */
  export interface ToolUseBlock {
    type: 'tool_use';
    id: string;
    name: string;
    input: Record<string, unknown>;
  }

  /**
   * Content block union
   */
  export type ContentBlock = TextBlock | ToolUseBlock;

  /**
   * Assistant message
   */
  export interface AssistantMessage {
    type: 'assistant';
    message: {
      content: ContentBlock[];
    };
    session_id: string;
  }

  /**
   * Result message
   */
  export interface ResultMessage {
    type: 'result';
    subtype: 'success' | 'error';
    result?: string;
    error?: string;
  }

  /**
   * Query message union
   */
  export type QueryMessage = AssistantMessage | ResultMessage;

  /**
   * Main query function
   */
  export function query(options: QueryOptions): AsyncIterable<QueryMessage>;
}
