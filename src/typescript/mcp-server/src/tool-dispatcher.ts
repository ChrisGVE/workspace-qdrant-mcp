/**
 * Tool dispatcher — executes a named MCP tool and returns its result.
 *
 * Extracted from WorkspaceQdrantMcpServer.handleToolCall to keep server.ts
 * within the 300-line file-size limit.
 */

import type { SessionState } from './server-types.js';
import { logToolCall } from './utils/logger.js';
import type { ServerComponents } from './server-factory.js';
import {
  buildSearchOptions,
  buildRetrieveOptions,
  buildRuleOptions,
  buildStoreOptions,
  buildGrepOptions,
  buildListOptions,
} from './tool-builders/index.js';
import { storeUrl, storeScratchpad } from './store-handlers.js';
import { handleEmbedding } from './tools/embedding.js';
import { registerProjectFromTool, sendHeartbeat } from './session-lifecycle.js';
import { withToolMetrics } from './telemetry/metrics.js';

export type ToolResult = {
  content: Array<{ type: string; text: string }>;
  isError?: boolean;
};

const KNOWN_TOOLS = ['search', 'retrieve', 'rules', 'store', 'grep', 'list', 'embedding'] as const;

/** Dispatch the 'store' tool subtypes. */
async function dispatchStore(
  args: Record<string, unknown> | undefined,
  components: ServerComponents,
  sessionState: SessionState
): Promise<unknown> {
  const storeType = (args?.['type'] as string) ?? 'library';
  if (storeType === 'project')
    return registerProjectFromTool(args, sessionState, components.daemonClient);
  if (storeType === 'url') return storeUrl(args, components.stateManager, sessionState);
  if (storeType === 'scratchpad')
    return storeScratchpad(args, components.stateManager, sessionState);
  return components.storeTool.store(buildStoreOptions(args, sessionState));
}

/**
 * Dispatch a tool call to the appropriate handler.
 *
 * Fires an implicit heartbeat (fire-and-forget) before dispatching so that
 * active sessions keep their daemon connection alive without adding latency.
 */
export async function dispatchToolCall(
  toolName: string,
  args: Record<string, unknown> | undefined,
  components: ServerComponents,
  sessionState: SessionState
): Promise<ToolResult> {
  const startTime = Date.now();
  const { searchTool, retrieveTool, rulesTool, grepTool, listTool, healthMonitor, daemonClient } =
    components;

  sendHeartbeat(sessionState, daemonClient);

  if (!KNOWN_TOOLS.includes(toolName as (typeof KNOWN_TOOLS)[number])) {
    logToolCall(toolName, Date.now() - startTime, false, { error: 'Unknown tool' });
    return { content: [{ type: 'text', text: `Unknown tool: ${toolName}` }], isError: true };
  }

  try {
    const result = await withToolMetrics(toolName, async () => {
      switch (toolName) {
        case 'search': {
          const searchResult = await searchTool.search(buildSearchOptions(args));
          return healthMonitor.augmentSearchResults({ success: true, ...searchResult });
        }
        case 'retrieve':
          return retrieveTool.retrieve(buildRetrieveOptions(args));
        case 'rules':
          return rulesTool.execute(buildRuleOptions(args));
        case 'store':
          return dispatchStore(args, components, sessionState);
        case 'grep':
          return grepTool.grep(buildGrepOptions(args));
        case 'list':
          return listTool.list(buildListOptions(args));
        case 'embedding':
          return handleEmbedding(args, daemonClient);
        default:
          throw new Error(`Unexpected tool: ${toolName}`);
      }
    });

    logToolCall(toolName, Date.now() - startTime, true);
    return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logToolCall(toolName, Date.now() - startTime, false, { error: errorMessage });
    return { content: [{ type: 'text', text: `Error: ${errorMessage}` }], isError: true };
  }
}
