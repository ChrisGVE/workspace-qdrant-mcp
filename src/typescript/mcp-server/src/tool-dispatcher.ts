/**
 * Tool dispatcher ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â executes a named MCP tool and returns its result.
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
import { handleWorkspaceIndex } from './tools/workspace-index.js';
import { handleGraph } from './tools/graph.js';
import { runSearchEval } from './tools/search-eval.js';
import { ensureProjectFresh, registerProjectFromTool, sendHeartbeat } from './session-lifecycle.js';
import { withToolMetrics } from './telemetry/metrics.js';
import mcpPublicConfig from './constants/mcp-public-config.json' with { type: 'json' };

export type ToolResult = {
  content: Array<{ type: string; text: string }>;
  isError?: boolean;
};

// Derived from src/constants/mcp-public-config.json (single source of truth).
// publicTools = tools exposed in client `enabled_tools` lists.
// internalTools = tools the server accepts but not advertised to clients by default.
const KNOWN_TOOLS = [...mcpPublicConfig.publicTools, ...mcpPublicConfig.internalTools] as const;

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
    return storeScratchpad(args, components.stateManager, components.projectDetector, sessionState);
  return components.storeTool.store(buildStoreOptions(args, sessionState));
}

/**
 * Dispatch a tool call to the appropriate handler.
 *
 * Fires an implicit heartbeat (fire-and-forget) before dispatching so that
 * active sessions keep their daemon connection alive without adding latency.
 */
/** Route a validated tool name to its handler and return the raw result. */
async function routeTool(
  toolName: string,
  args: Record<string, unknown> | undefined,
  components: ServerComponents,
  sessionState: SessionState
): Promise<unknown> {
  const {
    searchTool,
    retrieveTool,
    rulesTool,
    grepTool,
    listTool,
    healthMonitor,
    daemonClient,
    projectDetector,
  } = components;
  switch (toolName) {
    case 'search': {
      const searchResult = await searchTool.search(
        buildSearchOptions(args, { branch: sessionState.currentBranch })
      );
      return healthMonitor.augmentSearchResults({ success: true, ...searchResult });
    }
    case 'retrieve':
      return retrieveTool.retrieve(buildRetrieveOptions(args));
    case 'rules':
      return rulesTool.execute(buildRuleOptions(args));
    case 'store':
      return dispatchStore(args, components, sessionState);
    case 'grep':
      return grepTool.grep(buildGrepOptions(args, { branch: sessionState.currentBranch }));
    case 'list':
      return listTool.list(buildListOptions(args));
    case 'embedding':
      return handleEmbedding(args, daemonClient);
    case 'workspace_index':
      return handleWorkspaceIndex(args, daemonClient);
    case 'graph':
      return handleGraph(args, daemonClient);
    case 'search_eval':
      return runSearchEval(searchTool, projectDetector, args);
    default:
      throw new Error(`Unexpected tool: ${toolName}`);
  }
}

export async function dispatchToolCall(
  toolName: string,
  args: Record<string, unknown> | undefined,
  components: ServerComponents,
  sessionState: SessionState
): Promise<ToolResult> {
  const startTime = Date.now();

  sendHeartbeat(sessionState, components.daemonClient);

  // Refresh cached git state (branch + worktree flag) if stale. Cheap inside
  // the TTL window; ~3ms `git` invocation outside it. Search/grep read
  // `sessionState.currentBranch` as default when the caller omits `branch`.
  ensureProjectFresh(sessionState);

  if (!KNOWN_TOOLS.includes(toolName as (typeof KNOWN_TOOLS)[number])) {
    logToolCall(toolName, Date.now() - startTime, false, { error: 'Unknown tool' });
    return { content: [{ type: 'text', text: `Unknown tool: ${toolName}` }], isError: true };
  }

  try {
    const result = await withToolMetrics(toolName, () =>
      routeTool(toolName, args, components, sessionState)
    );
    logToolCall(toolName, Date.now() - startTime, true);
    return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logToolCall(toolName, Date.now() - startTime, false, { error: errorMessage });
    return { content: [{ type: 'text', text: `Error: ${errorMessage}` }], isError: true };
  }
}
