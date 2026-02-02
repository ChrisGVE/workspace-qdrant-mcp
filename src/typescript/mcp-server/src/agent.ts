/**
 * Claude Agent SDK integration for session hooks and memory rule injection
 *
 * This module wraps the MCP server functionality with the Claude Agent SDK
 * to enable:
 * - SessionStart hook for memory rule fetching and injection
 * - SessionEnd hook for cleanup
 * - systemPrompt injection for memory rules
 *
 * Architecture:
 * - MCP Server (server.ts): Provides tools via @modelcontextprotocol/sdk
 * - Agent Wrapper (this file): Manages session lifecycle via @anthropic-ai/claude-agent-sdk
 *
 * Usage:
 * - Run agent instead of MCP server directly for memory rule injection
 * - Agent connects to MCP server and injects memory rules at session start
 */

import { query } from '@anthropic-ai/claude-agent-sdk';
import type {
  ClaudeAgentOptions,
  HookCallback,
  HookMatcher,
  BaseHookInput,
} from '@anthropic-ai/claude-agent-sdk';

import { loadConfig } from './config.js';
import { SqliteStateManager } from './clients/sqlite-state-manager.js';
import { DaemonClient } from './clients/daemon-client.js';
import { ProjectDetector } from './utils/project-detector.js';
import { MemoryTool, type MemoryRule } from './tools/memory.js';

// Session state for agent lifecycle
interface AgentSessionState {
  sessionId: string | null;
  projectId: string | null;
  projectPath: string | null;
  memoryRules: MemoryRule[];
}

const sessionState: AgentSessionState = {
  sessionId: null,
  projectId: null,
  projectPath: null,
  memoryRules: [],
};

/**
 * Fetch memory rules from Qdrant via MemoryTool
 *
 * Fetches both global rules and project-specific rules (if project detected).
 * Rules are sorted by priority (highest first) and formatted for injection.
 */
async function fetchMemoryRules(
  projectId: string | null,
  config: ReturnType<typeof loadConfig>
): Promise<MemoryRule[]> {
  const rules: MemoryRule[] = [];

  try {
    // Initialize components needed for MemoryTool
    const daemonClient = new DaemonClient({
      port: config.daemon.grpcPort,
      timeoutMs: 5000,
    });

    const stateManager = new SqliteStateManager({
      dbPath: config.database.path.replace('~', process.env['HOME'] ?? ''),
    });
    await stateManager.initialize();

    const projectDetector = new ProjectDetector();

    // Create MemoryTool instance
    const memoryToolConfig = {
      qdrantUrl: config.qdrant?.url ?? 'http://localhost:6333',
      qdrantTimeout: 5000,
    } as { qdrantUrl: string; qdrantApiKey?: string; qdrantTimeout?: number };

    if (config.qdrant?.apiKey) {
      memoryToolConfig.qdrantApiKey = config.qdrant.apiKey;
    }

    const memoryTool = new MemoryTool(
      memoryToolConfig,
      daemonClient,
      stateManager,
      projectDetector
    );

    // Fetch global rules
    const globalResponse = await memoryTool.execute({
      action: 'list',
      scope: 'global',
      limit: 50,
    });

    if (globalResponse.success && globalResponse.rules) {
      rules.push(...globalResponse.rules);
      console.log(`[Agent] Fetched ${globalResponse.rules.length} global rule(s)`);
    }

    // Fetch project-specific rules if project detected
    if (projectId) {
      const projectResponse = await memoryTool.execute({
        action: 'list',
        scope: 'project',
        projectId,
        limit: 50,
      });

      if (projectResponse.success && projectResponse.rules) {
        rules.push(...projectResponse.rules);
        console.log(`[Agent] Fetched ${projectResponse.rules.length} project rule(s) for ${projectId}`);
      }
    }

    // Sort by priority (highest first), then by creation date (newest first)
    rules.sort((a, b) => {
      const priorityDiff = (b.priority ?? 0) - (a.priority ?? 0);
      if (priorityDiff !== 0) return priorityDiff;

      // Sort by createdAt (newest first) if priority is equal
      const aDate = a.createdAt ? new Date(a.createdAt).getTime() : 0;
      const bDate = b.createdAt ? new Date(b.createdAt).getTime() : 0;
      return bDate - aDate;
    });

    console.log(`[Agent] Total memory rules fetched: ${rules.length}`);
    return rules;
  } catch (error) {
    console.error('[Agent] Error fetching memory rules:', error);
    return rules;
  }
}

/**
 * Format memory rules for system prompt injection
 *
 * Organizes rules by scope and priority for clear presentation:
 * 1. Global rules (apply everywhere)
 * 2. Project-specific rules (apply to current project)
 *
 * Each rule shows title (if available), priority indicator, and content.
 */
function formatMemoryRulesForPrompt(rules: MemoryRule[]): string {
  if (rules.length === 0) {
    return '';
  }

  const lines: string[] = [
    '# Memory Rules',
    '',
    'The following behavioral rules have been configured and should be followed:',
    '',
  ];

  // Separate rules by scope
  const globalRules = rules.filter(r => r.scope === 'global');
  const projectRules = rules.filter(r => r.scope === 'project');

  // Format global rules
  if (globalRules.length > 0) {
    lines.push('## Global Rules');
    lines.push('');
    globalRules.forEach((rule, index) => {
      const title = rule.title ? `**${rule.title}**` : `Rule ${index + 1}`;
      const priority = rule.priority !== undefined ? ` [Priority: ${rule.priority}]` : '';
      lines.push(`### ${title}${priority}`);
      lines.push('');
      lines.push(rule.content);
      lines.push('');
    });
  }

  // Format project rules
  if (projectRules.length > 0) {
    lines.push('## Project-Specific Rules');
    lines.push('');
    projectRules.forEach((rule, index) => {
      const title = rule.title ? `**${rule.title}**` : `Rule ${index + 1}`;
      const priority = rule.priority !== undefined ? ` [Priority: ${rule.priority}]` : '';
      lines.push(`### ${title}${priority}`);
      lines.push('');
      lines.push(rule.content);
      lines.push('');
    });
  }

  return lines.join('\n');
}

/**
 * SessionStart hook callback
 * Called when a new Claude session begins
 */
const sessionStartHook: HookCallback = async (
  input: BaseHookInput,
  _toolUseId: string | undefined,
  _context: { signal: AbortSignal }
) => {
  // Type guard for SessionStart input
  if (input.hook_event_name !== 'SessionStart') {
    return {};
  }

  const sessionInput = input as BaseHookInput & { source: string };
  console.log(`[Agent] SessionStart hook fired (source: ${sessionInput.source})`);

  try {
    const config = loadConfig();

    // Detect current project
    const cwd = process.cwd();
    const projectDetector = new ProjectDetector();
    const projectRoot = projectDetector.findProjectRoot(cwd);

    if (projectRoot) {
      sessionState.projectPath = projectRoot;
      // Get project ID via async method
      const projectInfo = await projectDetector.getProjectInfo(projectRoot);
      if (projectInfo) {
        sessionState.projectId = projectInfo.projectId;
        console.log(`[Agent] Project detected: ${projectRoot} (tenant_id: ${projectInfo.projectId})`);
      }
    }

    // Fetch memory rules
    sessionState.memoryRules = await fetchMemoryRules(sessionState.projectId, config);

    // Return system message with memory rules if any
    if (sessionState.memoryRules.length > 0) {
      const formattedRules = formatMemoryRulesForPrompt(sessionState.memoryRules);
      return {
        systemMessage: formattedRules,
      };
    }

    return {};
  } catch (error) {
    console.error('[Agent] Error in SessionStart hook:', error);
    return {};
  }
};

/**
 * SessionEnd hook callback
 * Called when a Claude session ends
 */
const sessionEndHook: HookCallback = async (
  input: BaseHookInput,
  _toolUseId: string | undefined,
  _context: { signal: AbortSignal }
) => {
  // Type guard for SessionEnd input
  if (input.hook_event_name !== 'SessionEnd') {
    return {};
  }

  const sessionInput = input as BaseHookInput & { reason: string };
  console.log(`[Agent] SessionEnd hook fired (reason: ${sessionInput.reason})`);

  try {
    // Clean up session state
    sessionState.sessionId = null;
    sessionState.projectId = null;
    sessionState.projectPath = null;
    sessionState.memoryRules = [];

    // Additional cleanup can be added here
    // e.g., deprioritize project with daemon

    return {};
  } catch (error) {
    console.error('[Agent] Error in SessionEnd hook:', error);
    return {};
  }
};

/**
 * Build agent options with hooks and MCP server configuration
 */
function buildAgentOptions(): ClaudeAgentOptions {
  const config = loadConfig();

  // Get the MCP server command path
  const mcpServerPath = process.argv[1] ?? 'workspace-qdrant-mcp'; // Current script path or fallback

  const options: ClaudeAgentOptions = {
    // Configure hooks
    hooks: {
      SessionStart: [
        {
          hooks: [sessionStartHook],
        } as HookMatcher,
      ],
      SessionEnd: [
        {
          hooks: [sessionEndHook],
        } as HookMatcher,
      ],
    },

    // Configure MCP servers
    mcpServers: {
      'workspace-qdrant': {
        command: 'node',
        args: [mcpServerPath, '--mcp-only'],
        env: {
          QDRANT_URL: config.qdrant?.url ?? 'http://localhost:6333',
          QDRANT_API_KEY: config.qdrant?.apiKey ?? '',
          WQM_SQLITE_DB_PATH: config.database.path,
        },
      },
    },

    // Allow all workspace-qdrant tools
    allowedTools: [
      'mcp__workspace-qdrant__search',
      'mcp__workspace-qdrant__retrieve',
      'mcp__workspace-qdrant__memory',
      'mcp__workspace-qdrant__store',
    ],
  };

  return options;
}

/**
 * Run the agent with optional initial prompt
 */
export async function runAgent(prompt?: string): Promise<void> {
  console.log('[Agent] Starting workspace-qdrant agent...');

  const options = buildAgentOptions();

  // If memory rules are already fetched, inject via systemPrompt
  if (sessionState.memoryRules.length > 0) {
    const formattedRules = formatMemoryRulesForPrompt(sessionState.memoryRules);
    options.systemPrompt = {
      type: 'preset',
      preset: 'claude_code',
      append: formattedRules,
    };
  }

  try {
    // Run agent query
    for await (const message of query({
      prompt: prompt ?? '',
      options,
    })) {
      // Process messages
      if (message.type === 'assistant') {
        // Assistant response
        const content = message.message.content;
        for (const block of content) {
          if (block.type === 'text') {
            console.log(block.text);
          }
        }
      } else if (message.type === 'result') {
        // Final result
        if (message.subtype === 'success') {
          console.log('[Agent] Session completed successfully');
        } else {
          console.error('[Agent] Session ended with error:', message.error);
        }
      }
    }
  } catch (error) {
    console.error('[Agent] Error running agent:', error);
    throw error;
  }
}

/**
 * Agent entry point
 */
export async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
workspace-qdrant-agent - Claude Agent with memory rule injection

Usage:
  workspace-qdrant-agent [options] [prompt]

Options:
  --help, -h     Show this help message
  --mcp-only     Run only the MCP server (no agent wrapper)

The agent wrapper provides:
  - Automatic memory rule injection at session start
  - Project detection and context awareness
  - Integration with workspace-qdrant MCP tools
`);
    return;
  }

  if (args.includes('--mcp-only')) {
    // Run MCP server directly (for use as child process)
    const { startServer } = await import('./index.js');
    await startServer();
    return;
  }

  // Run the agent
  const prompt = args.filter(arg => !arg.startsWith('--')).join(' ');
  await runAgent(prompt || undefined);
}

// Run if executed directly
const currentFile = import.meta.url;
const mainModule = `file://${process.argv[1]}`;
if (currentFile === mainModule || currentFile === mainModule.replace(/\.js$/, '.ts')) {
  main().catch((error) => {
    console.error('[Agent] Fatal error:', error);
    process.exit(1);
  });
}
