/**
 * Claude Agent SDK integration for session hooks and memory rule injection.
 *
 * Wraps the MCP server with Agent SDK to enable:
 * - SessionStart hook for memory rule fetching and injection
 * - SessionEnd hook for cleanup
 * - systemPrompt injection for memory rules
 */

import { query } from '@anthropic-ai/claude-agent-sdk';
import type {
  ClaudeAgentOptions,
  HookCallback,
  HookMatcher,
  BaseHookInput,
} from '@anthropic-ai/claude-agent-sdk';

import { loadConfig } from './config.js';
import { ProjectDetector } from './utils/project-detector.js';
import { type Rule, fetchRules, formatRulesForPrompt } from './agent-rules.js';

// Session state for agent lifecycle
interface AgentSessionState {
  sessionId: string | null;
  projectId: string | null;
  projectPath: string | null;
  rules: Rule[];
}

const sessionState: AgentSessionState = {
  sessionId: null, projectId: null, projectPath: null, rules: [],
};

/** SessionStart hook — fetch memory rules and inject into system prompt. */
const sessionStartHook: HookCallback = async (
  input: BaseHookInput,
  _toolUseId: string | undefined,
  _context: { signal: AbortSignal },
) => {
  if (input.hook_event_name !== 'SessionStart') return {};

  const sessionInput = input as BaseHookInput & { source: string };
  console.log(`[Agent] SessionStart hook fired (source: ${sessionInput.source})`);

  try {
    const config = loadConfig();
    const cwd = process.cwd();
    const projectDetector = new ProjectDetector();
    const projectInfo = await projectDetector.getProjectInfo(cwd);

    if (projectInfo) {
      sessionState.projectPath = projectInfo.projectPath;
      sessionState.projectId = projectInfo.projectId;
      console.log(`[Agent] Project detected: ${projectInfo.projectPath} (tenant_id: ${projectInfo.projectId})`);
    }

    sessionState.rules = await fetchRules(sessionState.projectId, config);

    if (sessionState.rules.length > 0) {
      return { systemMessage: formatRulesForPrompt(sessionState.rules) };
    }
    return {};
  } catch (error) {
    console.error('[Agent] Error in SessionStart hook:', error);
    return {};
  }
};

/** SessionEnd hook — clean up session state. */
const sessionEndHook: HookCallback = async (
  input: BaseHookInput,
  _toolUseId: string | undefined,
  _context: { signal: AbortSignal },
) => {
  if (input.hook_event_name !== 'SessionEnd') return {};

  const sessionInput = input as BaseHookInput & { reason: string };
  console.log(`[Agent] SessionEnd hook fired (reason: ${sessionInput.reason})`);

  try {
    sessionState.sessionId = null;
    sessionState.projectId = null;
    sessionState.projectPath = null;
    sessionState.rules = [];
    return {};
  } catch (error) {
    console.error('[Agent] Error in SessionEnd hook:', error);
    return {};
  }
};

/** Build agent options with hooks and MCP server configuration. */
function buildAgentOptions(): ClaudeAgentOptions {
  const config = loadConfig();
  const mcpServerPath = process.argv[1] ?? 'workspace-qdrant-mcp';

  return {
    hooks: {
      SessionStart: [{ hooks: [sessionStartHook] } as HookMatcher],
      SessionEnd: [{ hooks: [sessionEndHook] } as HookMatcher],
    },
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
    allowedTools: [
      'mcp__workspace-qdrant__search',
      'mcp__workspace-qdrant__retrieve',
      'mcp__workspace-qdrant__rules',
      'mcp__workspace-qdrant__store',
    ],
  };
}

/** Run the agent with optional initial prompt. */
export async function runAgent(prompt?: string): Promise<void> {
  console.log('[Agent] Starting workspace-qdrant agent...');
  const options = buildAgentOptions();

  if (sessionState.rules.length > 0) {
    options.systemPrompt = {
      type: 'preset',
      preset: 'claude_code',
      append: formatRulesForPrompt(sessionState.rules),
    };
  }

  try {
    for await (const message of query({ prompt: prompt ?? '', options })) {
      if (message.type === 'assistant') {
        for (const block of message.message.content) {
          if (block.type === 'text') console.log(block.text);
        }
      } else if (message.type === 'result') {
        if (message.subtype === 'success') console.log('[Agent] Session completed successfully');
        else console.error('[Agent] Session ended with error:', message.error);
      }
    }
  } catch (error) {
    console.error('[Agent] Error running agent:', error);
    throw error;
  }
}

/** Agent entry point. */
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
`);
    return;
  }

  if (args.includes('--mcp-only')) {
    const { startServer } = await import('./index.js');
    await startServer();
    return;
  }

  const prompt = args.filter(arg => !arg.startsWith('--')).join(' ');
  await runAgent(prompt || undefined);
}

const currentFile = import.meta.url;
const mainModule = `file://${process.argv[1]}`;
if (currentFile === mainModule || currentFile === mainModule.replace(/\.js$/, '.ts')) {
  main().catch((error) => {
    console.error('[Agent] Fatal error:', error);
    process.exit(1);
  });
}
