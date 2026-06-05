#!/usr/bin/env node
/**
 * workspace-qdrant-mcp - MCP Server Entry Point
 *
 * Provides project-scoped Qdrant vector database operations with hybrid search.
 * Exactly 4 tools: search, retrieve, rules, store
 *
 * Architecture:
 * - TypeScript MCP server using @modelcontextprotocol/sdk
 * - Communicates with Rust daemon (memexd) via gRPC for session management
 * - Direct Qdrant access for read operations
 * - SQLite queue for write operations (fallback when daemon unavailable)
 */

import { loadConfig } from './config.js';
import { createServer, WorkspaceQdrantMcpServer } from './server.js';
import {
  DEFAULT_HTTP_HOST,
  DEFAULT_HTTP_PORT,
  DEFAULT_HTTP_PATH,
  type HttpTransportOptions,
  type ServerMode,
} from './server-types.js';
import { pushMetricsOnExit } from './telemetry/metrics.js';
import { startMetricsServer } from './telemetry/http-server.js';

/**
 * Resolve the server transport mode from environment variables.
 *
 * - `MCP_SERVER_MODE=http` forces HTTP mode explicitly (preferred).
 * - Legacy `WQM_HTTP_MODE` / `WQM_CLI_MODE` env vars also disable stdio; they
 *   are kept because existing docker/compose configs set them.
 */
function resolveServerMode(): ServerMode {
  const explicit = process.env['MCP_SERVER_MODE']?.toLowerCase();
  if (explicit === 'http') return 'http';
  if (explicit === 'stdio') return 'stdio';
  if (process.env['WQM_HTTP_MODE'] || process.env['WQM_CLI_MODE']) return 'http';
  return 'stdio';
}

function resolveHttpOptions(): HttpTransportOptions {
  const portEnv = process.env['MCP_HTTP_PORT'];
  const parsed = portEnv ? Number.parseInt(portEnv, 10) : NaN;
  const port = Number.isFinite(parsed) && parsed > 0 && parsed < 65536 ? parsed : DEFAULT_HTTP_PORT;
  const host = process.env['MCP_HTTP_HOST'] ?? DEFAULT_HTTP_HOST;
  const path = process.env['MCP_HTTP_PATH'] ?? DEFAULT_HTTP_PATH;
  const options: HttpTransportOptions = { host, port, path };

  // Optional native TLS. Leave unset to run plain HTTP behind a reverse proxy
  // (Caddy/Traefik), which is the recommended production deployment.
  const certPath = process.env['MCP_HTTP_TLS_CERT'];
  const keyPath = process.env['MCP_HTTP_TLS_KEY'];
  if (certPath && keyPath) {
    const caPath = process.env['MCP_HTTP_TLS_CA'];
    options.tls = caPath ? { certPath, keyPath, caPath } : { certPath, keyPath };
  } else if (certPath || keyPath) {
    throw new Error('MCP_HTTP_TLS_CERT and MCP_HTTP_TLS_KEY must both be set to enable native TLS');
  }

  return options;
}

const serverMode: ServerMode = resolveServerMode();
const isStdioMode = serverMode === 'stdio';

// In stdio mode, suppress all console.log to prevent protocol contamination
// Keep console.error and console.warn for critical issues (sent to stderr)
if (isStdioMode) {
  console.log = (): void => {};
  console.info = (): void => {};
  console.debug = (): void => {};
}

let server: WorkspaceQdrantMcpServer | null = null;

function registerSignalHandlers(shutdown: () => Promise<void>): void {
  const handleSignal = (): void => {
    shutdown().catch((error) => {
      console.error('Error during shutdown:', error);
      process.exit(1);
    });
  };
  process.on('SIGINT', handleSignal);
  process.on('SIGTERM', handleSignal);

  // Survive transient faults instead of crashing. The decisive case is a daemon
  // (memexd) restart: it drops the gRPC channel mid-session, and the session
  // cleanup path previously surfaced a stray unhandled rejection
  // (RangeError during retry/teardown) that, with no handler installed, took the
  // whole MCP process down — killing every connected client's SSE session
  // (observed as "SSE stream disconnected" + 404 on the client's reconnect).
  // Log and keep running; the gRPC client self-heals on the next request and the
  // existing session stays alive.
  process.on('unhandledRejection', (reason) => {
    console.error('[wqm] Unhandled promise rejection (non-fatal):', reason);
  });
  process.on('uncaughtException', (error) => {
    console.error('[wqm] Uncaught exception (non-fatal):', error);
  });

  if (isStdioMode) {
    process.on('exit', () => {
      process.stderr.write('[wqm-metrics] process exit\n');
    });
  }
}

async function createAndStartServer(
  config: ReturnType<typeof loadConfig>
): Promise<WorkspaceQdrantMcpServer> {
  if (serverMode === 'http') {
    const httpOptions = resolveHttpOptions();
    const srv = await createServer(config, 'http', httpOptions);
    startMetricsServer();
    return srv;
  }
  return createServer(config, serverMode);
}

async function main(): Promise<void> {
  const config = loadConfig();

  const shutdown = async (): Promise<void> => {
    if (server) await server.stop();
    if (isStdioMode) await pushMetricsOnExit();
    process.exit(0);
  };

  registerSignalHandlers(shutdown);

  try {
    server = await createAndStartServer(config);

    const sessionState = server.getSessionState();
    if (!isStdioMode) {
      console.log('workspace-qdrant-mcp server started');
      console.log(`Mode: ${serverMode}`);
      console.log(`Session ID: ${sessionState.sessionId}`);
      console.log(`Project: ${sessionState.projectPath ?? 'none'}`);
      console.log(`Project ID: ${sessionState.projectId ?? 'none'}`);
      // Daemon connection state is dynamic and not yet established at this
      // point in startup — exposing it here would always print "false". The
      // live status is available via the `:9092/metrics` endpoint and the
      // health-monitor utility used by MCP tools.
    }
  } catch (error) {
    console.error('Failed to start MCP server:', error);
    process.exit(1);
  }
}

main().catch((error: unknown) => {
  console.error('Fatal error:', error);
  process.exit(1);
});

/**
 * Start the MCP server
 * Used by agent.ts when running in --mcp-only mode
 */
export async function startServer(): Promise<void> {
  await main();
}

// Export for testing
export { WorkspaceQdrantMcpServer, createServer, startServer as main };
