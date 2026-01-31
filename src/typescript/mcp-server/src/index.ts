#!/usr/bin/env node
/**
 * workspace-qdrant-mcp - MCP Server Entry Point
 *
 * Provides project-scoped Qdrant vector database operations with hybrid search.
 * Exactly 4 tools: search, retrieve, memory, store
 *
 * Architecture:
 * - TypeScript MCP server using @modelcontextprotocol/sdk
 * - Communicates with Rust daemon (memexd) via gRPC for session management
 * - Direct Qdrant access for read operations
 * - SQLite queue for write operations (fallback when daemon unavailable)
 */

import { loadConfig } from './config.js';
import { createServer, WorkspaceQdrantMcpServer } from './server.js';

// Detect stdio mode (MCP protocol requires clean stdout)
const isStdioMode = !process.env['WQM_CLI_MODE'] && !process.env['WQM_HTTP_MODE'];

// In stdio mode, suppress all console.log to prevent protocol contamination
// Keep console.error and console.warn for critical issues (sent to stderr)
if (isStdioMode) {
  console.log = (): void => {};
  console.info = (): void => {};
  console.debug = (): void => {};
}

let server: WorkspaceQdrantMcpServer | null = null;

async function main(): Promise<void> {
  // Load configuration
  const config = loadConfig();

  // Handle graceful shutdown
  const shutdown = async (): Promise<void> => {
    if (server) {
      await server.stop();
    }
    process.exit(0);
  };

  process.on('SIGINT', () => {
    shutdown().catch((error) => {
      console.error('Error during shutdown:', error);
      process.exit(1);
    });
  });

  process.on('SIGTERM', () => {
    shutdown().catch((error) => {
      console.error('Error during shutdown:', error);
      process.exit(1);
    });
  });

  // Create and start the MCP server
  try {
    server = await createServer(config, isStdioMode);

    // Log startup info (only in non-stdio mode since we use stderr in stdio mode)
    const sessionState = server.getSessionState();
    if (!isStdioMode) {
      console.log('workspace-qdrant-mcp server started');
      console.log(`Session ID: ${sessionState.sessionId}`);
      console.log(`Project: ${sessionState.projectPath ?? 'none'}`);
      console.log(`Project ID: ${sessionState.projectId ?? 'none'}`);
      console.log(`Daemon connected: ${sessionState.daemonConnected}`);
    }

    // Server is now running - MCP SDK handles the event loop
  } catch (error) {
    console.error('Failed to start MCP server:', error);
    process.exit(1);
  }
}

main().catch((error: unknown) => {
  console.error('Fatal error:', error);
  process.exit(1);
});

// Export for testing
export { WorkspaceQdrantMcpServer, createServer };
