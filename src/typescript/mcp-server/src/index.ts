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
import type { ServerConfig } from './types/index.js';

// Detect stdio mode (MCP protocol requires clean stdout)
const isStdioMode = !process.env['WQM_CLI_MODE'] && !process.env['WQM_HTTP_MODE'];

// In stdio mode, suppress all console output to prevent protocol contamination
if (isStdioMode) {
  console.log = (): void => {};
  console.info = (): void => {};
  console.debug = (): void => {};
  // Keep console.error and console.warn for critical issues (sent to stderr)
}

async function main(): Promise<void> {
  // Load configuration
  const config: ServerConfig = loadConfig();

  // TODO: Initialize components (will be implemented in subsequent tasks)
  // - Task 479: gRPC client for daemon communication
  // - Task 480: SQLite state manager
  // - Task 481: Qdrant client with hybrid search
  // - Task 482: Session lifecycle management (SessionStart/SessionEnd hooks)
  // - Task 483-486: MCP tools (search, retrieve, memory, store)

  if (!isStdioMode) {
    console.error('workspace-qdrant-mcp server initialized');
    console.error(`Qdrant URL: ${config.qdrant.url}`);
    console.error(`Daemon port: ${config.daemon.grpcPort}`);
    console.error(`Database path: ${config.database.path}`);
  }

  // Placeholder for MCP server setup
  // The actual server will be created when implementing the tools
  console.error('MCP server starting... (skeleton implementation)');

  // Keep process alive for stdio mode
  if (isStdioMode) {
    // In production, MCP SDK handles the event loop
    // For now, just wait
    await new Promise<void>((resolve) => {
      process.on('SIGINT', () => resolve());
      process.on('SIGTERM', () => resolve());
    });
  }
}

main().catch((error: unknown) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
