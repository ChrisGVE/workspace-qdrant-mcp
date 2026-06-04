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
import { createServer, WorkspaceQdrantMcpServer } from './server.js';
/**
 * Start the MCP server
 * Used by agent.ts when running in --mcp-only mode
 */
export declare function startServer(): Promise<void>;
export { WorkspaceQdrantMcpServer, createServer, startServer as main };
//# sourceMappingURL=index.d.ts.map