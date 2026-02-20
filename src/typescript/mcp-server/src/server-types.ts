/**
 * Shared types and constants for the MCP server
 */

import type { ServerConfig } from './types/index.js';
import { BUILD_NUMBER } from './build-info.js';

// Heartbeat interval: 1 hour (in milliseconds)
export const HEARTBEAT_INTERVAL_MS = 1 * 60 * 60 * 1000;

// Server name and version for MCP protocol
export const SERVER_NAME = 'workspace-qdrant-mcp';
export const SERVER_VERSION = `0.1.0-beta1 (${BUILD_NUMBER})`;

export interface SessionState {
  sessionId: string;
  projectId: string | null;
  projectPath: string | null;
  heartbeatInterval: ReturnType<typeof setInterval> | null;
  daemonConnected: boolean;
}

export interface ServerOptions {
  config: ServerConfig;
  stdio?: boolean;
}
