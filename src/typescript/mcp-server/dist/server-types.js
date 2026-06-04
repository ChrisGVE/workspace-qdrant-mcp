/**
 * Shared types and constants for the MCP server
 */
import { BUILD_NUMBER } from './build-info.js';
// Heartbeat interval: 30 seconds (in milliseconds)
// Must be well under the daemon's orphan-cleanup timeout (120s) to prevent
// false deactivation. 30s gives 4 missed heartbeats before deactivation.
export const HEARTBEAT_INTERVAL_MS = 30 * 1000;
// Server name and version for MCP protocol
export const SERVER_NAME = 'workspace-qdrant-mcp';
export const SERVER_VERSION = `0.1.0-beta1 (${BUILD_NUMBER})`;
/** Default HTTP listener configuration for `mode: 'http'`. */
export const DEFAULT_HTTP_HOST = '127.0.0.1';
export const DEFAULT_HTTP_PORT = 6335;
export const DEFAULT_HTTP_PATH = '/mcp';
//# sourceMappingURL=server-types.js.map