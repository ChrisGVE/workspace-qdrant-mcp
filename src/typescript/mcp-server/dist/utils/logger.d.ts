/**
 * Structured logging for the MCP Server using pino
 *
 * Logs are written to a file (mcp-server.jsonl) to avoid protocol conflicts
 * when running in stdio mode. The file path follows OS conventions:
 * - Linux: $XDG_STATE_HOME/workspace-qdrant/logs/mcp-server.jsonl
 * - macOS: ~/Library/Logs/workspace-qdrant/mcp-server.jsonl
 * - Windows: %LOCALAPPDATA%\workspace-qdrant\logs\mcp-server.jsonl
 */
import pino from 'pino';
import type { Logger } from 'pino';
/**
 * Set the session ID for log correlation
 */
export declare function setSessionId(sessionId: string): void;
/**
 * Get the current session ID
 */
export declare function getSessionId(): string | undefined;
declare const logger: pino.Logger;
/**
 * Create a child logger with session context
 */
export declare function createSessionLogger(sessionId?: string): Logger;
/**
 * Log an info message with session context
 */
export declare function logInfo(msg: string, context?: Record<string, unknown>): void;
/**
 * Log a debug message with session context
 */
export declare function logDebug(msg: string, context?: Record<string, unknown>): void;
/**
 * Log a warning message with session context
 */
export declare function logWarn(msg: string, context?: Record<string, unknown>): void;
/**
 * Log an error message with session context
 */
export declare function logError(msg: string, error?: unknown, context?: Record<string, unknown>): void;
/**
 * Log a tool invocation
 */
export declare function logToolCall(tool: string, durationMs?: number, success?: boolean, context?: Record<string, unknown>): void;
/**
 * Log session lifecycle event
 */
export declare function logSessionEvent(event: 'start' | 'end' | 'heartbeat' | 'register' | 'deprioritize', context?: Record<string, unknown>): void;
/**
 * Log daemon connection status
 */
export declare function logDaemonStatus(connected: boolean, context?: Record<string, unknown>): void;
export { logger };
//# sourceMappingURL=logger.d.ts.map