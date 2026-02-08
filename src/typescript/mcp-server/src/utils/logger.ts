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
import type { Logger, LoggerOptions } from 'pino';
import { ensureLogDirectory, getMcpServerLogPath } from './paths.js';

// Session ID for correlation across log entries
let currentSessionId: string | undefined;

/**
 * Set the session ID for log correlation
 */
export function setSessionId(sessionId: string): void {
  currentSessionId = sessionId;
}

/**
 * Get the current session ID
 */
export function getSessionId(): string | undefined {
  return currentSessionId;
}

/**
 * Create the pino logger instance
 *
 * In production, logs to file only (no stderr) to avoid MCP protocol conflicts.
 * If log directory creation fails, falls back to /dev/null.
 */
function createLogger(): Logger {
  const level = process.env['WQM_MCP_LOG_LEVEL'] ?? process.env['WQM_LOG_LEVEL'] ?? 'info';

  // Ensure log directory exists
  const dirCreated = ensureLogDirectory();

  // Configure pino options
  const options: LoggerOptions = {
    name: 'mcp-server',
    level,
    // Add base fields to every log entry
    base: {
      component: 'mcp-server',
    },
    // Use ISO timestamps for JSON logs
    timestamp: pino.stdTimeFunctions.isoTime,
  };

  // File-only logging to avoid MCP protocol conflicts
  // Important: MCP server MUST NOT write to stdout/stderr as it would corrupt the protocol
  if (dirCreated) {
    const logPath = getMcpServerLogPath();

    // Rotation settings matching daemon defaults:
    // - 50MB max file size
    // - 5 rotated files kept (+ 1 active)
    const rotationSizeMb = parseInt(process.env['WQM_LOG_ROTATION_SIZE_MB'] ?? '50', 10);
    const rotationCount = parseInt(process.env['WQM_LOG_ROTATION_COUNT'] ?? '5', 10);

    // Use pino-roll transport for size-based file rotation
    return pino(
      options,
      pino.transport({
        target: 'pino-roll',
        options: {
          file: logPath,
          size: `${rotationSizeMb}m`,
          limit: { count: rotationCount },
          mkdir: true,
        },
      })
    );
  } else {
    // Fallback: silent logging if we can't create the log directory
    // This prevents any output that could interfere with the MCP protocol
    return pino({ ...options, level: 'silent' });
  }
}

// Create singleton logger instance
const logger = createLogger();

/**
 * Create a child logger with session context
 */
export function createSessionLogger(sessionId?: string): Logger {
  const sid = sessionId ?? currentSessionId;
  if (sid) {
    return logger.child({ session_id: sid });
  }
  return logger;
}

/**
 * Log an info message with session context
 */
export function logInfo(msg: string, context?: Record<string, unknown>): void {
  const log = createSessionLogger();
  if (context) {
    log.info(context, msg);
  } else {
    log.info(msg);
  }
}

/**
 * Log a debug message with session context
 */
export function logDebug(msg: string, context?: Record<string, unknown>): void {
  const log = createSessionLogger();
  if (context) {
    log.debug(context, msg);
  } else {
    log.debug(msg);
  }
}

/**
 * Log a warning message with session context
 */
export function logWarn(msg: string, context?: Record<string, unknown>): void {
  const log = createSessionLogger();
  if (context) {
    log.warn(context, msg);
  } else {
    log.warn(msg);
  }
}

/**
 * Log an error message with session context
 */
export function logError(msg: string, error?: unknown, context?: Record<string, unknown>): void {
  const log = createSessionLogger();
  const errorContext: Record<string, unknown> = context ?? {};

  if (error instanceof Error) {
    errorContext.error = error.message;
    errorContext.stack = error.stack;
  } else if (error !== undefined) {
    errorContext.error = String(error);
  }

  log.error(errorContext, msg);
}

/**
 * Log a tool invocation
 */
export function logToolCall(
  tool: string,
  durationMs?: number,
  success?: boolean,
  context?: Record<string, unknown>
): void {
  const log = createSessionLogger();
  log.info(
    {
      tool,
      duration_ms: durationMs,
      success,
      ...context,
    },
    'Tool called'
  );
}

/**
 * Log session lifecycle event
 */
export function logSessionEvent(
  event: 'start' | 'end' | 'heartbeat' | 'register' | 'deprioritize',
  context?: Record<string, unknown>
): void {
  const log = createSessionLogger();
  log.info({ event, ...context }, `Session ${event}`);
}

/**
 * Log daemon connection status
 */
export function logDaemonStatus(
  connected: boolean,
  context?: Record<string, unknown>
): void {
  const log = createSessionLogger();
  if (connected) {
    log.info(context ?? {}, 'Daemon connected');
  } else {
    log.warn(context ?? {}, 'Daemon disconnected');
  }
}

// Export the raw logger for advanced use cases
export { logger };
