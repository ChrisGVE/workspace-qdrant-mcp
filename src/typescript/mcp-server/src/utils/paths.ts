/**
 * Platform-specific path utilities for workspace-qdrant-mcp
 */

import { join } from 'node:path';
import { platform, homedir, tmpdir } from 'node:os';
import { mkdirSync, existsSync } from 'node:fs';

/**
 * Returns the canonical OS-specific log directory for workspace-qdrant logs.
 *
 * Precedence:
 * 1. `WQM_LOG_DIR` environment variable (explicit override)
 * 2. Platform-specific default:
 *    - Linux: $XDG_STATE_HOME/workspace-qdrant/logs/ (default: ~/.local/state/workspace-qdrant/logs/)
 *    - macOS: ~/Library/Logs/workspace-qdrant/
 *    - Windows: %LOCALAPPDATA%\workspace-qdrant\logs\
 *
 * Falls back to temp directory if home cannot be determined.
 */
export function getLogDirectory(): string {
  // WQM_LOG_DIR takes highest precedence
  const customDir = process.env['WQM_LOG_DIR'];
  if (customDir) {
    return customDir;
  }

  const home = homedir() || tmpdir();
  const currentPlatform = platform();

  let logDir: string;

  switch (currentPlatform) {
    case 'linux': {
      // Follow XDG Base Directory Specification
      const xdgStateHome = process.env['XDG_STATE_HOME'] ?? join(home, '.local', 'state');
      logDir = join(xdgStateHome, 'workspace-qdrant', 'logs');
      break;
    }

    case 'darwin': {
      // macOS uses ~/Library/Logs for application logs
      logDir = join(home, 'Library', 'Logs', 'workspace-qdrant');
      break;
    }

    case 'win32': {
      // Windows uses %LOCALAPPDATA%
      const localAppData = process.env['LOCALAPPDATA'] ?? join(home, 'AppData', 'Local');
      logDir = join(localAppData, 'workspace-qdrant', 'logs');
      break;
    }

    default: {
      // Fallback for other platforms
      logDir = join(home, '.workspace-qdrant', 'logs');
      break;
    }
  }

  return logDir;
}

/**
 * Ensures the log directory exists, creating it if necessary.
 * Returns true if directory exists or was created, false on error.
 */
export function ensureLogDirectory(): boolean {
  const logDir = getLogDirectory();

  try {
    if (!existsSync(logDir)) {
      mkdirSync(logDir, { recursive: true });
    }
    return true;
  } catch {
    // Log directory creation failed - will fall back to stderr
    return false;
  }
}

/**
 * Returns the full path to the MCP server log file.
 */
export function getMcpServerLogPath(): string {
  return join(getLogDirectory(), 'mcp-server.jsonl');
}

/**
 * Returns the canonical config directory for workspace-qdrant.
 *
 * Platform-specific paths:
 * - Linux: $XDG_CONFIG_HOME/workspace-qdrant/ (default: ~/.config/workspace-qdrant/)
 * - Windows: %LOCALAPPDATA%\workspace-qdrant\ (default: AppData\Local\workspace-qdrant\)
 * - macOS/other: ~/.workspace-qdrant/
 */
export function getConfigDirectory(): string {
  const home = homedir() || tmpdir();
  const currentPlatform = platform();

  if (currentPlatform === 'linux') {
    const xdgConfigHome = process.env['XDG_CONFIG_HOME'] ?? join(home, '.config');
    return join(xdgConfigHome, 'workspace-qdrant');
  }

  if (currentPlatform === 'win32') {
    const localAppData = process.env['LOCALAPPDATA'] ?? join(home, 'AppData', 'Local');
    return join(localAppData, 'workspace-qdrant');
  }

  return join(home, '.workspace-qdrant');
}

/**
 * Returns the canonical state directory for workspace-qdrant (SQLite database, etc).
 * Note: On Linux, config and state are in different directories per XDG spec.
 */
export function getStateDirectory(): string {
  const home = homedir() || tmpdir();

  // State directory is always ~/.workspace-qdrant regardless of platform
  // This matches the Rust daemon's database path
  return join(home, '.workspace-qdrant');
}
