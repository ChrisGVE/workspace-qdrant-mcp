/**
 * XDG-compliant path utilities for workspace-qdrant-mcp.
 *
 * Single source of truth — never hardcode directory names elsewhere.
 *
 * Layout (macOS, no env overrides):
 *   Config: ~/.config/workspace-qdrant/
 *   Data:   ~/.local/share/workspace-qdrant/
 *   Cache:  ~/.cache/workspace-qdrant/
 *   Logs:   ~/Library/Logs/workspace-qdrant/
 */
import { join } from 'node:path';
import { platform, homedir, tmpdir } from 'node:os';
import { mkdirSync, existsSync } from 'node:fs';
const DIR_NAME = 'workspace-qdrant';
/**
 * Config directory: user-editable settings files.
 *
 * Precedence: WQM_CONFIG_DIR > XDG_CONFIG_HOME > ~/.config
 */
export function getConfigDirectory() {
    if (process.env['WQM_CONFIG_DIR']) {
        return process.env['WQM_CONFIG_DIR'];
    }
    const home = homedir() || tmpdir();
    const xdgConfigHome = process.env['XDG_CONFIG_HOME'] ?? join(home, '.config');
    return join(xdgConfigHome, DIR_NAME);
}
/**
 * Data directory: databases and runtime state the daemon owns.
 *
 * Precedence: WQM_DATA_DIR > XDG_DATA_HOME > ~/.local/share
 */
export function getDataDirectory() {
    if (process.env['WQM_DATA_DIR']) {
        return process.env['WQM_DATA_DIR'];
    }
    const home = homedir() || tmpdir();
    const xdgDataHome = process.env['XDG_DATA_HOME'] ?? join(home, '.local', 'share');
    return join(xdgDataHome, DIR_NAME);
}
/**
 * Cache directory: re-downloadable artifacts (grammars, models).
 *
 * Precedence: WQM_CACHE_DIR > XDG_CACHE_HOME > ~/.cache
 */
export function getCacheDirectory() {
    if (process.env['WQM_CACHE_DIR']) {
        return process.env['WQM_CACHE_DIR'];
    }
    const home = homedir() || tmpdir();
    const xdgCacheHome = process.env['XDG_CACHE_HOME'] ?? join(home, '.cache');
    return join(xdgCacheHome, DIR_NAME);
}
/**
 * Canonical log directory.
 *
 * Precedence: WQM_LOG_DIR > platform-specific default.
 */
export function getLogDirectory() {
    if (process.env['WQM_LOG_DIR']) {
        return process.env['WQM_LOG_DIR'];
    }
    const home = homedir() || tmpdir();
    const currentPlatform = platform();
    switch (currentPlatform) {
        case 'linux': {
            const xdgStateHome = process.env['XDG_STATE_HOME'] ?? join(home, '.local', 'state');
            return join(xdgStateHome, DIR_NAME, 'logs');
        }
        case 'darwin':
            return join(home, 'Library', 'Logs', DIR_NAME);
        case 'win32': {
            const localAppData = process.env['LOCALAPPDATA'] ?? join(home, 'AppData', 'Local');
            return join(localAppData, DIR_NAME, 'logs');
        }
        default:
            return join(getDataDirectory(), 'logs');
    }
}
/**
 * Canonical database path.
 *
 * Precedence: WQM_DATABASE_PATH > <data_dir>/state.db
 */
export function getDatabasePath() {
    if (process.env['WQM_DATABASE_PATH']) {
        return process.env['WQM_DATABASE_PATH'];
    }
    return join(getDataDirectory(), 'state.db');
}
/**
 * Ensures the log directory exists, creating it if necessary.
 */
export function ensureLogDirectory() {
    const logDir = getLogDirectory();
    try {
        if (!existsSync(logDir)) {
            mkdirSync(logDir, { recursive: true });
        }
        return true;
    }
    catch {
        return false;
    }
}
/**
 * Full path to the MCP server log file.
 */
export function getMcpServerLogPath() {
    return join(getLogDirectory(), 'mcp-server.jsonl');
}
//# sourceMappingURL=paths.js.map