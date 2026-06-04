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
/**
 * Config directory: user-editable settings files.
 *
 * Precedence: WQM_CONFIG_DIR > XDG_CONFIG_HOME > ~/.config
 */
export declare function getConfigDirectory(): string;
/**
 * Data directory: databases and runtime state the daemon owns.
 *
 * Precedence: WQM_DATA_DIR > XDG_DATA_HOME > ~/.local/share
 */
export declare function getDataDirectory(): string;
/**
 * Cache directory: re-downloadable artifacts (grammars, models).
 *
 * Precedence: WQM_CACHE_DIR > XDG_CACHE_HOME > ~/.cache
 */
export declare function getCacheDirectory(): string;
/**
 * Canonical log directory.
 *
 * Precedence: WQM_LOG_DIR > platform-specific default.
 */
export declare function getLogDirectory(): string;
/**
 * Canonical database path.
 *
 * Precedence: WQM_DATABASE_PATH > <data_dir>/state.db
 */
export declare function getDatabasePath(): string;
/**
 * Ensures the log directory exists, creating it if necessary.
 */
export declare function ensureLogDirectory(): boolean;
/**
 * Full path to the MCP server log file.
 */
export declare function getMcpServerLogPath(): string;
//# sourceMappingURL=paths.d.ts.map