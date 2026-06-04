/**
 * Project detection and tenant_id fetching
 *
 * Fetches project_id from daemon's SQLite state database (NOT calculated locally).
 * This prevents drift between MCP and daemon's project_id calculations.
 */
import { SqliteStateManager } from '../clients/sqlite-state-manager.js';
export interface ProjectDetectorConfig {
    stateManager?: SqliteStateManager;
    maxSearchDepth?: number;
    cacheTtlMs?: number;
}
export interface ProjectInfo {
    projectId: string;
    projectPath: string;
    isActive: boolean;
    gitRemote?: string | undefined;
}
/**
 * Project detector for MCP server
 *
 * Detects the current project and fetches its project_id from the daemon's
 * SQLite database. Uses caching to avoid repeated database queries.
 *
 * Usage:
 * ```typescript
 * const detector = new ProjectDetector({ stateManager });
 *
 * // Find project root from current directory
 * const root = detector.findProjectRoot(process.cwd());
 *
 * // Get project_id for the root (fetched from daemon's database)
 * const info = await detector.getProjectInfo(root);
 * if (info) {
 *   console.log(`Project ID: ${info.projectId}`);
 * }
 * ```
 */
export declare class ProjectDetector {
    private readonly stateManager;
    private readonly maxSearchDepth;
    private readonly cacheTtlMs;
    private readonly cache;
    constructor(config?: ProjectDetectorConfig);
    /**
     * Find project root by walking up directory tree
     *
     * Looks for project markers (.git, package.json, etc.) to identify
     * the project root directory.
     *
     * @param startPath Starting path to search from
     * @returns Project root path or null if not found
     */
    findProjectRoot(startPath: string): string | null;
    /**
     * Check if a directory contains any project markers
     */
    private hasProjectMarker;
    /**
     * Get project info for a project path
     *
     * Fetches project_id from daemon's watch_folders table (collection='projects').
     * Uses caching to avoid repeated database queries.
     * Retries with exponential backoff if project not found (daemon may still be registering).
     *
     * @param projectPath Absolute path to project root
     * @param waitForRegistration If true, retry if project not found
     * @returns ProjectInfo or null if not found/registered
     */
    getProjectInfo(projectPath: string, waitForRegistration?: boolean): Promise<ProjectInfo | null>;
    /**
     * Get current project info based on working directory
     *
     * Combines findProjectRoot and getProjectInfo.
     *
     * @param cwd Current working directory
     * @param waitForRegistration If true, retry if project not found
     * @returns ProjectInfo or null
     */
    getCurrentProject(cwd?: string, waitForRegistration?: boolean): Promise<ProjectInfo | null>;
    /**
     * Get current project_id only
     *
     * Convenience method that returns just the project_id.
     */
    getCurrentProjectId(cwd?: string, waitForRegistration?: boolean): Promise<string | null>;
    /**
     * Clear the cache
     */
    clearCache(): void;
    /**
     * Clear cache entry for a specific path
     */
    clearCacheForPath(projectPath: string): void;
    /**
     * Fetch project info from database
     */
    private fetchProjectInfo;
    /**
     * Fetch project info with retry for registration
     */
    private fetchWithRetry;
    /**
     * Sleep for a specified duration
     */
    private sleep;
}
export { isGitRepository, getGitRemoteUrl, findGitRoot } from './git-utils.js';
//# sourceMappingURL=project-detector.d.ts.map