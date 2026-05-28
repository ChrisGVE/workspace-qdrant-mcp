/**
 * Project detection and tenant_id fetching
 *
 * Fetches project_id from daemon's SQLite state database (NOT calculated locally).
 * This prevents drift between MCP and daemon's project_id calculations.
 */

import { readdirSync } from 'node:fs';
import { dirname, posix, resolve } from 'node:path';

import { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import { canonicalizeHostPath } from '../clients/project-queries.js';
import { getEffectiveCwd } from './request-context.js';

// Project signature files/directories
const PROJECT_MARKERS = [
  '.git', // Git repository
  'package.json', // Node.js project
  'Cargo.toml', // Rust project
  'pyproject.toml', // Python (modern)
  'setup.py', // Python (legacy)
  'go.mod', // Go project
  'pom.xml', // Java Maven
  'build.gradle', // Java Gradle
  'Makefile', // Make-based project
  '.workspace-qdrant', // Our own marker
];

// Maximum depth to search upward for project root
const MAX_SEARCH_DEPTH = 20;

// Cache TTL in milliseconds (5 minutes)
const CACHE_TTL_MS = 5 * 60 * 1000;

// Retry configuration for first search
const INITIAL_RETRY_DELAY_MS = 100;
const MAX_RETRIES = 5;
const MAX_RETRY_DELAY_MS = 2000;

interface CacheEntry {
  projectId: string | null;
  timestamp: number;
}

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

export interface GetProjectInfoOptions {
  /**
   * When path detection finds no project AND exactly one project is
   * registered, assume that project instead of returning null. Intended for
   * CWD-based MCP tool detection — where the host path may not be reconcilable
   * with the daemon-stored path (e.g. the host-cwd header was absent and the
   * container WORKDIR leaked through) — not for strict lookups.
   */
  fallbackToSoleProject?: boolean;
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
export class ProjectDetector {
  private readonly stateManager: SqliteStateManager;
  private readonly maxSearchDepth: number;
  private readonly cacheTtlMs: number;

  // Cache: projectPath -> { projectId, timestamp }
  private readonly cache = new Map<string, CacheEntry>();

  constructor(config: ProjectDetectorConfig = {}) {
    this.stateManager = config.stateManager ?? new SqliteStateManager();
    this.maxSearchDepth = config.maxSearchDepth ?? MAX_SEARCH_DEPTH;
    this.cacheTtlMs = config.cacheTtlMs ?? CACHE_TTL_MS;
  }

  /**
   * Find project root by walking up directory tree
   *
   * Looks for project markers (.git, package.json, etc.) to identify
   * the project root directory.
   *
   * @param startPath Starting path to search from
   * @returns Project root path or null if not found
   */
  findProjectRoot(startPath: string): string | null {
    let currentPath = resolve(startPath);
    let depth = 0;

    while (depth < this.maxSearchDepth) {
      // Check if current directory has any project markers
      if (this.hasProjectMarker(currentPath)) {
        return currentPath;
      }

      // Move up one directory
      const parentPath = dirname(currentPath);

      // Stop if we've reached the filesystem root
      if (parentPath === currentPath) {
        break;
      }

      currentPath = parentPath;
      depth++;
    }

    return null;
  }

  /**
   * Check if a directory contains any project markers
   */
  private hasProjectMarker(dirPath: string): boolean {
    try {
      const entries = readdirSync(dirPath);
      return PROJECT_MARKERS.some((marker) => entries.includes(marker));
    } catch {
      // Directory doesn't exist or not readable
      return false;
    }
  }

  /**
   * Get project info for a project path
   *
   * Fetches project_id from daemon's watch_folders table (collection='projects').
   * Uses caching to avoid repeated database queries.
   * Retries with exponential backoff if project not found (daemon may still be registering).
   *
   * @param projectPath Absolute path to project root
   * @param waitForRegistration If true, retry if project not found
   * @param options Optional resolution behavior (see {@link GetProjectInfoOptions})
   * @returns ProjectInfo or null if not found/registered
   */
  async getProjectInfo(
    projectPath: string,
    waitForRegistration = false,
    options: GetProjectInfoOptions = {}
  ): Promise<ProjectInfo | null> {
    const info = await this.detectProjectByPath(projectPath, waitForRegistration);
    if (info) {
      return info;
    }
    if (options.fallbackToSoleProject) {
      return this.soleRegisteredProject();
    }
    return null;
  }

  /**
   * Normalize a host working directory for cross-namespace comparison and
   * cache keying. Uses `posix.resolve` (NOT the platform `resolve`):
   * canonicalizeHostPath yields a POSIX-form path (e.g. `C:\…` → `/c/…`), and
   * on Windows the win32 `resolve` would re-mangle that back into `C:\c\…`.
   * POSIX semantics keep the result identical on every host, so the
   * longest-prefix match in getProjectByPath can bridge a client cwd to the
   * daemon-stored `/run/desktop/mnt/host/c/…` mount path.
   */
  private normalizeHostCwd(projectPath: string): string {
    return posix.resolve(canonicalizeHostPath(projectPath));
  }

  /**
   * Path-based detection: cache lookup, then a longest-prefix match against
   * the daemon's registered project paths. Returns null on miss.
   */
  private async detectProjectByPath(
    projectPath: string,
    waitForRegistration: boolean
  ): Promise<ProjectInfo | null> {
    // Bridge the host/container path namespace before any matching. An HTTP
    // client on Windows sends a cwd like `C:\…`; see normalizeHostCwd for why
    // this must canonicalize and use posix.resolve rather than the platform one.
    const normalizedPath = this.normalizeHostCwd(projectPath);

    // Check cache first
    const cached = this.cache.get(normalizedPath);
    if (cached && Date.now() - cached.timestamp < this.cacheTtlMs) {
      if (cached.projectId === null) {
        return null;
      }
      // Fetch full info from database (cache only stores projectId)
      return this.fetchProjectInfo(normalizedPath);
    }

    // Initialize state manager if needed
    if (!this.stateManager.isConnected()) {
      const initResult = this.stateManager.initialize();
      if (initResult.status === 'degraded') {
        // Database not available, can't fetch project_id
        return null;
      }
    }

    // Fetch from database with optional retry
    if (waitForRegistration) {
      return this.fetchWithRetry(normalizedPath);
    }

    return this.fetchProjectInfo(normalizedPath);
  }

  /**
   * Fallback for CWD-based detection: when no project maps to the host path
   * and exactly one project is registered, assume it. Returns null when zero
   * or 2+ projects are registered, or when the database is unavailable.
   */
  private soleRegisteredProject(): ProjectInfo | null {
    if (!this.stateManager.isConnected()) {
      const initResult = this.stateManager.initialize();
      if (initResult.status === 'degraded') {
        return null;
      }
    }

    const all = this.stateManager.listAllProjects();
    if (all.status !== 'ok' || all.data.length !== 1) {
      return null;
    }

    const [project] = all.data;
    if (!project) {
      return null;
    }
    return {
      projectId: project.project_id,
      projectPath: project.project_path,
      isActive: project.is_active,
      gitRemote: project.git_remote_url,
    };
  }

  /**
   * Get current project info based on working directory
   *
   * Combines findProjectRoot and getProjectInfo.
   *
   * @param cwd Current working directory
   * @param waitForRegistration If true, retry if project not found
   * @returns ProjectInfo or null
   */
  async getCurrentProject(
    cwd: string = getEffectiveCwd(),
    waitForRegistration = false
  ): Promise<ProjectInfo | null> {
    // Pass cwd directly — the database query uses longest-prefix matching
    // to resolve subdirectories to their registered project root.
    return this.getProjectInfo(cwd, waitForRegistration);
  }

  /**
   * Get current project_id only
   *
   * Convenience method that returns just the project_id.
   */
  async getCurrentProjectId(
    cwd: string = getEffectiveCwd(),
    waitForRegistration = false
  ): Promise<string | null> {
    const info = await this.getCurrentProject(cwd, waitForRegistration);
    return info?.projectId ?? null;
  }

  /**
   * Clear the cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Clear cache entry for a specific path
   */
  clearCacheForPath(projectPath: string): void {
    this.cache.delete(this.normalizeHostCwd(projectPath));
  }

  /**
   * Fetch project info from database
   */
  private fetchProjectInfo(projectPath: string): ProjectInfo | null {
    const result = this.stateManager.getProjectByPath(projectPath);

    if (result.status === 'degraded' || !result.data) {
      // Cache the negative result
      this.cache.set(projectPath, {
        projectId: null,
        timestamp: Date.now(),
      });
      return null;
    }

    // Cache the result
    this.cache.set(projectPath, {
      projectId: result.data.project_id,
      timestamp: Date.now(),
    });

    return {
      projectId: result.data.project_id,
      projectPath: result.data.project_path,
      isActive: result.data.is_active,
      gitRemote: result.data.git_remote_url,
    };
  }

  /**
   * Fetch project info with retry for registration
   */
  private async fetchWithRetry(projectPath: string): Promise<ProjectInfo | null> {
    let delay = INITIAL_RETRY_DELAY_MS;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      const info = this.fetchProjectInfo(projectPath);

      if (info) {
        return info;
      }

      // Wait before retry (daemon may still be registering)
      if (attempt < MAX_RETRIES - 1) {
        await this.sleep(delay);
        delay = Math.min(delay * 2, MAX_RETRY_DELAY_MS);
      }
    }

    return null;
  }

  /**
   * Sleep for a specified duration
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Re-export git utilities for backward compatibility
export { isGitRepository, getGitRemoteUrl, findGitRoot } from './git-utils.js';
