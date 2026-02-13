/**
 * Project detection and tenant_id fetching
 *
 * Fetches project_id from daemon's SQLite state database (NOT calculated locally).
 * This prevents drift between MCP and daemon's project_id calculations.
 */

import { existsSync, statSync, readdirSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';

import { SqliteStateManager } from '../clients/sqlite-state-manager.js';

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
   * @returns ProjectInfo or null if not found/registered
   */
  async getProjectInfo(
    projectPath: string,
    waitForRegistration = false
  ): Promise<ProjectInfo | null> {
    const normalizedPath = resolve(projectPath);

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
   * Get current project info based on working directory
   *
   * Combines findProjectRoot and getProjectInfo.
   *
   * @param cwd Current working directory
   * @param waitForRegistration If true, retry if project not found
   * @returns ProjectInfo or null
   */
  async getCurrentProject(
    cwd: string = process.cwd(),
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
    cwd: string = process.cwd(),
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
    this.cache.delete(resolve(projectPath));
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

/**
 * Check if a path is a git repository
 */
export function isGitRepository(path: string): boolean {
  const gitDir = join(path, '.git');
  try {
    return existsSync(gitDir) && statSync(gitDir).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Get git remote URL for a repository
 *
 * Note: This is for informational purposes only.
 * The daemon computes the authoritative project_id.
 */
export function getGitRemoteUrl(repoPath: string): string | null {
  const gitConfigPath = join(repoPath, '.git', 'config');

  try {
    const { readFileSync } = require('node:fs');
    const config = readFileSync(gitConfigPath, 'utf-8');

    // Parse remote "origin" URL
    const remoteMatch = config.match(/\[remote "origin"\][^\[]*url = (.+)/m);
    if (remoteMatch?.[1]) {
      return remoteMatch[1].trim();
    }
  } catch {
    // File doesn't exist or not readable
  }

  return null;
}
