/**
 * Configuration loading and management
 * Loads from YAML config file with environment variable overrides
 */

import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { parse as parseYaml } from 'yaml';
import { type ServerConfig, DEFAULT_CONFIG } from './types/config.js';

/**
 * Unified configuration search paths (in priority order)
 *
 * Search order:
 * 1. WQM_CONFIG_PATH environment variable (if set)
 * 2. Project-local: .workspace-qdrant.yaml (CWD, daemon/MCP only)
 * 3. User home: ~/.workspace-qdrant/config.yaml
 * 4. XDG/platform config dir: $XDG_CONFIG_HOME/workspace-qdrant/config.yaml (Linux),
 *    %LOCALAPPDATA%\workspace-qdrant\config.yaml (Windows)
 * 5. macOS: ~/Library/Application Support/workspace-qdrant/config.yaml
 */
function getConfigSearchPaths(): string[] {
  const home = homedir();
  const paths: string[] = [];

  // 1. Explicit path via environment variable (highest priority)
  const explicitPath = process.env['WQM_CONFIG_PATH'];
  if (explicitPath) {
    paths.push(explicitPath);
  }

  // 2. Project-local configs (current working directory)
  const cwd = process.cwd();
  paths.push(join(cwd, '.workspace-qdrant.yaml'));
  paths.push(join(cwd, '.workspace-qdrant.yml'));

  // 3. User home: ~/.workspace-qdrant/config.yaml
  paths.push(join(home, '.workspace-qdrant', 'config.yaml'));
  paths.push(join(home, '.workspace-qdrant', 'config.yml'));

  // 4. Platform-specific config directory
  const currentPlatform = process.platform;
  if (currentPlatform === 'linux') {
    // XDG Base Directory Specification
    const xdgConfigHome = process.env['XDG_CONFIG_HOME'] ?? join(home, '.config');
    paths.push(join(xdgConfigHome, 'workspace-qdrant', 'config.yaml'));
    paths.push(join(xdgConfigHome, 'workspace-qdrant', 'config.yml'));
  } else if (currentPlatform === 'win32') {
    // Windows: %LOCALAPPDATA%
    const localAppData = process.env['LOCALAPPDATA'] ?? join(home, 'AppData', 'Local');
    paths.push(join(localAppData, 'workspace-qdrant', 'config.yaml'));
    paths.push(join(localAppData, 'workspace-qdrant', 'config.yml'));
  } else if (currentPlatform === 'darwin') {
    // macOS: ~/.config fallback then Application Support
    paths.push(join(home, '.config', 'workspace-qdrant', 'config.yaml'));
    paths.push(join(home, '.config', 'workspace-qdrant', 'config.yml'));
  }

  // 5. macOS Application Support
  if (currentPlatform === 'darwin') {
    paths.push(join(home, 'Library', 'Application Support', 'workspace-qdrant', 'config.yaml'));
  }

  return paths;
}

function expandPath(path: string): string {
  if (path.startsWith('~')) {
    return join(homedir(), path.slice(1));
  }
  return path;
}

function findConfigFile(): string | null {
  for (const path of getConfigSearchPaths()) {
    if (existsSync(path)) {
      return path;
    }
  }
  return null;
}

function loadConfigFromFile(filePath: string): Partial<ServerConfig> {
  try {
    const content = readFileSync(filePath, 'utf-8');
    return parseYaml(content) as Partial<ServerConfig>;
  } catch {
    console.warn(`Failed to load config from ${filePath}`);
    return {};
  }
}

function mergeConfigs(base: ServerConfig, override: Partial<ServerConfig>): ServerConfig {
  return {
    database: { ...base.database, ...override.database },
    qdrant: { ...base.qdrant, ...override.qdrant },
    daemon: { ...base.daemon, ...override.daemon },
    watching: { ...base.watching, ...override.watching },
    collections: { ...base.collections, ...override.collections },
    environment: { ...base.environment, ...override.environment },
  };
}

function applyEnvironmentOverrides(config: ServerConfig): ServerConfig {
  const result = { ...config };

  // Qdrant overrides
  if (process.env['QDRANT_URL']) {
    result.qdrant = { ...result.qdrant, url: process.env['QDRANT_URL'] };
  }
  if (process.env['QDRANT_API_KEY']) {
    result.qdrant = { ...result.qdrant, apiKey: process.env['QDRANT_API_KEY'] };
  }

  // Database path override
  if (process.env['WQM_DATABASE_PATH']) {
    result.database = { ...result.database, path: process.env['WQM_DATABASE_PATH'] };
  }

  // Daemon port override
  if (process.env['WQM_DAEMON_PORT']) {
    const port = parseInt(process.env['WQM_DAEMON_PORT'], 10);
    if (!isNaN(port)) {
      result.daemon = { ...result.daemon, grpcPort: port };
    }
  }

  return result;
}

export function loadConfig(): ServerConfig {
  let config = { ...DEFAULT_CONFIG };

  // Load from file if exists
  const configPath = findConfigFile();
  if (configPath) {
    const fileConfig = loadConfigFromFile(configPath);
    config = mergeConfigs(config, fileConfig);
  }

  // Apply environment overrides
  config = applyEnvironmentOverrides(config);

  // Expand paths
  config.database.path = expandPath(config.database.path);

  return config;
}

export function getDatabasePath(config?: ServerConfig): string {
  const cfg = config ?? loadConfig();
  return cfg.database.path;
}

export function getQdrantUrl(config?: ServerConfig): string {
  const cfg = config ?? loadConfig();
  return cfg.qdrant.url;
}

export function getDaemonPort(config?: ServerConfig): number {
  const cfg = config ?? loadConfig();
  return cfg.daemon.grpcPort;
}
