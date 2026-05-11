/**
 * Configuration loading and management
 * Loads from YAML config file with environment variable overrides
 */

import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { parse as parseYaml } from 'yaml';
import { type ServerConfig, DEFAULT_CONFIG } from './types/config.js';
import { getConfigDirectory } from './utils/paths.js';

/**
 * Config search paths (priority order). First existing file wins.
 *
 * 1. WQM_CONFIG_PATH environment variable
 * 2. XDG config dir: ~/.config/workspace-qdrant/config.yaml
 */
function getConfigSearchPaths(): string[] {
  const paths: string[] = [];

  const explicitPath = process.env['WQM_CONFIG_PATH'];
  if (explicitPath) {
    paths.push(explicitPath);
  }

  const configDir = getConfigDirectory();
  paths.push(join(configDir, 'config.yaml'));
  paths.push(join(configDir, 'config.yml'));

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
  const merged: ServerConfig = {
    database: { ...base.database, ...override.database },
    qdrant: { ...base.qdrant, ...override.qdrant },
    daemon: { ...base.daemon, ...override.daemon },
    watching: { ...base.watching, ...override.watching },
    collections: { ...base.collections, ...override.collections },
    environment: { ...base.environment, ...override.environment },
  };

  // Merge rules block: override values win, base provides defaults.
  if (base.rules !== undefined || override.rules !== undefined) {
    merged.rules = {
      ...(base.rules ?? {}),
      ...(override.rules ?? {}),
      limits: {
        ...(base.rules?.limits ?? {
          maxLabelLength: 0,
          maxTitleLength: 0,
          maxTagLength: 0,
          maxTagsPerRule: 0,
        }),
        ...(override.rules?.limits ?? {}),
      },
    };
  }

  return merged;
}

/**
 * Parse a gRPC endpoint string into host and port components.
 *
 * Accepted formats:
 *   - "http://host:port"   (scheme stripped, port parsed)
 *   - "host:port"          (bare host:port)
 *   - "host"               (host only, port defaults to 50051)
 */
export function parseGrpcEndpoint(endpoint: string): { host: string; port: number } {
  const DEFAULT_PORT = 50051;
  // Strip http:// or https:// scheme if present.
  const withoutScheme = endpoint.replace(/^https?:\/\//, '');
  const colonIndex = withoutScheme.lastIndexOf(':');
  if (colonIndex === -1) {
    return { host: withoutScheme, port: DEFAULT_PORT };
  }
  const host = withoutScheme.slice(0, colonIndex);
  const portStr = withoutScheme.slice(colonIndex + 1);
  const port = parseInt(portStr, 10);
  if (isNaN(port) || port <= 0 || port > 65535) {
    return { host, port: DEFAULT_PORT };
  }
  return { host, port };
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

  // Daemon endpoint override: WQM_DAEMON_ENDPOINT preferred, MEMEXD_GRPC_URL as alias.
  const endpointEnv = process.env['WQM_DAEMON_ENDPOINT'] ?? process.env['MEMEXD_GRPC_URL'];
  if (endpointEnv) {
    const { host, port } = parseGrpcEndpoint(endpointEnv);
    result.daemon = { ...result.daemon, grpcHost: host, grpcPort: port };
  }

  // Daemon port-only override (legacy; endpoint env takes precedence when both set).
  if (!endpointEnv && process.env['WQM_DAEMON_PORT']) {
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

export function getDaemonHost(config?: ServerConfig): string {
  const cfg = config ?? loadConfig();
  return cfg.daemon.grpcHost;
}
