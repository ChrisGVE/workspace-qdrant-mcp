/**
 * Configuration types for workspace-qdrant-mcp server
 */

export interface DatabaseConfig {
  path: string;
}

export interface QdrantConfig {
  url: string;
  apiKey?: string;
  timeout: number;
}

export interface DaemonConfig {
  grpcPort: number;
  queuePollIntervalMs: number;
  queueBatchSize: number;
}

export interface WatchingConfig {
  patterns: string[];
  ignorePatterns: string[];
}

export interface CollectionsConfig {
  memoryCollectionName: string;
}

export interface MemoryLimitsConfig {
  maxLabelLength: number;
  maxTitleLength: number;
  maxTagLength: number;
  maxTagsPerRule: number;
}

export interface MemoryConfig {
  limits: MemoryLimitsConfig;
}

export interface EnvironmentConfig {
  userPath?: string;
}

export interface ServerConfig {
  database: DatabaseConfig;
  qdrant: QdrantConfig;
  daemon: DaemonConfig;
  watching: WatchingConfig;
  collections: CollectionsConfig;
  environment: EnvironmentConfig;
  memory?: MemoryConfig;
}

// DEFAULT_CONFIG is generated from assets/default_configuration.yaml
// Re-generate with: npx tsx scripts/generate-config-defaults.ts
export { DEFAULT_CONFIG } from './generated-defaults.js';
