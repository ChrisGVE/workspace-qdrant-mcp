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

export const DEFAULT_CONFIG: ServerConfig = {
  database: {
    path: '~/.workspace-qdrant/state.db',
  },
  qdrant: {
    url: 'http://localhost:6333',
    timeout: 30000,
  },
  daemon: {
    grpcPort: 50051,
    queuePollIntervalMs: 1000,
    queueBatchSize: 10,
  },
  watching: {
    patterns: ['*.py', '*.rs', '*.md', '*.js', '*.ts'],
    ignorePatterns: [
      '*.pyc',
      '__pycache__/*',
      '.git/*',
      'node_modules/*',
      'target/*',
      '.venv/*',
    ],
  },
  collections: {
    memoryCollectionName: 'memory',
  },
  environment: {},
  memory: {
    limits: {
      maxLabelLength: 15,
      maxTitleLength: 50,
      maxTagLength: 20,
      maxTagsPerRule: 5,
    },
  },
};
