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
    grpcHost: string;
    grpcPort: number;
    queuePollIntervalMs: number;
    queueBatchSize: number;
}
export interface WatchingConfig {
    patterns: string[];
    ignorePatterns: string[];
}
export interface CollectionsConfig {
    rulesCollectionName: string;
}
export interface RuleLimitsConfig {
    maxLabelLength: number;
    maxTitleLength: number;
    maxTagLength: number;
    maxTagsPerRule: number;
}
export interface RuleConfig {
    limits: RuleLimitsConfig;
    duplicationThreshold?: number;
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
    rules?: RuleConfig;
}
export { DEFAULT_CONFIG } from './generated-defaults.js';
//# sourceMappingURL=config.d.ts.map