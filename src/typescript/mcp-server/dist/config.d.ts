/**
 * Configuration loading and management
 * Loads from YAML config file with environment variable overrides
 */
import { type ServerConfig } from './types/config.js';
/**
 * Parse a gRPC endpoint string into host and port components.
 *
 * Accepted formats:
 *   - "http://host:port"   (scheme stripped, port parsed)
 *   - "host:port"          (bare host:port)
 *   - "host"               (host only, port defaults to 50051)
 */
export declare function parseGrpcEndpoint(endpoint: string): {
    host: string;
    port: number;
};
export declare function loadConfig(): ServerConfig;
export declare function getDatabasePath(config?: ServerConfig): string;
export declare function getQdrantUrl(config?: ServerConfig): string;
export declare function getDaemonPort(config?: ServerConfig): number;
export declare function getDaemonHost(config?: ServerConfig): string;
//# sourceMappingURL=config.d.ts.map