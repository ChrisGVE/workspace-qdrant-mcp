/**
 * Shared types and constants for the MCP server
 */
import type { ServerConfig } from './types/index.js';
export declare const HEARTBEAT_INTERVAL_MS: number;
export declare const SERVER_NAME = "workspace-qdrant-mcp";
export declare const SERVER_VERSION = "0.1.0-beta1 (1285)";
export interface SessionState {
    sessionId: string;
    projectId: string | null;
    projectPath: string | null;
    /** Canonical watch path returned by daemon (may differ from projectPath due to symlink resolution) */
    watchPath: string | null;
    isWorktree: boolean;
    heartbeatInterval: ReturnType<typeof setInterval> | null;
    daemonConnected: boolean;
    /**
     * Idempotence flag for cleanupSession (F-049).
     * Set to `true` on the first cleanup invocation; subsequent calls no-op.
     */
    cleaned: boolean;
    /**
     * Current git branch detected at session start.
     * Used as the default branch filter for search and list tools.
     * "default" when not inside a git repository.
     */
    currentBranch: string | null;
}
/**
 * Transport mode for the MCP server.
 *
 * - `stdio`: MCP over stdin/stdout (default; Claude Desktop, `claude mcp ...` CLI).
 * - `http`:  MCP Streamable HTTP transport. Required for Docker deployments and
 *            for any client that cannot spawn a subprocess.
 * - `test`:  In-process only; the server is constructed and wired but no
 *            transport is connected. Used by unit tests that drive the server
 *            directly through its class API.
 */
export type ServerMode = 'stdio' | 'http' | 'test';
/**
 * Optional TLS configuration for the HTTP transport.
 *
 * When populated the server terminates TLS itself (Node `https` module).
 * Recommended production pattern is to run plain HTTP behind a reverse proxy
 * (Caddy / Traefik / nginx) and leave these fields unset; native TLS exists
 * as a fallback for deployments that do not want a proxy.
 */
export interface HttpTlsOptions {
    /** Filesystem path to the server certificate in PEM format (chain ok). */
    certPath: string;
    /** Filesystem path to the server private key in PEM format. */
    keyPath: string;
    /** Optional PEM path for the CA bundle, for intermediate chain serving. */
    caPath?: string;
}
/**
 * HTTP transport configuration.
 *
 * `host` defaults to `127.0.0.1`. Bind to `0.0.0.0` only inside a container
 * (Docker will expose the listener explicitly via `-p`). `path` is the request
 * route; defaults to `/mcp` to match the Streamable HTTP spec convention.
 */
export interface HttpTransportOptions {
    host: string;
    port: number;
    path: string;
    /** Optional native TLS termination. Unset → plain HTTP. */
    tls?: HttpTlsOptions;
}
export interface ServerOptions {
    config: ServerConfig;
    /**
     * Transport mode. When omitted, `stdio` is inferred from the legacy `stdio`
     * boolean (kept for back-compat with older tests): `stdio: false` → `test`,
     * otherwise `stdio`.
     */
    mode?: ServerMode;
    /** Legacy toggle. Prefer `mode`. `stdio: false` maps to `mode: 'test'`. */
    stdio?: boolean;
    /** HTTP transport settings. Required when `mode === 'http'`. */
    http?: HttpTransportOptions;
    /**
     * HTTP auth / rate limit / CORS. If omitted, the server reads the same
     * settings from the process environment (`MCP_HTTP_TOKEN`,
     * `MCP_HTTP_RATE_LIMIT`, `MCP_HTTP_CORS_ORIGINS`). Tests inject an explicit
     * config to avoid touching real env vars.
     */
    auth?: import('./auth-middleware.js').AuthConfig;
}
/** Default HTTP listener configuration for `mode: 'http'`. */
export declare const DEFAULT_HTTP_HOST = "127.0.0.1";
export declare const DEFAULT_HTTP_PORT = 6335;
export declare const DEFAULT_HTTP_PATH = "/mcp";
//# sourceMappingURL=server-types.d.ts.map