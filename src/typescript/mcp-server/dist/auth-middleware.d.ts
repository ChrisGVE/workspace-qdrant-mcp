/**
 * HTTP authentication, rate limiting, and CORS for MCP HTTP mode.
 *
 * The design is intentionally minimal and stateless (no DB, no user records):
 *
 * 1. **Bearer token.** Clients send `Authorization: Bearer <token>`. The
 *    token is compared to the configured secret with `timingSafeEqual` to
 *    avoid leaking length/content via timing. The server refuses to start
 *    in `http` mode without a token — mis-configuration fails loudly.
 *
 * 2. **Rate limit.** A sliding-window counter per client IP caps abuse at
 *    `MCP_HTTP_RATE_LIMIT` requests per 60 seconds (default 100). The
 *    counter lives in process memory; it is *not* a DoS mitigation, only a
 *    tripwire against misbehaving clients.
 *
 * 3. **CORS.** When `MCP_HTTP_CORS_ORIGINS` is set (comma-separated), the
 *    server echoes the matching `Origin` header and answers `OPTIONS`
 *    preflights. Otherwise CORS headers are never emitted — browsers from
 *    third-party origins are blocked by the same-origin policy.
 *
 * None of these are substitutes for running behind a reverse proxy (Caddy,
 * Traefik) with TLS. They are the baseline guarantees the MCP server
 * itself provides when its port is exposed.
 */
import type { IncomingMessage, ServerResponse } from 'node:http';
/** Default rate limit (requests per minute per client IP). */
export declare const DEFAULT_RATE_LIMIT_PER_MIN = 100;
export interface AuthConfig {
    /**
     * The bearer secret. `null` disables auth entirely — only valid for
     * non-production modes; `requireAuth()` throws if the secret is missing
     * when HTTP mode is active.
     */
    token: string | null;
    /** Max requests per minute per client IP. */
    rateLimitPerMin: number;
    /**
     * Allowed CORS origins. Empty array → CORS disabled (default); browsers on
     * other origins are blocked by the same-origin policy.
     */
    corsOrigins: string[];
}
/**
 * Decide outcome of a single request. Either the caller proceeds to the MCP
 * transport (`authorized: true`) or the middleware has already written a
 * terminal response (`authorized: false`).
 */
export interface AuthDecision {
    authorized: boolean;
}
/**
 * Parse environment variables into an `AuthConfig`. Unset values fall back to
 * safe defaults; invalid values throw so misconfiguration surfaces at startup
 * rather than on first request.
 */
export declare function loadAuthConfig(env?: NodeJS.ProcessEnv): AuthConfig;
/**
 * Enforce that HTTP mode has a usable bearer token. Call from the startup
 * path before binding the listener. Logs a redacted digest of the token so
 * operators can confirm rotation without the secret leaving the process.
 */
export declare function requireAuth(config: AuthConfig): void;
/**
 * Build a per-request middleware closure. The returned function inspects the
 * request, applies CORS/rate-limit/auth checks in that order, and returns
 * whether the caller should proceed.
 */
export declare function createAuthMiddleware(config: AuthConfig): (req: IncomingMessage, res: ServerResponse) => AuthDecision;
/** Parse an `Authorization: Bearer <token>` header. Returns the token or `null`. */
export declare function extractBearer(header: string | undefined): string | null;
/** Length-insensitive constant-time string compare. */
export declare function constantTimeEquals(a: string, b: string): boolean;
/** First 8 hex chars of SHA-256(token) — safe to log for audit/rotation. */
export declare function tokenDigest(token: string): string;
//# sourceMappingURL=auth-middleware.d.ts.map