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
import { timingSafeEqual, createHash } from 'node:crypto';
import { recordHttpAuthFailure, recordHttpRateLimited } from './telemetry/metrics.js';
import { logInfo, logDebug } from './utils/logger.js';
/** Default rate limit (requests per minute per client IP). */
export const DEFAULT_RATE_LIMIT_PER_MIN = 100;
/** Rate-limit window in milliseconds. */
const RATE_LIMIT_WINDOW_MS = 60_000;
/** Hard minimum token length. Rejects trivially-guessable tokens on startup. */
const MIN_TOKEN_LENGTH = 16;
/**
 * Parse environment variables into an `AuthConfig`. Unset values fall back to
 * safe defaults; invalid values throw so misconfiguration surfaces at startup
 * rather than on first request.
 */
export function loadAuthConfig(env = process.env) {
    const token = env['MCP_HTTP_TOKEN'] ?? null;
    const rateLimitRaw = env['MCP_HTTP_RATE_LIMIT'];
    let rateLimitPerMin = DEFAULT_RATE_LIMIT_PER_MIN;
    if (rateLimitRaw !== undefined && rateLimitRaw !== '') {
        const parsed = Number.parseInt(rateLimitRaw, 10);
        if (!Number.isFinite(parsed) || parsed <= 0) {
            throw new Error(`MCP_HTTP_RATE_LIMIT must be a positive integer (got: ${rateLimitRaw})`);
        }
        rateLimitPerMin = parsed;
    }
    const corsRaw = env['MCP_HTTP_CORS_ORIGINS'] ?? '';
    const corsOrigins = corsRaw
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
    return { token, rateLimitPerMin, corsOrigins };
}
/**
 * Enforce that HTTP mode has a usable bearer token. Call from the startup
 * path before binding the listener. Logs a redacted digest of the token so
 * operators can confirm rotation without the secret leaving the process.
 */
export function requireAuth(config) {
    if (config.token === null || config.token === '') {
        throw new Error('MCP_HTTP_TOKEN is required when MCP_SERVER_MODE=http. ' +
            'Generate one with: openssl rand -hex 32');
    }
    if (config.token.length < MIN_TOKEN_LENGTH) {
        throw new Error(`MCP_HTTP_TOKEN must be at least ${MIN_TOKEN_LENGTH} characters (got ${config.token.length}).`);
    }
    logInfo('HTTP auth enabled', {
        tokenDigest: tokenDigest(config.token),
        rateLimitPerMin: config.rateLimitPerMin,
        corsOrigins: config.corsOrigins.length > 0 ? config.corsOrigins : 'disabled',
    });
}
/**
 * Apply CORS headers and handle OPTIONS preflight. Returns `false` if the
 * request was fully handled (preflight) and the caller should stop processing.
 */
function applyCors(req, res, corsOrigins) {
    const origin = req.headers['origin'] ?? '';
    const corsAllowed = corsOrigins.includes(origin);
    if (corsAllowed) {
        res.setHeader('Access-Control-Allow-Origin', origin);
        res.setHeader('Vary', 'Origin');
        res.setHeader('Access-Control-Allow-Credentials', 'true');
    }
    if (req.method === 'OPTIONS') {
        if (corsAllowed) {
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, Accept, Mcp-Session-Id');
            res.setHeader('Access-Control-Max-Age', '600');
        }
        res.writeHead(204);
        res.end();
        return { corsAllowed, handled: true };
    }
    return { corsAllowed, handled: false };
}
/** Check bearer token validity. Returns an AuthDecision or null to continue. */
function checkBearer(req, res, token, clientIp) {
    const presented = extractBearer(req.headers['authorization']);
    if (presented === null) {
        writeUnauthorized(res, 'Missing or malformed Authorization header');
        logDebug('HTTP auth rejected: missing bearer', { clientIp });
        recordHttpAuthFailure('missing_header');
        return { authorized: false };
    }
    if (token === null) {
        writeUnauthorized(res, 'Server is not configured for authentication');
        recordHttpAuthFailure('not_configured');
        return { authorized: false };
    }
    if (!constantTimeEquals(presented, token)) {
        writeUnauthorized(res, 'Invalid token');
        logDebug('HTTP auth rejected: token mismatch', {
            clientIp,
            presentedDigest: tokenDigest(presented),
        });
        recordHttpAuthFailure('invalid_token');
        return { authorized: false };
    }
    return null;
}
/**
 * Build a per-request middleware closure. The returned function inspects the
 * request, applies CORS/rate-limit/auth checks in that order, and returns
 * whether the caller should proceed.
 */
export function createAuthMiddleware(config) {
    const limiter = new SlidingWindowLimiter(config.rateLimitPerMin, RATE_LIMIT_WINDOW_MS);
    return (req, res) => {
        const { handled } = applyCors(req, res, config.corsOrigins);
        if (handled)
            return { authorized: false };
        const clientIp = (req.socket.remoteAddress ?? 'unknown').replace(/^::ffff:/, '');
        if (!limiter.allow(clientIp)) {
            res.writeHead(429, { 'Content-Type': 'text/plain', 'Retry-After': '60' });
            res.end('Too Many Requests');
            logDebug('HTTP rate limit exceeded', { clientIp });
            recordHttpRateLimited();
            return { authorized: false };
        }
        const bearerDecision = checkBearer(req, res, config.token, clientIp);
        if (bearerDecision !== null)
            return bearerDecision;
        return { authorized: true };
    };
}
/** Parse an `Authorization: Bearer <token>` header. Returns the token or `null`. */
export function extractBearer(header) {
    if (header === undefined)
        return null;
    const match = /^Bearer\s+(.+)$/.exec(header.trim());
    if (match === null)
        return null;
    const token = match[1]?.trim() ?? '';
    return token.length === 0 ? null : token;
}
/** Length-insensitive constant-time string compare. */
export function constantTimeEquals(a, b) {
    const bufA = Buffer.from(a, 'utf8');
    const bufB = Buffer.from(b, 'utf8');
    if (bufA.length !== bufB.length) {
        // Still perform a dummy compare so the observable work does not depend
        // on the length relationship between `a` and `b`.
        const dummy = Buffer.alloc(bufA.length);
        timingSafeEqual(bufA, dummy);
        return false;
    }
    return timingSafeEqual(bufA, bufB);
}
/** First 8 hex chars of SHA-256(token) — safe to log for audit/rotation. */
export function tokenDigest(token) {
    return createHash('sha256').update(token, 'utf8').digest('hex').slice(0, 8);
}
function writeUnauthorized(res, message) {
    res.writeHead(401, {
        'Content-Type': 'text/plain',
        'WWW-Authenticate': 'Bearer realm="workspace-qdrant-mcp"',
    });
    res.end(message);
}
/**
 * Minimal in-memory sliding-window limiter. One list of timestamps per IP;
 * entries older than the window are dropped on each touch. Intended for
 * single-process deployments — multi-node deployments should terminate at
 * the reverse proxy where a shared limiter lives.
 */
class SlidingWindowLimiter {
    limit;
    windowMs;
    buckets = new Map();
    constructor(limit, windowMs) {
        this.limit = limit;
        this.windowMs = windowMs;
    }
    allow(key) {
        const now = Date.now();
        const threshold = now - this.windowMs;
        const bucket = this.buckets.get(key) ?? [];
        // Drop stale entries.
        const live = bucket.filter((ts) => ts > threshold);
        if (live.length >= this.limit) {
            this.buckets.set(key, live);
            return false;
        }
        live.push(now);
        this.buckets.set(key, live);
        return true;
    }
}
//# sourceMappingURL=auth-middleware.js.map