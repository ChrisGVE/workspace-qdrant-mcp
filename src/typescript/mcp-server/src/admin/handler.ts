/**
 * Top-level admin handler: serves the static SPA at `/admin/` and
 * dispatches JSON REST calls under `/admin/api/*`.
 *
 * Called from `mcp-http-server.ts` after the auth middleware passes and
 * before the request falls through to the MCP transport.
 */

import { createReadStream, existsSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { fileURLToPath } from 'node:url';
import { dirname, join, normalize, resolve as resolvePath } from 'node:path';

import { logError } from '../utils/logger.js';

import { dispatchAdminApi, type AdminDeps } from './routes.js';

/**
 * Static file root: `<this-file>/../admin/static`. After build, this
 * resolves to `dist/admin/static`. The build step ships the static
 * assets via `copy:admin-static` (see package.json).
 */
function staticRoot(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  return resolvePath(here, 'static');
}

const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.png': 'image/png',
};

function mimeFor(path: string): string {
  const dot = path.lastIndexOf('.');
  if (dot < 0) return 'application/octet-stream';
  return MIME_TYPES[path.slice(dot).toLowerCase()] ?? 'application/octet-stream';
}

/**
 * Map an `/admin/*` URL to a file inside `staticRoot`, refusing any path
 * that would escape the static root via `..` traversal or symlink games.
 *
 * `/admin/` and `/admin` both resolve to `index.html`. Anything outside
 * the static root returns `null` and the caller serves a 404.
 */
function resolveStaticFile(urlPath: string): string | null {
  const root = staticRoot();

  // Strip the `/admin/` prefix; treat root and bare `/admin` as index.
  let relative = urlPath.replace(/^\/admin\/?/, '');
  if (relative === '' || relative === '/') relative = 'index.html';

  // Refuse anything with `..` after normalization — basic anti-traversal.
  const normalized = normalize(relative);
  if (normalized.split(/[\\/]/).some((seg) => seg === '..')) return null;

  const candidate = join(root, normalized);
  // Belt-and-suspenders: ensure the resolved path is still under root.
  if (!candidate.startsWith(root)) return null;

  if (!existsSync(candidate)) return null;
  try {
    if (!statSync(candidate).isFile()) return null;
  } catch {
    return null;
  }
  return candidate;
}

function serveStatic(req: IncomingMessage, res: ServerResponse, urlPath: string): boolean {
  if (req.method !== 'GET' && req.method !== 'HEAD') return false;

  const file = resolveStaticFile(urlPath);
  if (!file) return false;

  res.writeHead(200, {
    'Content-Type': mimeFor(file),
    'Cache-Control': 'no-cache',
  });

  if (req.method === 'HEAD') {
    res.end();
    return true;
  }

  const stream = createReadStream(file);
  stream.on('error', (err) => {
    logError('admin static stream error', err, { file });
    if (!res.headersSent) {
      res.writeHead(500);
    }
    res.end();
  });
  stream.pipe(res);
  return true;
}

/**
 * Entry point used by `mcp-http-server.ts`.
 *
 * Returns `true` when the request was a `/admin/*` path that was either
 * served (200) or rejected (404/405) — in both cases the HTTP server
 * should NOT fall through to the MCP transport. Returns `false` when
 * the path doesn't look like ours.
 */
export async function handleAdminRequest(
  req: IncomingMessage,
  res: ServerResponse,
  urlPath: string,
  deps: AdminDeps
): Promise<boolean> {
  if (!urlPath.startsWith('/admin')) return false;

  // REST API first; if the path doesn't match a registered route, fall
  // through to the static handler. This lets the SPA itself live at
  // /admin/ without colliding with /admin/api/*.
  if (urlPath.startsWith('/admin/api/')) {
    const handled = await dispatchAdminApi(req, res, urlPath, deps);
    if (handled) return true;
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'unknown admin endpoint', url: urlPath }));
    return true;
  }

  if (serveStatic(req, res, urlPath)) return true;

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
  return true;
}

export type { AdminDeps } from './routes.js';
