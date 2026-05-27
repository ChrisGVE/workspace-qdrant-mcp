/**
 * Per-request execution context for MCP tool handlers.
 *
 * The MCP server can run in two transports:
 *   - stdio: the process is spawned by the client; `process.cwd()` already
 *     reflects the user's working directory.
 *   - http  (typically Docker): a long-lived container whose `process.cwd()`
 *     is fixed at WORKDIR and unrelated to the client. Per-request metadata
 *     (e.g. the host CWD) must be carried explicitly from the client.
 *
 * `getEffectiveCwd()` is the single source of truth tool handlers should
 * use instead of `process.cwd()`. Its resolution order is:
 *
 *   1. Per-request override stored in {@link requestContext} (set by the HTTP
 *      transport from the `X-MCP-Host-Cwd` header).
 *   2. `WQM_DEFAULT_HOST_CWD` env var — useful as a stdio fallback when the
 *      MCP is launched from a directory that does not match any registered
 *      project (e.g. Claude Code starts MCP from the user's home).
 *   3. `process.cwd()` — works for stdio when the client launches the MCP
 *      from inside the project / worktree.
 */

import { AsyncLocalStorage } from 'node:async_hooks';

export interface RequestContext {
  /** Host-side absolute path the client is operating from. */
  hostCwd?: string;
}

const storage = new AsyncLocalStorage<RequestContext>();

/** Run `fn` with the given request context bound to AsyncLocalStorage. */
export function runWithRequestContext<T>(ctx: RequestContext, fn: () => T): T {
  return storage.run(ctx, fn);
}

/** Read the current request context, or `undefined` outside a request. */
export function getRequestContext(): RequestContext | undefined {
  return storage.getStore();
}

/**
 * Resolve the effective working directory for project detection.
 *
 * See module docs for the full resolution chain.
 */
export function getEffectiveCwd(): string {
  const ctx = storage.getStore();
  if (ctx?.hostCwd && ctx.hostCwd.length > 0) {
    return ctx.hostCwd;
  }
  const envCwd = process.env.WQM_DEFAULT_HOST_CWD;
  if (envCwd && envCwd.length > 0) {
    return envCwd;
  }
  return process.cwd();
}
