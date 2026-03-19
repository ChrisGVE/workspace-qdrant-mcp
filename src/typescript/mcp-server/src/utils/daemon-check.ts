/**
 * Daemon availability check with TTL-based caching.
 *
 * Verifies daemon connectivity before write operations, with a 5-second
 * cache to avoid repeated health checks on rapid successive calls.
 */

import type { DaemonClient } from '../clients/daemon-client.js';

let lastCheck: { available: boolean; timestamp: number } | null = null;
const TTL_MS = 5000;

/**
 * Ensure the daemon is available for write operations.
 *
 * Returns the validated DaemonClient or throws if unavailable.
 * Caches the result for TTL_MS to reduce health check overhead.
 */
export async function ensureDaemonAvailable(client: DaemonClient | null): Promise<DaemonClient> {
  if (!client) {
    throw new Error('Daemon unavailable. Cannot process write operation.');
  }
  const now = Date.now();
  if (lastCheck && now - lastCheck.timestamp < TTL_MS) {
    if (!lastCheck.available) {
      throw new Error('Daemon unavailable. Cannot process write operation.');
    }
    return client;
  }
  try {
    await client.healthCheck();
    lastCheck = { available: true, timestamp: now };
    return client;
  } catch {
    lastCheck = { available: false, timestamp: now };
    throw new Error('Daemon unavailable. Cannot process write operation.');
  }
}

/** Reset the cached check (useful for testing). */
export function resetDaemonCheck(): void {
  lastCheck = null;
}
