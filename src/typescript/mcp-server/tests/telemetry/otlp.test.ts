/**
 * Tests for telemetry/otlp.ts
 *
 * Verifies:
 *  - Default endpoint (http://localhost:4318) used when OTEL_EXPORTER_OTLP_ENDPOINT unset
 *  - Custom endpoint respected when env var is set
 *  - Resolves successfully when collector returns 200
 *  - Resolves (does not throw) when endpoint times out (slow > 1s response)
 *  - Resolves when endpoint returns 500
 *  - Resolves when DNS lookup fails (unreachable host)
 *  - Custom headers from OTEL_EXPORTER_OTLP_HEADERS are forwarded
 *
 * Strategy: monkey-patch globalThis.fetch with a vi.fn() stub.
 * This is simpler than installing nock and avoids ESM interop issues.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { pushMetricsToOTLP } from '../../src/telemetry/otlp.js';

// ── Fetch stub helpers ────────────────────────────────────────────────────────

type FetchStub = ReturnType<typeof vi.fn>;

/** Install a fetch stub on globalThis; returns it for assertion. */
function stubFetch(impl: (url: string, init?: RequestInit) => Promise<Response>): FetchStub {
  const stub = vi.fn(impl);
  vi.stubGlobal('fetch', stub);
  return stub;
}

/** Create a minimal ok Response. */
function okResponse(): Response {
  return new Response(null, { status: 200 });
}

/** Create a minimal error Response. */
function errorResponse(status: number): Response {
  return new Response(null, { status });
}

// ── Setup / teardown ──────────────────────────────────────────────────────────

const originalEnv = process.env;

beforeEach(() => {
  // Isolate env mutations per test
  process.env = { ...originalEnv };
  delete process.env['OTEL_EXPORTER_OTLP_ENDPOINT'];
  delete process.env['OTEL_EXPORTER_OTLP_HEADERS'];
});

afterEach(() => {
  process.env = originalEnv;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ── Endpoint resolution ───────────────────────────────────────────────────────

describe('endpoint resolution', () => {
  it('uses default endpoint when env var is unset', async () => {
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    expect(stub).toHaveBeenCalledOnce();
    const calledUrl = stub.mock.calls[0]?.[0] as string;
    expect(calledUrl).toBe('http://localhost:4318/v1/metrics');
  });

  it('uses custom endpoint when OTEL_EXPORTER_OTLP_ENDPOINT is set', async () => {
    process.env['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://collector.example.com:4318';
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    const calledUrl = stub.mock.calls[0]?.[0] as string;
    expect(calledUrl).toBe('http://collector.example.com:4318/v1/metrics');
  });

  it('strips trailing slash from endpoint', async () => {
    process.env['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318/';
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    const calledUrl = stub.mock.calls[0]?.[0] as string;
    expect(calledUrl).toBe('http://localhost:4318/v1/metrics');
  });
});

// ── Success path ──────────────────────────────────────────────────────────────

describe('successful push', () => {
  it('resolves when collector returns 200', async () => {
    stubFetch(async () => okResponse());
    await expect(pushMetricsToOTLP()).resolves.toBeUndefined();
  });

  it('sends POST with Content-Type application/json', async () => {
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    const init = stub.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe('POST');
    const headers = init.headers as Record<string, string>;
    expect(headers['Content-Type']).toBe('application/json');
  });

  it('body is valid JSON with resourceMetrics key', async () => {
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    const init = stub.mock.calls[0]?.[1] as RequestInit;
    const payload = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(payload).toHaveProperty('resourceMetrics');
    expect(Array.isArray(payload['resourceMetrics'])).toBe(true);
  });
});

// ── Error resilience ──────────────────────────────────────────────────────────

describe('error resilience', () => {
  it('resolves without throwing when collector returns 500', async () => {
    stubFetch(async () => errorResponse(500));
    await expect(pushMetricsToOTLP()).resolves.toBeUndefined();
  });

  it('resolves without throwing when fetch rejects (DNS / network failure)', async () => {
    stubFetch(async () => {
      throw new TypeError('fetch failed: getaddrinfo ENOTFOUND collector.invalid');
    });
    await expect(pushMetricsToOTLP()).resolves.toBeUndefined();
  });

  it('resolves without throwing when AbortSignal timeout fires', async () => {
    stubFetch(async (_url, init) => {
      // Simulate the browser/Node AbortError thrown by AbortSignal.timeout()
      const signal = (init as RequestInit & { signal?: AbortSignal }).signal;
      if (signal) {
        return new Promise<Response>((_resolve, reject) => {
          // Fire synchronously to simulate immediate timeout in tests
          const err = new DOMException('The operation was aborted', 'AbortError');
          reject(err);
        });
      }
      return okResponse();
    });
    await expect(pushMetricsToOTLP()).resolves.toBeUndefined();
  });

  it('resolves without throwing on generic fetch rejection', async () => {
    stubFetch(async () => {
      throw new Error('ECONNREFUSED');
    });
    await expect(pushMetricsToOTLP()).resolves.toBeUndefined();
  });
});

// ── Custom headers ────────────────────────────────────────────────────────────

describe('custom headers', () => {
  it('forwards headers from OTEL_EXPORTER_OTLP_HEADERS', async () => {
    process.env['OTEL_EXPORTER_OTLP_HEADERS'] = 'Authorization=Bearer token123,x-tenant=acme';
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    const init = stub.mock.calls[0]?.[1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer token123');
    expect(headers['x-tenant']).toBe('acme');
  });

  it('sends no extra headers when OTEL_EXPORTER_OTLP_HEADERS is unset', async () => {
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    const init = stub.mock.calls[0]?.[1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    // Only the mandatory Content-Type should be present
    expect(Object.keys(headers)).toEqual(['Content-Type']);
  });
});

// ── AbortSignal present ───────────────────────────────────────────────────────

describe('abort signal', () => {
  it('passes an AbortSignal to fetch', async () => {
    const stub = stubFetch(async () => okResponse());

    await pushMetricsToOTLP();

    const init = stub.mock.calls[0]?.[1] as RequestInit & { signal?: AbortSignal };
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });
});
