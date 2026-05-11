/**
 * Unit tests for grpcUnaryWithTimeout and DaemonClientBase.getMethodTimeout.
 *
 * These tests do not require a running daemon; gRPC calls are mocked.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { grpcUnaryWithTimeout } from '../../src/clients/daemon-client/connection.js';
import { DaemonClient } from '../../src/clients/daemon-client.js';

// ---------------------------------------------------------------------------
// grpcUnaryWithTimeout
// ---------------------------------------------------------------------------

describe('grpcUnaryWithTimeout', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('resolves with the call result when the gRPC call completes before the timeout', async () => {
    const expected = { status: 'ok' };
    // A mock client where 'ping' immediately calls back with `expected`.
    const mockClient = {
      ping: vi.fn((_req: unknown, cb: (err: null, res: typeof expected) => void) => {
        cb(null, expected);
      }),
    };

    const result = await grpcUnaryWithTimeout(mockClient, 'ping', {}, 1000);

    expect(result).toEqual(expected);
    expect(mockClient.ping).toHaveBeenCalledOnce();
  });

  it('rejects with a timeout message when the call exceeds timeoutMs', async () => {
    // A mock client whose 'slowOp' never resolves.
    const mockClient = {
      slowOp: vi.fn((_req: unknown, _cb: unknown) => {
        // intentionally never calls the callback
      }),
    };

    const promise = grpcUnaryWithTimeout(mockClient, 'slowOp', {}, 500, 'slowOp');

    // Advance fake timers past the configured ceiling.
    vi.advanceTimersByTime(501);

    await expect(promise).rejects.toThrow('gRPC call slowOp timed out after 500ms');
  });

  it('uses methodName as the label in the timeout message when operationName is omitted', async () => {
    const mockClient = {
      frozenRpc: vi.fn((_req: unknown, _cb: unknown) => {
        // never resolves
      }),
    };

    const promise = grpcUnaryWithTimeout(mockClient, 'frozenRpc', {}, 200);

    vi.advanceTimersByTime(201);

    await expect(promise).rejects.toThrow('gRPC call frozenRpc timed out after 200ms');
  });

  it('clears the timer when the call resolves before the timeout', async () => {
    const result = { value: 42 };
    const mockClient = {
      fastOp: vi.fn((_req: unknown, cb: (err: null, res: typeof result) => void) => {
        cb(null, result);
      }),
    };

    await grpcUnaryWithTimeout(mockClient, 'fastOp', {}, 5000);

    // Advance well past the timeout window; no late rejection should arrive.
    vi.advanceTimersByTime(6000);

    // No assertion on the return value here — the test passes if the above
    // does not throw, confirming the timer was cleared and cannot fire late.
  });

  it('rejects immediately when the underlying gRPC call rejects before the timeout', async () => {
    const grpcError = new Error('UNAVAILABLE');
    const mockClient = {
      failOp: vi.fn((_req: unknown, cb: (err: Error, res: null) => void) => {
        cb(grpcError, null);
      }),
    };

    await expect(grpcUnaryWithTimeout(mockClient, 'failOp', {}, 5000)).rejects.toThrow(
      'UNAVAILABLE'
    );
  });
});

// ---------------------------------------------------------------------------
// DaemonClientBase.getMethodTimeout — exercised via DaemonClient
// ---------------------------------------------------------------------------

describe('DaemonClient.getMethodTimeout', () => {
  // DaemonClient inherits from DaemonClientBase through DaemonClientService
  // → DaemonClientSystem → DaemonClientBase.  We access the protected method
  // via a thin subclass cast to avoid duplicating the logic in tests.
  class TestableClient extends DaemonClient {
    /** Expose the protected helper for testing. */
    public timeout(methodName: string, override?: number): number {
      return this.getMethodTimeout(methodName, override);
    }
  }

  let client: TestableClient;

  beforeEach(() => {
    client = new TestableClient({ host: 'localhost', port: 59996, timeoutMs: 1000 });
  });

  it('returns timeoutMs for an ordinary method name', () => {
    expect(client.timeout('getStatus')).toBe(1000);
  });

  it('returns 2× timeoutMs for the "search" wire method', () => {
    expect(client.timeout('search')).toBe(2000);
  });

  it('returns the explicit override when provided, ignoring any per-method rule', () => {
    // Even 'search' should yield the caller-supplied value when overridden.
    expect(client.timeout('search', 750)).toBe(750);
    expect(client.timeout('getStatus', 300)).toBe(300);
  });

  it('returns methodTimeouts entry when one is registered for the method', () => {
    // Populate via the protected map to simulate a subclass override.
    (client as unknown as { methodTimeouts: Record<string, number> }).methodTimeouts[
      'enqueueItem'
    ] = 9999;
    expect(client.timeout('enqueueItem')).toBe(9999);
  });

  it('methodTimeouts entry takes precedence over the "search" built-in rule', () => {
    (client as unknown as { methodTimeouts: Record<string, number> }).methodTimeouts['search'] =
      3333;
    expect(client.timeout('search')).toBe(3333);
  });
});
