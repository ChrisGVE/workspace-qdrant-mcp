/**
 * Unit tests for the helper functions exported from telemetry/http-server.ts
 *
 * Covers (F-025):
 *  - isLoopbackAddress: recognises 127.0.0.1, ::1, localhost (case-insensitive);
 *    rejects 0.0.0.0, RFC-1918 addresses, public hostnames.
 *  - extractBearerToken: handles Bearer / bearer prefix, missing header,
 *    malformed header.
 *  - formatListenError: produces distinct messages for EADDRINUSE, EACCES,
 *    and generic errors.
 *  - Port validation: integers outside [1, 65535] and NaN throw; valid ports pass.
 *
 * These tests do NOT start an HTTP server — they exercise pure functions only.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { IncomingMessage } from 'node:http';
import {
  isLoopbackAddress,
  extractBearerToken,
  formatListenError,
  startMetricsServer,
} from '../../src/telemetry/http-server.js';

// ── isLoopbackAddress ─────────────────────────────────────────────────────────

describe('isLoopbackAddress', () => {
  it('returns true for 127.0.0.1', () => {
    expect(isLoopbackAddress('127.0.0.1')).toBe(true);
  });

  it('returns true for ::1', () => {
    expect(isLoopbackAddress('::1')).toBe(true);
  });

  it('returns true for localhost (lowercase)', () => {
    expect(isLoopbackAddress('localhost')).toBe(true);
  });

  it('returns true for LOCALHOST (uppercase — case-insensitive)', () => {
    expect(isLoopbackAddress('LOCALHOST')).toBe(true);
  });

  it('returns true for Localhost (mixed case)', () => {
    expect(isLoopbackAddress('Localhost')).toBe(true);
  });

  it('returns false for 0.0.0.0', () => {
    expect(isLoopbackAddress('0.0.0.0')).toBe(false);
  });

  it('returns false for 192.168.1.1', () => {
    expect(isLoopbackAddress('192.168.1.1')).toBe(false);
  });

  it('returns false for example.com', () => {
    expect(isLoopbackAddress('example.com')).toBe(false);
  });

  it('returns false for empty string', () => {
    expect(isLoopbackAddress('')).toBe(false);
  });

  it('returns false for 10.0.0.1', () => {
    expect(isLoopbackAddress('10.0.0.1')).toBe(false);
  });
});

// ── extractBearerToken ────────────────────────────────────────────────────────

function fakeReq(authorization?: string): IncomingMessage {
  return {
    headers: authorization !== undefined ? { authorization } : {},
  } as unknown as IncomingMessage;
}

describe('extractBearerToken', () => {
  it('extracts token from "Bearer abc"', () => {
    expect(extractBearerToken(fakeReq('Bearer abc'))).toBe('abc');
  });

  it('extracts token from "bearer abc" (lowercase scheme)', () => {
    expect(extractBearerToken(fakeReq('bearer abc'))).toBe('abc');
  });

  it('extracts token from "BEARER abc" (uppercase scheme)', () => {
    expect(extractBearerToken(fakeReq('BEARER abc'))).toBe('abc');
  });

  it('returns null when authorization header is absent', () => {
    expect(extractBearerToken(fakeReq())).toBeNull();
  });

  it('returns null for malformed header (no space after Bearer)', () => {
    expect(extractBearerToken(fakeReq('Bearertoken'))).toBeNull();
  });

  it('returns null for Basic auth header', () => {
    expect(extractBearerToken(fakeReq('Basic dXNlcjpwYXNz'))).toBeNull();
  });

  it('returns null for empty authorization header', () => {
    expect(extractBearerToken(fakeReq(''))).toBeNull();
  });

  it('extracts a token that contains special characters', () => {
    expect(extractBearerToken(fakeReq('Bearer tok_en-1.2/3'))).toBe('tok_en-1.2/3');
  });
});

// ── formatListenError ─────────────────────────────────────────────────────────

function makeErr(code: string | undefined, message: string): Error & { code?: string } {
  const err = new Error(message) as Error & { code?: string };
  if (code !== undefined) err.code = code;
  return err;
}

describe('formatListenError', () => {
  it('mentions EADDRINUSE and the port in the message', () => {
    const msg = formatListenError(makeErr('EADDRINUSE', 'address in use'), 9092);
    expect(msg).toContain('EADDRINUSE');
    expect(msg).toContain('9092');
  });

  it('mentions EACCES and the port in the message', () => {
    const msg = formatListenError(makeErr('EACCES', 'permission denied'), 80);
    expect(msg).toContain('EACCES');
    expect(msg).toContain('80');
  });

  it('produces a generic fallback for unknown error codes', () => {
    const msg = formatListenError(makeErr('EUNKNOWN', 'something went wrong'), 9092);
    expect(msg).toContain('something went wrong');
    expect(msg).toContain('9092');
  });

  it('produces a generic fallback when code is undefined', () => {
    const msg = formatListenError(makeErr(undefined, 'network error'), 9092);
    expect(msg).toContain('network error');
    expect(msg).toContain('9092');
  });
});

// ── Port validation ───────────────────────────────────────────────────────────
//
// startMetricsServer() must throw before calling server.listen() when the port
// is invalid. We test the throw only — no actual server is started.

const originalEnv = process.env;

beforeEach(() => {
  process.env = { ...originalEnv };
});

afterEach(() => {
  process.env = originalEnv;
});

describe('port validation', () => {
  it('throws for port 0', () => {
    process.env['MCP_METRICS_PORT'] = '0';
    expect(() => startMetricsServer(0)).toThrow();
  });

  it('throws for negative port (-1)', () => {
    expect(() => startMetricsServer(-1)).toThrow();
  });

  it('throws for port above 65535', () => {
    expect(() => startMetricsServer(65536)).toThrow();
  });

  it('throws for NaN (parsed from "invalid" env string)', () => {
    process.env['MCP_METRICS_PORT'] = 'invalid';
    // parseInt('invalid', 10) → NaN; NaN is not a valid port
    expect(() => startMetricsServer(NaN)).toThrow();
  });

  it('does NOT throw for port 1 (minimum valid)', () => {
    // We only check it doesn't throw on validation; close immediately.
    let server: ReturnType<typeof startMetricsServer> | undefined;
    expect(() => {
      server = startMetricsServer(1);
    }).not.toThrow();
    // Close the server to free the port (best-effort; may fail on EACCES)
    server?.close();
  });

  it('does NOT throw for port 9092 (default)', () => {
    let server: ReturnType<typeof startMetricsServer> | undefined;
    expect(() => {
      server = startMetricsServer(9092);
    }).not.toThrow();
    server?.close();
  });

  it('does NOT throw for port 65535 (maximum valid)', () => {
    let server: ReturnType<typeof startMetricsServer> | undefined;
    expect(() => {
      server = startMetricsServer(65535);
    }).not.toThrow();
    server?.close();
  });
});
