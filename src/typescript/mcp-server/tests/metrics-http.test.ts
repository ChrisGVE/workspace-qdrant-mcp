/**
 * Unit tests for the HTTP-transport metric helpers in telemetry/metrics.ts.
 *
 * These only exercise label mapping + counter bookkeeping; the Prometheus
 * registry is already integration-tested elsewhere.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  httpAuthFailures,
  httpPathLabel,
  httpRateLimited,
  httpRequests,
  httpStatusClass,
  recordHttpAuthFailure,
  recordHttpRateLimited,
  recordHttpRequest,
} from '../src/telemetry/metrics.js';

describe('metrics (HTTP transport)', () => {
  beforeEach(() => {
    httpRequests.reset();
    httpAuthFailures.reset();
    httpRateLimited.reset();
  });

  describe('httpPathLabel', () => {
    it('maps /mcp and /mcp/... to the mcp bucket', () => {
      expect(httpPathLabel('/mcp')).toBe('mcp');
      expect(httpPathLabel('/mcp/session')).toBe('mcp');
      expect(httpPathLabel('/mcp?trace=1')).toBe('mcp');
    });

    it('maps /healthz exactly', () => {
      expect(httpPathLabel('/healthz')).toBe('/healthz');
      expect(httpPathLabel('/healthz?foo=bar')).toBe('/healthz');
    });

    it('returns `other` for unexpected paths', () => {
      expect(httpPathLabel('/debug')).toBe('other');
      expect(httpPathLabel(undefined)).toBe('other');
      expect(httpPathLabel('')).toBe('other');
    });
  });

  describe('httpStatusClass', () => {
    it('classifies standard status code ranges', () => {
      expect(httpStatusClass(200)).toBe('2xx');
      expect(httpStatusClass(204)).toBe('2xx');
      expect(httpStatusClass(400)).toBe('4xx');
      expect(httpStatusClass(401)).toBe('4xx');
      expect(httpStatusClass(429)).toBe('4xx');
      expect(httpStatusClass(500)).toBe('5xx');
      expect(httpStatusClass(503)).toBe('5xx');
    });

    it('falls back to `other` for unexpected codes', () => {
      expect(httpStatusClass(100)).toBe('other');
      expect(httpStatusClass(301)).toBe('other');
      expect(httpStatusClass(0)).toBe('other');
    });
  });

  describe('recordHttpRequest', () => {
    it('increments the right {path, status_class} cell', async () => {
      recordHttpRequest('/mcp', 200);
      recordHttpRequest('/mcp', 200);
      recordHttpRequest('/healthz', 200);
      recordHttpRequest('/mcp', 401);
      recordHttpRequest('/anything-else', 500);

      const cells = (await httpRequests.get()).values;
      const bucket = (path: string, status: string): number =>
        cells.find((c) => c.labels['path'] === path && c.labels['status_class'] === status)
          ?.value ?? 0;

      expect(bucket('mcp', '2xx')).toBe(2);
      expect(bucket('/healthz', '2xx')).toBe(1);
      expect(bucket('mcp', '4xx')).toBe(1);
      expect(bucket('other', '5xx')).toBe(1);
    });
  });

  describe('recordHttpAuthFailure', () => {
    it('tags failures by reason', async () => {
      recordHttpAuthFailure('missing_header');
      recordHttpAuthFailure('invalid_token');
      recordHttpAuthFailure('invalid_token');
      const cells = (await httpAuthFailures.get()).values;
      const byReason = (reason: string): number =>
        cells.find((c) => c.labels['reason'] === reason)?.value ?? 0;
      expect(byReason('missing_header')).toBe(1);
      expect(byReason('invalid_token')).toBe(2);
    });
  });

  describe('recordHttpRateLimited', () => {
    it('increments the singleton counter', async () => {
      recordHttpRateLimited();
      recordHttpRateLimited();
      recordHttpRateLimited();
      const cells = (await httpRateLimited.get()).values;
      expect(cells[0]?.value).toBe(3);
    });
  });
});
