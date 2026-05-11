/**
 * Unit tests for httpPathLabel() in telemetry/metrics.ts (F-021).
 *
 * MCP_HTTP_PATH is cached at module load, so each test group that changes
 * the env var must use vi.resetModules() + dynamic re-import to get a fresh
 * module instance with the new cached value.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const originalEnv = process.env;

beforeEach(() => {
  process.env = { ...originalEnv };
  vi.resetModules();
});

afterEach(() => {
  process.env = originalEnv;
  vi.resetModules();
});

// ── Default path (/mcp) ──────────────────────────────────────────────────────

describe('httpPathLabel with default MCP_HTTP_PATH (/mcp)', () => {
  it('labels /mcp as "mcp"', async () => {
    delete process.env['MCP_HTTP_PATH'];
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/mcp')).toBe('mcp');
  });

  it('labels /mcp/session/abc as "mcp" (sub-path)', async () => {
    delete process.env['MCP_HTTP_PATH'];
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/mcp/session/abc')).toBe('mcp');
  });

  it('labels /healthz as "/healthz"', async () => {
    delete process.env['MCP_HTTP_PATH'];
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/healthz')).toBe('/healthz');
  });

  it('labels /unknown as "other"', async () => {
    delete process.env['MCP_HTTP_PATH'];
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/unknown')).toBe('other');
  });

  it('labels undefined as "other"', async () => {
    delete process.env['MCP_HTTP_PATH'];
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel(undefined)).toBe('other');
  });

  it('strips query string before matching', async () => {
    delete process.env['MCP_HTTP_PATH'];
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/mcp?foo=bar')).toBe('mcp');
  });
});

// ── Custom path (/api/mcp) ────────────────────────────────────────────────────

describe('httpPathLabel with MCP_HTTP_PATH=/api/mcp', () => {
  it('labels /api/mcp as "mcp"', async () => {
    process.env['MCP_HTTP_PATH'] = '/api/mcp';
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/api/mcp')).toBe('mcp');
  });

  it('labels /api/mcp/sub as "mcp"', async () => {
    process.env['MCP_HTTP_PATH'] = '/api/mcp';
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/api/mcp/sub')).toBe('mcp');
  });

  it('does NOT label bare /mcp as "mcp" when path is /api/mcp', async () => {
    process.env['MCP_HTTP_PATH'] = '/api/mcp';
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/mcp')).toBe('other');
  });

  it('labels /healthz as "/healthz" regardless of custom path', async () => {
    process.env['MCP_HTTP_PATH'] = '/api/mcp';
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/healthz')).toBe('/healthz');
  });

  it('labels everything else as "other"', async () => {
    process.env['MCP_HTTP_PATH'] = '/api/mcp';
    const { httpPathLabel } = await import('../../src/telemetry/metrics.js');
    expect(httpPathLabel('/metrics')).toBe('other');
    expect(httpPathLabel('/api')).toBe('other');
  });
});
