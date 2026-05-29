/**
 * Tests for telemetry/metrics.ts
 *
 * Verifies:
 *  - Core metrics (tool, session, daemon-fallback, cache) registered with
 *    correct names, label names, and histogram buckets
 *  - withToolMetrics records success and failure cases
 *  - withToolMetrics preserves return value and rethrows errors unchanged
 *  - recordSessionStart / recordSessionEnd update session gauge
 *  - recordDaemonFallback increments the counter
 *  - Git hook metrics (wqm_git_hook_*) registered with bucketed duration
 *    and startGitHookTimer records both invocation counter and histogram
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { Registry, Counter, Histogram, Gauge } from 'prom-client';

// We import the live module but reset its registry state between tests by
// inspecting the shared registry.  Each test grabs metric values from the
// registry directly so that it does not depend on internal state.
import {
  register,
  toolInvocations,
  toolDuration,
  sessionCount,
  daemonFallback,
  cacheHits,
  cacheMisses,
  gitHookInvocations,
  gitHookDuration,
  withToolMetrics,
  recordSessionStart,
  recordSessionEnd,
  recordDaemonFallback,
  startGitHookTimer,
} from '../../src/telemetry/metrics.js';

// Helper: read a counter value for a given label set from the registry
async function getCounterValue(
  metricName: string,
  labels: Record<string, string>
): Promise<number> {
  const metrics = await register.getMetricsAsJSON();
  const metric = metrics.find((m) => m.name === metricName);
  if (!metric) return 0;
  const values = metric.values as Array<{ labels: Record<string, string>; value: number }>;
  const entry = values.find((v) => Object.entries(labels).every(([k, val]) => v.labels[k] === val));
  return entry?.value ?? 0;
}

// Helper: read the current gauge value
async function getGaugeValue(metricName: string): Promise<number> {
  const metrics = await register.getMetricsAsJSON();
  const metric = metrics.find((m) => m.name === metricName);
  if (!metric) return 0;
  const values = metric.values as Array<{ value: number }>;
  return values[0]?.value ?? 0;
}

// Reset all counters/gauges before each test so tests are independent
beforeEach(async () => {
  register.resetMetrics();
});

// ── Registration tests ────────────────────────────────────────────────────────

describe('metric registration', () => {
  it('registers wqm_mcp_tool_invocations_total as a Counter', () => {
    expect(toolInvocations).toBeInstanceOf(Counter);
  });

  it('registers wqm_mcp_tool_duration_seconds as a Histogram', () => {
    expect(toolDuration).toBeInstanceOf(Histogram);
  });

  it('registers wqm_mcp_session_count as a Gauge', () => {
    expect(sessionCount).toBeInstanceOf(Gauge);
  });

  it('registers wqm_mcp_daemon_fallback_total as a Counter', () => {
    expect(daemonFallback).toBeInstanceOf(Counter);
  });

  it('registers wqm_mcp_cache_hits_total as a Counter', () => {
    expect(cacheHits).toBeInstanceOf(Counter);
  });

  it('registers wqm_mcp_cache_misses_total as a Counter', () => {
    expect(cacheMisses).toBeInstanceOf(Counter);
  });

  it('all 6 metrics appear in the registry', async () => {
    const names = (await register.getMetricsAsJSON()).map((m) => m.name);
    expect(names).toContain('wqm_mcp_tool_invocations_total');
    expect(names).toContain('wqm_mcp_tool_duration_seconds');
    expect(names).toContain('wqm_mcp_session_count');
    expect(names).toContain('wqm_mcp_daemon_fallback_total');
    expect(names).toContain('wqm_mcp_cache_hits_total');
    expect(names).toContain('wqm_mcp_cache_misses_total');
  });

  it('wqm_mcp_tool_invocations_total has label names [tool, status]', async () => {
    // Trigger a label observation so label metadata is present
    toolInvocations.labels({ tool: 'search', status: 'success' }).inc();
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_mcp_tool_invocations_total');
    expect(metric).toBeDefined();
    const sampleLabels = (metric!.values[0] as { labels: Record<string, string> }).labels;
    expect(Object.keys(sampleLabels)).toContain('tool');
    expect(Object.keys(sampleLabels)).toContain('status');
  });

  it('wqm_mcp_tool_duration_seconds has label name [tool]', async () => {
    toolDuration.observe({ tool: 'search' }, 0.05);
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_mcp_tool_duration_seconds');
    expect(metric).toBeDefined();
    const sampleLabels = (metric!.values[0] as { labels: Record<string, string> }).labels;
    expect(Object.keys(sampleLabels)).toContain('tool');
  });

  it('wqm_mcp_tool_duration_seconds has the correct buckets', async () => {
    toolDuration.observe({ tool: 'search' }, 0.05);
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_mcp_tool_duration_seconds');
    expect(metric).toBeDefined();
    // Bucket values appear as label le="0.01" etc.
    const bucketLabels = (
      metric!.values as Array<{ labels: Record<string, string>; value: number }>
    )
      .filter((v) => v.labels['le'] !== undefined && v.labels['le'] !== '+Inf')
      .map((v) => parseFloat(v.labels['le'] ?? '0'));
    expect(bucketLabels).toEqual(expect.arrayContaining([0.01, 0.05, 0.1, 0.5, 1, 5]));
  });

  it('wqm_mcp_daemon_fallback_total has label names [tool, reason]', async () => {
    daemonFallback.labels({ tool: 'session', reason: 'connection_failed' }).inc();
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_mcp_daemon_fallback_total');
    expect(metric).toBeDefined();
    const sampleLabels = (metric!.values[0] as { labels: Record<string, string> }).labels;
    expect(Object.keys(sampleLabels)).toContain('tool');
    expect(Object.keys(sampleLabels)).toContain('reason');
  });
});

// ── withToolMetrics ───────────────────────────────────────────────────────────

describe('withToolMetrics', () => {
  it('returns the value from a successful handler', async () => {
    const result = await withToolMetrics('search', async () => 'ok');
    expect(result).toBe('ok');
  });

  it('increments success counter on success', async () => {
    await withToolMetrics('search', async () => 'ok');
    const count = await getCounterValue('wqm_mcp_tool_invocations_total', {
      tool: 'search',
      status: 'success',
    });
    expect(count).toBe(1);
  });

  it('does not increment error counter on success', async () => {
    await withToolMetrics('search', async () => 'ok');
    const count = await getCounterValue('wqm_mcp_tool_invocations_total', {
      tool: 'search',
      status: 'error',
    });
    expect(count).toBe(0);
  });

  it('records a histogram observation on success', async () => {
    await withToolMetrics('retrieve', async () => 42);
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_mcp_tool_duration_seconds');
    // _count bucket should be 1
    const countEntry = (
      metric!.values as Array<{
        labels: Record<string, string>;
        value: number;
        metricName?: string;
      }>
    ).find(
      (v) =>
        v.metricName === 'wqm_mcp_tool_duration_seconds_count' && v.labels['tool'] === 'retrieve'
    );
    expect(countEntry?.value).toBe(1);
  });

  it('rethrows the original error on failure', async () => {
    const original = new Error('tool exploded');
    await expect(
      withToolMetrics('store', async () => {
        throw original;
      })
    ).rejects.toBe(original);
  });

  it('increments error counter on failure', async () => {
    await withToolMetrics('store', async () => {
      throw new Error('boom');
    }).catch(() => {});
    const count = await getCounterValue('wqm_mcp_tool_invocations_total', {
      tool: 'store',
      status: 'error',
    });
    expect(count).toBe(1);
  });

  it('does not increment success counter on failure', async () => {
    await withToolMetrics('store', async () => {
      throw new Error('boom');
    }).catch(() => {});
    const count = await getCounterValue('wqm_mcp_tool_invocations_total', {
      tool: 'store',
      status: 'success',
    });
    expect(count).toBe(0);
  });

  it('records a histogram observation on failure', async () => {
    await withToolMetrics('grep', async () => {
      throw new Error('fail');
    }).catch(() => {});
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_mcp_tool_duration_seconds');
    const countEntry = (
      metric!.values as Array<{
        labels: Record<string, string>;
        value: number;
        metricName?: string;
      }>
    ).find(
      (v) => v.metricName === 'wqm_mcp_tool_duration_seconds_count' && v.labels['tool'] === 'grep'
    );
    expect(countEntry?.value).toBe(1);
  });
});

// ── Session gauge ─────────────────────────────────────────────────────────────

describe('session count gauge', () => {
  it('increments on recordSessionStart', async () => {
    recordSessionStart();
    expect(await getGaugeValue('wqm_mcp_session_count')).toBe(1);
  });

  it('decrements on recordSessionEnd', async () => {
    recordSessionStart();
    recordSessionStart();
    recordSessionEnd();
    expect(await getGaugeValue('wqm_mcp_session_count')).toBe(1);
  });

  it('gauge is 0 after balanced start/end', async () => {
    recordSessionStart();
    recordSessionEnd();
    expect(await getGaugeValue('wqm_mcp_session_count')).toBe(0);
  });
});

// ── Daemon fallback counter ───────────────────────────────────────────────────

describe('recordDaemonFallback', () => {
  it('increments daemon fallback counter', async () => {
    recordDaemonFallback('session', 'connection_failed');
    const count = await getCounterValue('wqm_mcp_daemon_fallback_total', {
      tool: 'session',
      reason: 'connection_failed',
    });
    expect(count).toBe(1);
  });

  it('increments independently for different reason labels', async () => {
    recordDaemonFallback('session', 'connection_failed');
    recordDaemonFallback('session', 'heartbeat_failed');
    const c1 = await getCounterValue('wqm_mcp_daemon_fallback_total', {
      tool: 'session',
      reason: 'connection_failed',
    });
    const c2 = await getCounterValue('wqm_mcp_daemon_fallback_total', {
      tool: 'session',
      reason: 'heartbeat_failed',
    });
    expect(c1).toBe(1);
    expect(c2).toBe(1);
  });
});

// ── Git hook metrics ──────────────────────────────────────────────────────────

describe('git hook metrics', () => {
  it('registers wqm_git_hook_invocations_total as a Counter', () => {
    expect(gitHookInvocations).toBeInstanceOf(Counter);
  });

  it('registers wqm_git_hook_duration_seconds as a Histogram', () => {
    expect(gitHookDuration).toBeInstanceOf(Histogram);
  });

  it('git hook metrics appear in the registry', async () => {
    const names = (await register.getMetricsAsJSON()).map((m) => m.name);
    expect(names).toContain('wqm_git_hook_invocations_total');
    expect(names).toContain('wqm_git_hook_duration_seconds');
  });

  it('wqm_git_hook_invocations_total has label names [hook, result, is_worktree]', async () => {
    gitHookInvocations
      .labels({ hook: 'post-commit', result: 'success', is_worktree: 'false' })
      .inc();
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_git_hook_invocations_total');
    expect(metric).toBeDefined();
    const sampleLabels = (metric!.values[0] as { labels: Record<string, string> }).labels;
    expect(Object.keys(sampleLabels)).toEqual(
      expect.arrayContaining(['hook', 'result', 'is_worktree'])
    );
  });

  it('wqm_git_hook_duration_seconds buckets cover the LSP-startup range', async () => {
    gitHookDuration.observe({ hook: 'post-commit', result: 'success' }, 0.1);
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_git_hook_duration_seconds');
    const bucketLabels = (
      metric!.values as Array<{ labels: Record<string, string>; value: number }>
    )
      .filter((v) => v.labels['le'] !== undefined && v.labels['le'] !== '+Inf')
      .map((v) => parseFloat(v.labels['le'] ?? '0'));
    expect(bucketLabels).toEqual(
      expect.arrayContaining([0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30])
    );
  });

  it('startGitHookTimer records a success invocation', async () => {
    const done = startGitHookTimer('post-merge', false);
    done('success');
    const count = await getCounterValue('wqm_git_hook_invocations_total', {
      hook: 'post-merge',
      result: 'success',
      is_worktree: 'false',
    });
    expect(count).toBe(1);
  });

  it('startGitHookTimer records an error invocation with is_worktree=true', async () => {
    const done = startGitHookTimer('post-checkout', true);
    done('error');
    const count = await getCounterValue('wqm_git_hook_invocations_total', {
      hook: 'post-checkout',
      result: 'error',
      is_worktree: 'true',
    });
    expect(count).toBe(1);
  });

  it('startGitHookTimer records a histogram observation under the matching labels', async () => {
    const done = startGitHookTimer('post-commit', false);
    done('success');
    const metrics = await register.getMetricsAsJSON();
    const metric = metrics.find((m) => m.name === 'wqm_git_hook_duration_seconds');
    const countEntry = (
      metric!.values as Array<{
        labels: Record<string, string>;
        value: number;
        metricName?: string;
      }>
    ).find(
      (v) =>
        v.metricName === 'wqm_git_hook_duration_seconds_count' &&
        v.labels['hook'] === 'post-commit' &&
        v.labels['result'] === 'success'
    );
    expect(countEntry?.value).toBe(1);
  });

  it('startGitHookTimer handles bad_request result without throwing', async () => {
    const done = startGitHookTimer('manual', false);
    expect(() => done('bad_request')).not.toThrow();
    const count = await getCounterValue('wqm_git_hook_invocations_total', {
      hook: 'manual',
      result: 'bad_request',
      is_worktree: 'false',
    });
    expect(count).toBe(1);
  });
});

// ── Registry instance ─────────────────────────────────────────────────────────

describe('register', () => {
  it('is a prom-client Registry instance', () => {
    expect(register).toBeInstanceOf(Registry);
  });

  it('returns Prometheus text format from register.metrics()', async () => {
    const text = await register.metrics();
    // Prometheus text format starts with #
    expect(typeof text).toBe('string');
    expect(text).toMatch(/^#/m);
  });

  it('contentType is the standard Prometheus content type', () => {
    expect(register.contentType).toContain('text/plain');
  });
});
