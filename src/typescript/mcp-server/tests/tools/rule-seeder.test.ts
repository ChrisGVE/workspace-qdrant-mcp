/**
 * Tests for the per-label idempotency guard in seedDefaultRule.
 *
 * The guard must never duplicate an existing default — a fresh install gets all
 * of them, a partially-seeded install gets only the missing labels, and a list
 * failure skips seeding entirely (cannot dedup without the existing set).
 */

import { describe, it, expect, vi } from 'vitest';
import { seedDefaultRule, DEFAULT_RULES } from '../../src/rule-seeder.js';
import type { RulesTool } from '../../src/tools/index.js';

type ExecArgs = { action: string; label?: string };

function mockRulesTool(listResult: unknown) {
  const calls: ExecArgs[] = [];
  const execute = vi.fn(async (args: ExecArgs) => {
    calls.push(args);
    if (args.action === 'list') return listResult;
    return { success: true, action: 'add', label: args.label };
  });
  return { tool: { execute } as unknown as RulesTool, calls, execute };
}

const addedLabels = (calls: ExecArgs[]) =>
  calls.filter((c) => c.action === 'add').map((c) => c.label);

describe('seedDefaultRule (per-label idempotency)', () => {
  it('seeds every default on a fresh (empty) install', async () => {
    const { tool, calls } = mockRulesTool({ success: true, action: 'list', rules: [] });

    await seedDefaultRule(tool);

    const added = addedLabels(calls);
    expect(added).toHaveLength(DEFAULT_RULES.length);
    expect(new Set(added)).toEqual(new Set(DEFAULT_RULES.map((r) => r.label)));
  });

  it('backfills only the missing labels, never duplicating an existing one', async () => {
    const present = ['search-first', 'grep-for-exact'];
    const { tool, calls } = mockRulesTool({
      success: true,
      action: 'list',
      rules: present.map((label) => ({ id: label, label, content: 'x', scope: 'global' })),
    });

    await seedDefaultRule(tool);

    const added = addedLabels(calls);
    // none of the already-present labels get re-added
    for (const p of present) expect(added).not.toContain(p);
    // every other default is backfilled
    expect(added).toHaveLength(DEFAULT_RULES.length - present.length);
  });

  it('adds nothing when all defaults already exist', async () => {
    const { tool, calls } = mockRulesTool({
      success: true,
      action: 'list',
      rules: DEFAULT_RULES.map((r) => ({ id: r.label, label: r.label, content: 'x', scope: 'global' })),
    });

    await seedDefaultRule(tool);

    expect(addedLabels(calls)).toHaveLength(0);
  });

  it('skips seeding entirely when the list call fails', async () => {
    const { tool, calls } = mockRulesTool({ success: false, action: 'list' });

    await seedDefaultRule(tool);

    expect(addedLabels(calls)).toHaveLength(0);
  });
});
