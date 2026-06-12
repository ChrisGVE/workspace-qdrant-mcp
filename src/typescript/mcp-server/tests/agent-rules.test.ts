import { describe, expect, it } from 'vitest';

import { formatRulesForPrompt, type Rule } from '../src/agent-rules.js';

describe('agent rules prompt formatting', () => {
  it('includes global rules before project rules in the injected prompt', () => {
    const rules: Rule[] = [
      {
        id: 'global-1',
        scope: 'global',
        content: 'Always search before answering.',
        title: 'Global rule',
        priority: 100,
      },
      {
        id: 'project-1',
        scope: 'project',
        content: 'Use cargo test for Rust changes.',
        title: 'Project rule',
        priority: 50,
      },
    ];

    const prompt = formatRulesForPrompt(rules);

    expect(prompt).toContain('# Behavioral Rules');
    expect(prompt).toContain('## Global Rules');
    expect(prompt).toContain('Always search before answering.');
    expect(prompt).toContain('## Project-Specific Rules');
    expect(prompt).toContain('Use cargo test for Rust changes.');
    expect(prompt.indexOf('## Global Rules')).toBeLessThan(prompt.indexOf('## Project-Specific Rules'));
  });
});
