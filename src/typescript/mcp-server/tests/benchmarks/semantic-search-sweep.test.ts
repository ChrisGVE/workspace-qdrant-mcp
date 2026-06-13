import { describe, expect, it } from 'vitest';

import {
  buildDefaultScenarios,
  parseScenarioSpec,
} from '../../scripts/benchmark-semantic-search-sweep.js';

describe('semantic-search sweep scenario parsing', () => {
  it('parses named rerank scenarios', () => {
    expect(parseScenarioSpec('current')).toEqual({ name: 'current' });
    expect(parseScenarioSpec('off:rerank=false')).toEqual({ name: 'off', rerank: false });
    expect(parseScenarioSpec('weak:rerank=true,weight=0.25')).toEqual({
      name: 'weak',
      rerank: true,
      rerankWeight: 0.25,
    });
    expect(parseScenarioSpec('pure:w=1')).toEqual({
      name: 'pure',
      rerank: true,
      rerankWeight: 1,
    });
  });

  it('builds default current/off/weight sweep scenarios', () => {
    expect(buildDefaultScenarios('0,0.5')).toEqual([
      { name: 'current' },
      { name: 'rerank-off', rerank: false },
      { name: 'rerank-0', rerank: true, rerankWeight: 0 },
      { name: 'rerank-0.5', rerank: true, rerankWeight: 0.5 },
    ]);
  });

  it('rejects unsupported weights and settings', () => {
    expect(() => parseScenarioSpec('bad:w=2')).toThrow(/between 0 and 1/);
    expect(() => parseScenarioSpec('bad:foo=bar')).toThrow(/unsupported/i);
  });
});
