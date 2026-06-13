import { describe, it, expect } from 'vitest';
import {
  implementationIntentMultiplier,
  queryHasImplementationIntent,
} from '../../src/tools/search-helpers.js';
import type { SearchResult } from '../../src/tools/search-types.js';

function hit(relativePath: string, fileType?: string): SearchResult {
  return {
    id: relativePath,
    score: 1,
    collection: 'projects',
    content: '',
    metadata: {
      relative_path: relativePath,
      ...(fileType ? { file_type: fileType } : {}),
    },
  };
}

describe('implementation intent ranking nudge', () => {
  it('detects implementation-seeking queries without matching generic docs lookups', () => {
    expect(
      queryHasImplementationIntent('Where is the remote embedding provider implemented?')
    ).toBe(true);
    expect(queryHasImplementationIntent('Which code builds the Qdrant point payload?')).toBe(true);
    expect(queryHasImplementationIntent('Onde fica o codigo que calcula o project id?')).toBe(true);

    expect(queryHasImplementationIntent('What is the deployment guide for embeddings?')).toBe(
      false
    );
    expect(queryHasImplementationIntent('Which docs explain API keys?')).toBe(false);
  });

  it('boosts source-like paths and demotes docs/tests only for implementation intent', () => {
    expect(
      implementationIntentMultiplier(hit('src/rust/daemon/core/src/embedding/provider/openai.rs'))
    ).toBeGreaterThan(1);
    expect(
      implementationIntentMultiplier(
        hit('src/typescript/mcp-server/src/proto/workspace_daemon.proto')
      )
    ).toBeGreaterThan(1);
    expect(implementationIntentMultiplier(hit('docs/deployment/embeddings.md'))).toBeLessThan(1);
    expect(
      implementationIntentMultiplier(
        hit('src/typescript/mcp-server/tests/tools/search-rerank-blend.test.ts')
      )
    ).toBeLessThan(1);
    expect(implementationIntentMultiplier(hit('assets/default_configuration.yaml', 'config'))).toBe(
      1
    );
  });
});
