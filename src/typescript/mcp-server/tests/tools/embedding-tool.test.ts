/**
 * Tests for the `embedding` MCP tool handler.
 */

import { describe, it, expect, vi } from 'vitest';
import { handleEmbedding } from '../../src/tools/embedding.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';

function makeClient(stub: Partial<DaemonClient>): DaemonClient {
  return stub as unknown as DaemonClient;
}

describe('embedding tool', () => {
  it('returns success with provider metadata + probe state when daemon responds', async () => {
    const getEmbeddingProviderStatus = vi.fn().mockResolvedValue({
      provider: 'openai_compatible',
      model: 'text-embedding-3-small',
      output_dim: 1536,
      base_url: 'https://api.openai.com',
      probe_status: 'healthy',
      probe_message: 'Running normally',
    });

    const result = await handleEmbedding(undefined, makeClient({ getEmbeddingProviderStatus }));

    expect(result.success).toBe(true);
    expect(result.provider).toBe('openai_compatible');
    expect(result.model).toBe('text-embedding-3-small');
    expect(result.output_dim).toBe(1536);
    expect(result.probe_status).toBe('healthy');
    expect(getEmbeddingProviderStatus).toHaveBeenCalledTimes(1);
  });

  it('returns success=false with error message when the gRPC call rejects', async () => {
    const getEmbeddingProviderStatus = vi.fn().mockRejectedValue(new Error('connection refused'));

    const result = await handleEmbedding(undefined, makeClient({ getEmbeddingProviderStatus }));

    expect(result.success).toBe(false);
    expect(result.error).toMatch(/connection refused/);
    expect(result.provider).toBeUndefined();
  });
});
