/**
 * Regression tests for findRuleIdByLabel.
 *
 * The daemon derives the Qdrant point id from (tenant, branch, document_id,
 * chunk), so an UPDATE must re-supply the stored `document_id` — NOT the point
 * id — to land on the same point. Returning the point id made the daemon derive
 * a fresh id and create a DUPLICATE rule instead of updating in place.
 */

import { describe, it, expect, vi } from 'vitest';
import type { QdrantClient } from '@qdrant/js-client-rest';
import { findRuleIdByLabel } from '../../src/tools/rules-mutation-helpers.js';

function qdrantWith(scrollImpl: (coll: string, req: unknown) => Promise<unknown>): QdrantClient {
  return { scroll: vi.fn(scrollImpl) } as unknown as QdrantClient;
}

describe('findRuleIdByLabel', () => {
  it('returns the stored document_id, not the (derived) Qdrant point id', async () => {
    const q = qdrantWith(async () => ({
      points: [{ id: 'derived-point-id-XXXX', payload: { document_id: 'original-doc-id-1234' } }],
    }));

    const id = await findRuleIdByLabel(q, 'grep-for-exact', 'global', undefined);

    expect(id).toBe('original-doc-id-1234');
    expect(id).not.toBe('derived-point-id-XXXX');
  });

  it('requests the payload (with_payload:true) so document_id is available', async () => {
    const scroll = vi.fn(async () => ({
      points: [{ id: 'p', payload: { document_id: 'd' } }],
    }));
    const q = { scroll } as unknown as QdrantClient;

    await findRuleIdByLabel(q, 'search-first', 'global', undefined);

    expect(scroll).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ with_payload: true })
    );
  });

  it('falls back to the point id when a legacy point has no stored document_id', async () => {
    const q = qdrantWith(async () => ({ points: [{ id: 'legacy-point-id', payload: {} }] }));

    const id = await findRuleIdByLabel(q, 'old-rule', 'global', undefined);

    expect(id).toBe('legacy-point-id');
  });

  it('returns null when no rule matches the label/scope', async () => {
    const q = qdrantWith(async () => ({ points: [] }));

    const id = await findRuleIdByLabel(q, 'missing', 'global', undefined);

    expect(id).toBeNull();
  });

  it('scopes project rules by project_id in the filter', async () => {
    const scroll = vi.fn(async () => ({ points: [{ id: 'p', payload: { document_id: 'd' } }] }));
    const q = { scroll } as unknown as QdrantClient;

    await findRuleIdByLabel(q, 'proj-rule', 'project', 'tenant-xyz');

    const req = scroll.mock.calls[0][1] as { filter: { must: Array<{ match: { value: unknown } }> } };
    const values = req.filter.must.map((c) => c.match.value);
    expect(values).toContain('tenant-xyz');
  });
});
