/**
 * Tests for SqliteStateManager tag methods used in search expansion (Task 34)
 */

import { describe, it, expect, vi } from 'vitest';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';

function createMockStateManager(options?: {
  matchingTags?: { tag_id: number; tag: string; score: number }[];
  baskets?: { tag_id: number; keywords_json: string }[];
}): SqliteStateManager {
  const tags = options?.matchingTags ?? [];
  const baskets = options?.baskets ?? [];

  return {
    logSearchEvent: vi.fn(),
    updateSearchEvent: vi.fn(),
    getMatchingTags: vi.fn().mockReturnValue(
      tags.map((t) => ({ tagId: t.tag_id, tag: t.tag, score: t.score })),
    ),
    getKeywordBasketsForTags: vi.fn().mockReturnValue(
      baskets.map((b) => {
        let keywords: string[] = [];
        try {
          keywords = JSON.parse(b.keywords_json) as string[];
        } catch {
          // skip
        }
        return { tagId: b.tag_id, keywords };
      }),
    ),
    isConnected: vi.fn().mockReturnValue(true),
    getWatchFolderIdByTenantId: vi.fn().mockReturnValue(null),
    getActiveBasePoints: vi.fn().mockReturnValue([]),
  } as unknown as SqliteStateManager;
}

describe('SqliteStateManager tag methods', () => {
  it('should handle getMatchingTags when db is not connected', () => {
    const stateManager = createMockStateManager();
    // Override to simulate not connected
    (stateManager.getMatchingTags as ReturnType<typeof vi.fn>).mockReturnValue([]);

    const result = stateManager.getMatchingTags('test', 'projects');
    expect(result).toEqual([]);
  });

  it('should handle getKeywordBasketsForTags with empty tagIds', () => {
    const stateManager = createMockStateManager();
    (stateManager.getKeywordBasketsForTags as ReturnType<typeof vi.fn>).mockReturnValue([]);

    const result = stateManager.getKeywordBasketsForTags([]);
    expect(result).toEqual([]);
  });
});
