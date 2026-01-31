/**
 * Tests for DaemonEmbedder
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { DaemonEmbedder } from '../../src/clients/daemon-embedder.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';

function createMockDaemonClient(): DaemonClient {
  return {
    isConnected: vi.fn().mockReturnValue(true),
    embedText: vi.fn().mockResolvedValue({
      embedding: new Array(384).fill(0).map((_, i) => i / 384),
      dimensions: 384,
      model_name: 'all-MiniLM-L6-v2',
      success: true,
    }),
    generateSparseVector: vi.fn().mockResolvedValue({
      indices_values: { 1: 0.5, 2: 0.3, 3: 0.2 },
      vocab_size: 1000,
      success: true,
    }),
    // Include other required methods
    connect: vi.fn(),
    close: vi.fn(),
    getConnectionState: vi.fn(),
    healthCheck: vi.fn(),
    getStatus: vi.fn(),
    getMetrics: vi.fn(),
    notifyServerStatus: vi.fn(),
    registerProject: vi.fn(),
    deprioritizeProject: vi.fn(),
    heartbeat: vi.fn(),
    ingestText: vi.fn(),
  } as unknown as DaemonClient;
}

describe('DaemonEmbedder', () => {
  let embedder: DaemonEmbedder;
  let mockDaemonClient: DaemonClient;

  beforeEach(() => {
    vi.clearAllMocks();
    mockDaemonClient = createMockDaemonClient();
    embedder = new DaemonEmbedder(mockDaemonClient);
  });

  describe('generateEmbedding', () => {
    it('should generate embedding for text', async () => {
      const result = await embedder.generateEmbedding('test query');

      expect(result.embedding).toHaveLength(384);
      expect(result.dimensions).toBe(384);
      expect(result.model).toBe('all-MiniLM-L6-v2');
      expect(mockDaemonClient.embedText).toHaveBeenCalledWith({ text: 'test query' });
    });

    it('should throw error for empty text', async () => {
      await expect(embedder.generateEmbedding('')).rejects.toThrow('Text cannot be empty');
      await expect(embedder.generateEmbedding('   ')).rejects.toThrow('Text cannot be empty');
    });

    it('should throw error when daemon fails', async () => {
      vi.mocked(mockDaemonClient.embedText).mockResolvedValue({
        embedding: [],
        dimensions: 0,
        model_name: '',
        success: false,
        error_message: 'Model not initialized',
      });

      await expect(embedder.generateEmbedding('test')).rejects.toThrow('Model not initialized');
    });

    it('should warn on unexpected dimensions', async () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      vi.mocked(mockDaemonClient.embedText).mockResolvedValue({
        embedding: new Array(256).fill(0),
        dimensions: 256,
        model_name: 'different-model',
        success: true,
      });

      const result = await embedder.generateEmbedding('test');

      expect(result.dimensions).toBe(256);
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Unexpected embedding dimensions'));
      consoleSpy.mockRestore();
    });
  });

  describe('generateSparseVector', () => {
    it('should generate sparse vector for text', async () => {
      const result = await embedder.generateSparseVector('test query');

      expect(result.indices).toEqual([1, 2, 3]);
      expect(result.values).toEqual([0.5, 0.3, 0.2]);
      expect(result.vocabSize).toBe(1000);
      expect(mockDaemonClient.generateSparseVector).toHaveBeenCalledWith({ text: 'test query' });
    });

    it('should return empty result for empty text', async () => {
      const result = await embedder.generateSparseVector('');

      expect(result.indices).toEqual([]);
      expect(result.values).toEqual([]);
      expect(result.vocabSize).toBe(0);
      expect(mockDaemonClient.generateSparseVector).not.toHaveBeenCalled();
    });

    it('should return empty result for whitespace-only text', async () => {
      const result = await embedder.generateSparseVector('   ');

      expect(result.indices).toEqual([]);
      expect(result.values).toEqual([]);
    });

    it('should throw error when daemon fails', async () => {
      vi.mocked(mockDaemonClient.generateSparseVector).mockResolvedValue({
        indices_values: {},
        vocab_size: 0,
        success: false,
        error_message: 'BM25 not initialized',
      });

      await expect(embedder.generateSparseVector('test')).rejects.toThrow('BM25 not initialized');
    });
  });

  describe('generateHybridVectors', () => {
    it('should generate both dense and sparse vectors', async () => {
      const result = await embedder.generateHybridVectors('test query');

      expect(result.dense.embedding).toHaveLength(384);
      expect(result.dense.dimensions).toBe(384);
      expect(result.sparse.indices).toEqual([1, 2, 3]);
      expect(result.sparse.values).toEqual([0.5, 0.3, 0.2]);
    });

    it('should call both embedding methods in parallel', async () => {
      await embedder.generateHybridVectors('test query');

      expect(mockDaemonClient.embedText).toHaveBeenCalledTimes(1);
      expect(mockDaemonClient.generateSparseVector).toHaveBeenCalledTimes(1);
    });
  });

  describe('isReady', () => {
    it('should return true when daemon connected', () => {
      vi.mocked(mockDaemonClient.isConnected).mockReturnValue(true);

      expect(embedder.isReady()).toBe(true);
    });

    it('should return false when daemon disconnected', () => {
      vi.mocked(mockDaemonClient.isConnected).mockReturnValue(false);

      expect(embedder.isReady()).toBe(false);
    });
  });

  describe('getModelInfo', () => {
    it('should return model information', () => {
      const info = embedder.getModelInfo();

      expect(info.name).toBe('all-MiniLM-L6-v2');
      expect(info.dimensions).toBe(384);
    });
  });
});

describe('DaemonEmbedder with configuration', () => {
  it('should accept custom configuration', () => {
    const mockClient = createMockDaemonClient();
    const embedder = new DaemonEmbedder(mockClient, {
      retryOnFailure: false,
      maxRetries: 5,
    });

    // Embedder should be created successfully
    expect(embedder).toBeDefined();
    expect(embedder.isReady()).toBe(true);
  });
});
