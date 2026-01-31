/**
 * Tests for RetrieveTool
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RetrieveTool, type RetrieveOptions } from '../../src/tools/retrieve.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Mock the Qdrant client
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    retrieve: vi.fn().mockResolvedValue([
      {
        id: 'doc-123',
        payload: {
          content: 'Document content here',
          title: 'Test Document',
          source_type: 'user_input',
          tenant_id: 'test-project',
        },
      },
    ]),
    scroll: vi.fn().mockResolvedValue({
      points: [
        {
          id: 'doc-1',
          payload: {
            content: 'First document',
            title: 'Doc 1',
            source_type: 'file',
            tenant_id: 'test-project',
          },
        },
        {
          id: 'doc-2',
          payload: {
            content: 'Second document',
            title: 'Doc 2',
            source_type: 'web',
            tenant_id: 'test-project',
          },
        },
      ],
    }),
  })),
}));

function createMockProjectDetector(): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/test/project'),
    getProjectInfo: vi.fn().mockResolvedValue({
      projectId: 'test-project-123',
      projectPath: '/test/project',
      name: 'test-project',
    }),
  } as unknown as ProjectDetector;
}

describe('RetrieveTool', () => {
  let retrieveTool: RetrieveTool;
  let mockProjectDetector: ProjectDetector;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockProjectDetector = createMockProjectDetector();

    retrieveTool = new RetrieveTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockProjectDetector
    );
  });

  describe('retrieve by document ID', () => {
    it('should retrieve a document by ID', async () => {
      const options: RetrieveOptions = {
        documentId: 'doc-123',
        collection: 'projects',
      };

      const result = await retrieveTool.retrieve(options);

      expect(result.success).toBe(true);
      expect(result.documents).toHaveLength(1);
      expect(result.documents[0].id).toBe('doc-123');
      expect(result.documents[0].content).toBe('Document content here');
      expect(result.documents[0].metadata.title).toBe('Test Document');
      expect(result.total).toBe(1);
      expect(result.hasMore).toBe(false);
    });

    it('should exclude content from metadata', async () => {
      const options: RetrieveOptions = {
        documentId: 'doc-123',
        collection: 'projects',
      };

      const result = await retrieveTool.retrieve(options);

      expect(result.documents[0].content).toBe('Document content here');
      expect(result.documents[0].metadata.content).toBeUndefined();
    });

    it('should return not found for missing document', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            retrieve: vi.fn().mockResolvedValue([]),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({
        documentId: 'nonexistent',
        collection: 'projects',
      });

      expect(result.success).toBe(false);
      expect(result.documents).toHaveLength(0);
      expect(result.message).toContain('Document not found');
    });

    it('should handle retrieve errors gracefully', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            retrieve: vi.fn().mockRejectedValue(new Error('Connection failed')),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({
        documentId: 'doc-123',
        collection: 'projects',
      });

      expect(result.success).toBe(false);
      expect(result.message).toContain('Failed to retrieve document');
    });
  });

  describe('retrieve by filter', () => {
    it('should retrieve documents using scroll', async () => {
      const options: RetrieveOptions = {
        collection: 'projects',
        limit: 10,
      };

      const result = await retrieveTool.retrieve(options);

      expect(result.success).toBe(true);
      expect(result.documents).toHaveLength(2);
      expect(result.documents[0].id).toBe('doc-1');
      expect(result.documents[1].id).toBe('doc-2');
    });

    it('should apply custom filter conditions', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({
        collection: 'projects',
        filter: { source_type: 'file', file_type: 'typescript' },
      });

      expect(scrollMock).toHaveBeenCalledWith(
        'projects',
        expect.objectContaining({
          filter: expect.objectContaining({
            must: expect.arrayContaining([
              expect.objectContaining({ key: 'source_type', match: { value: 'file' } }),
              expect.objectContaining({ key: 'file_type', match: { value: 'typescript' } }),
            ]),
          }),
        })
      );
    });

    it('should apply tenant filter for projects collection', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({
        collection: 'projects',
        projectId: 'my-project',
      });

      expect(scrollMock).toHaveBeenCalledWith(
        'projects',
        expect.objectContaining({
          filter: expect.objectContaining({
            must: expect.arrayContaining([
              expect.objectContaining({ key: 'tenant_id', match: { value: 'my-project' } }),
            ]),
          }),
        })
      );
    });

    it('should use project detector when projectId not provided', async () => {
      await retrieveTool.retrieve({
        collection: 'projects',
      });

      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
    });

    it('should apply tenant filter for libraries collection', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({
        collection: 'libraries',
        libraryName: 'react',
      });

      expect(scrollMock).toHaveBeenCalledWith(
        'libraries',
        expect.objectContaining({
          filter: expect.objectContaining({
            must: expect.arrayContaining([
              expect.objectContaining({ key: 'tenant_id', match: { value: 'react' } }),
            ]),
          }),
        })
      );
    });
  });

  describe('pagination', () => {
    it('should respect limit parameter', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({
        collection: 'projects',
        limit: 5,
      });

      // Request limit+1 to check hasMore
      expect(scrollMock).toHaveBeenCalledWith(
        'projects',
        expect.objectContaining({
          limit: 6,
        })
      );
    });

    it('should apply offset parameter', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({
        collection: 'projects',
        offset: 10,
      });

      expect(scrollMock).toHaveBeenCalledWith(
        'projects',
        expect.objectContaining({
          offset: 10,
        })
      );
    });

    it('should indicate hasMore when more results exist', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: vi.fn().mockResolvedValue({
              points: [
                { id: '1', payload: { content: 'doc 1' } },
                { id: '2', payload: { content: 'doc 2' } },
                { id: '3', payload: { content: 'doc 3' } }, // Extra result indicates more
              ],
            }),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({
        collection: 'projects',
        limit: 2,
      });

      expect(result.hasMore).toBe(true);
      expect(result.documents).toHaveLength(2); // Returns only requested limit
    });

    it('should indicate no more when at end', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: vi.fn().mockResolvedValue({
              points: [
                { id: '1', payload: { content: 'doc 1' } },
              ],
            }),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({
        collection: 'projects',
        limit: 10,
      });

      expect(result.hasMore).toBe(false);
    });
  });

  describe('collection handling', () => {
    it('should use projects collection by default', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({});

      expect(scrollMock).toHaveBeenCalledWith('projects', expect.any(Object));
    });

    it('should use libraries collection when specified', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({ collection: 'libraries' });

      expect(scrollMock).toHaveBeenCalledWith('libraries', expect.any(Object));
    });

    it('should use memory collection when specified', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      const scrollMock = vi.fn().mockResolvedValue({ points: [] });
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: scrollMock,
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      await newTool.retrieve({ collection: 'memory' });

      expect(scrollMock).toHaveBeenCalledWith('memory', expect.any(Object));
    });
  });

  describe('error handling', () => {
    it('should handle collection not found gracefully', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: vi.fn().mockRejectedValue(new Error("Collection not found")),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({ collection: 'projects' });

      expect(result.success).toBe(true);
      expect(result.documents).toHaveLength(0);
      expect(result.message).toContain('Collection not found');
    });

    it('should handle scroll errors gracefully', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: vi.fn().mockRejectedValue(new Error('Connection timeout')),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({ collection: 'projects' });

      expect(result.success).toBe(false);
      expect(result.message).toContain('Failed to retrieve documents');
    });
  });

  describe('metadata extraction', () => {
    it('should exclude vector fields from metadata', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            retrieve: vi.fn().mockResolvedValue([
              {
                id: 'doc-123',
                payload: {
                  content: 'Document content',
                  dense_vector: [0.1, 0.2, 0.3],
                  sparse_vector: { indices: [1, 2], values: [0.5, 0.5] },
                  title: 'Test',
                },
              },
            ]),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({ documentId: 'doc-123' });

      expect(result.documents[0].metadata.title).toBe('Test');
      expect(result.documents[0].metadata.dense_vector).toBeUndefined();
      expect(result.documents[0].metadata.sparse_vector).toBeUndefined();
      expect(result.documents[0].metadata.content).toBeUndefined();
    });

    it('should handle null payload gracefully', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            retrieve: vi.fn().mockResolvedValue([
              {
                id: 'doc-123',
                payload: null,
              },
            ]),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new RetrieveTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockProjectDetector
      );

      const result = await newTool.retrieve({ documentId: 'doc-123' });

      expect(result.success).toBe(true);
      expect(result.documents[0].content).toBe('');
      expect(result.documents[0].metadata).toEqual({});
    });
  });
});
