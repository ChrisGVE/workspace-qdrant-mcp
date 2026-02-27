/**
 * Tests for RetrieveTool - retrieve by document ID and metadata extraction
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

describe('RetrieveTool - retrieve by document ID', () => {
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

describe('RetrieveTool - metadata extraction', () => {
  let mockProjectDetector: ProjectDetector;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockProjectDetector = createMockProjectDetector();
  });

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
