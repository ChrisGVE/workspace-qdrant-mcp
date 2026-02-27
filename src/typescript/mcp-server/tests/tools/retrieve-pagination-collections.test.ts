/**
 * Tests for RetrieveTool - pagination, collection handling, and error handling
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RetrieveTool } from '../../src/tools/retrieve.js';
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

describe('RetrieveTool - pagination', () => {
  let mockProjectDetector: ProjectDetector;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockProjectDetector = createMockProjectDetector();
  });

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

describe('RetrieveTool - collection handling', () => {
  let mockProjectDetector: ProjectDetector;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockProjectDetector = createMockProjectDetector();
  });

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

    await newTool.retrieve({ collection: 'rules' });

    expect(scrollMock).toHaveBeenCalledWith('rules', expect.any(Object));
  });
});

describe('RetrieveTool - error handling', () => {
  let mockProjectDetector: ProjectDetector;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockProjectDetector = createMockProjectDetector();
  });

  it('should handle collection not found gracefully', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          scroll: vi.fn().mockRejectedValue(new Error('Collection not found')),
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
