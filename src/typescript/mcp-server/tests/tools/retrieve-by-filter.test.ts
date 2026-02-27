/**
 * Tests for RetrieveTool - retrieve by filter
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

describe('RetrieveTool - retrieve by filter', () => {
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

  it('should retrieve documents using scroll', async () => {
    const result = await retrieveTool.retrieve({
      collection: 'projects',
      limit: 10,
    });

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
