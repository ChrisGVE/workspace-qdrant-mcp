/**
 * Regression tests for the unknown-argument refusal in `retrieve`.
 *
 * The builder used to drop unrecognized argument names silently, so a
 * mis-shaped call like `retrieve({ query: "select schema" })` degraded
 * into an empty call and surfaced as an unrelated unresolved-scope error.
 * The contract under test:
 * - `buildRetrieveOptions` flags unknown argument names in `unknownArgs`
 *   (the set of known names is derived from the published input schema,
 *   so `cwd` — consumed by the transport layer — is never flagged).
 * - `RetrieveTool.retrieve` refuses such calls with an explanatory
 *   message BEFORE any scope resolution or Qdrant read, hinting at the
 *   `search` tool when the stray argument is `query`.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RetrieveTool } from '../../src/tools/retrieve.js';
import { buildRetrieveOptions } from '../../src/tool-builders/retrieve.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Mock the Qdrant client so we can override its impl per test.
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    retrieve: vi.fn().mockResolvedValue([]),
    scroll: vi.fn().mockResolvedValue({ points: [] }),
  })),
}));

async function setupMockClient() {
  const QdrantClientMock = await import('@qdrant/js-client-rest');
  const retrieveFn = vi.fn().mockResolvedValue([]);
  const scrollFn = vi.fn().mockResolvedValue({ points: [] });
  vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
    () =>
      ({
        retrieve: retrieveFn,
        scroll: scrollFn,
      }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
  );
  return { retrieveFn, scrollFn };
}

function detectorReturning(projectId: string | undefined): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/test/project'),
    getProjectInfo: vi.fn().mockResolvedValue(projectId ? { projectId } : null),
  } as unknown as ProjectDetector;
}

describe('buildRetrieveOptions — unknown argument detection', () => {
  it('flags an unknown argument name', () => {
    const options = buildRetrieveOptions({ query: 'select schema' });
    expect(options.unknownArgs).toEqual(['query']);
  });

  it('does not set unknownArgs when every argument is known', () => {
    const options = buildRetrieveOptions({
      documentId: 'doc-1',
      collection: 'projects',
      filter: { document_id: 'abc' },
      limit: 5,
      offset: 0,
      projectId: 'project-a',
      cwd: '/home/user/project',
      libraryName: 'lib',
    });
    expect('unknownArgs' in options).toBe(false);
  });

  it('does not set unknownArgs for undefined args', () => {
    const options = buildRetrieveOptions(undefined);
    expect('unknownArgs' in options).toBe(false);
  });

  it('keeps known arguments while flagging the unknown ones', () => {
    const options = buildRetrieveOptions({ documentId: 'doc-1', foo: 1, bar: true });
    expect(options.documentId).toBe('doc-1');
    expect(options.unknownArgs).toEqual(['foo', 'bar']);
  });
});

describe('RetrieveTool — unknown argument refusal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('refuses a `query` call with a search-tool hint, without touching Qdrant or scope', async () => {
    const { retrieveFn, scrollFn } = await setupMockClient();
    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve(buildRetrieveOptions({ query: 'select schema' }));

    expect(result.success).toBe(false);
    expect(result.documents).toHaveLength(0);
    expect(result.message).toContain('Unknown retrieve parameter(s): query');
    expect(result.message).toContain('use the `search` tool');
    expect(result.message).toContain('documentId');
    expect(retrieveFn).not.toHaveBeenCalled();
    expect(scrollFn).not.toHaveBeenCalled();
    expect(detector.getProjectInfo).not.toHaveBeenCalled();
  });

  it('lists the stray names without the search hint when `query` is not among them', async () => {
    const { retrieveFn, scrollFn } = await setupMockClient();
    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve(buildRetrieveOptions({ documentID: 'doc-1' }));

    expect(result.success).toBe(false);
    expect(result.message).toContain('Unknown retrieve parameter(s): documentID');
    expect(result.message).not.toContain('use the `search` tool');
    expect(retrieveFn).not.toHaveBeenCalled();
    expect(scrollFn).not.toHaveBeenCalled();
  });
});
