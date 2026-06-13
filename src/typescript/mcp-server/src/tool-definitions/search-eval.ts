/**
 * MCP tool schema definition for the 'search_eval' tool.
 */

export const searchEvalToolDefinition = {
  name: 'search_eval',
  description:
    "Benchmark semantic-search quality. Runs known-item queries (each with the file(s) that SHOULD rank) through the live search pipeline and returns hit@1/3/10, recall@10, MRR, and duplicate-rate per mode (semantic/hybrid/exact) plus a quality verdict. Use it to measure the effect of a search-ranking change — the measure→edit→measure loop. Pass `cases` for an ad-hoc eval set, or omit to use the project's bundled dataset when available. Runs in-process against the real index (no extra setup).",
  inputSchema: {
    type: 'object' as const,
    properties: {
      cases: {
        type: 'array',
        description:
          'Ad-hoc evaluation set. Each case is a natural-language query plus the repo-relative file paths expected to rank for it. Omit to use the bundled dataset.',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Optional stable id (defaults to case-N).' },
            query: { type: 'string', description: 'The search query.' },
            expectedFiles: {
              type: 'array',
              items: { type: 'string' },
              description:
                'Repo-relative paths expected in the results (e.g. "src/tools/search.ts").',
            },
          },
          required: ['query', 'expectedFiles'],
        },
      },
      limit: { type: 'number', description: 'Results fetched per query (default: 10).' },
      topK: {
        type: 'number',
        description: 'Evaluation cutoff K for hit@k / recall (default: 10).',
      },
      projectId: {
        type: 'string',
        description: 'Tenant to evaluate against (default: auto-detect from cwd).',
      },
      cwd: {
        type: 'string',
        description:
          'Absolute working directory for project auto-detection over HTTP. Ignored when projectId is provided.',
      },
      scope: {
        type: 'string',
        enum: ['project', 'global', 'all'],
        description: 'Search scope (default: project).',
      },
      includeTopPaths: {
        type: 'boolean',
        description:
          'Include the returned file paths per query (semantic mode) for debugging misses (default: false).',
      },
      rerank: {
        type: 'boolean',
        description:
          'Force the cross-encoder rerank on/off for every query (default: deployment default, i.e. WQM_SEARCH_RERANK). Lets A/B sweeps run without redeploying.',
      },
      rerankWeight: {
        type: 'number',
        description:
          'Blend weight 0–1 for the rerank score: final pool order is (1-w)·norm(rrf_boosted) + w·norm(rerank). 1 = pure cross-encoder order; 0 = rerank off (default: WQM_SEARCH_RERANK_WEIGHT, else 0.05 — balanced BGE-M3 default after implementation-intent tuning; 0.10 maximizes top1/MRR).',
      },
    },
    required: [],
  },
};
