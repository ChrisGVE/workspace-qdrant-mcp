/**
 * MCP tool schema definition for the 'graph' tool — code-relationship
 * graph navigation backed by the daemon's GraphService.
 */

export const graphToolDefinition = {
  name: 'graph',
  description:
    'Navigate the code-relationship graph: callers/callees, change-impact, importance ranking, and module clusters. ' +
    'Built from symbol relations (calls, contains, uses-type, imports) extracted during indexing. ' +
    'Use this to understand how code connects before editing — e.g. "what calls this function?", "what breaks if I change X?", "what are the most central functions?".',
  inputSchema: {
    type: 'object' as const,
    properties: {
      action: {
        type: 'string',
        enum: ['stats', 'relations', 'impact', 'usages', 'hotspots', 'bridges', 'modules'],
        description:
          "stats: node/edge counts. relations: callers/callees of a symbol. impact: change blast-radius (what breaks if you change a symbol). usages: where/by what a symbol is used (find usages). hotspots: most central symbols (PageRank). bridges: bottleneck symbols on many shortest paths (betweenness). modules: code clusters. Default: 'stats'.",
      },
      symbol: {
        type: 'string',
        description: "Symbol name. Required for 'impact' and 'relations'.",
      },
      filePath: {
        type: 'string',
        description:
          "Relative file path of the symbol's definition. Required for 'relations'; optional narrowing for 'impact'.",
      },
      symbolType: {
        type: 'string',
        description:
          "Symbol kind for 'relations' node lookup (function, class, struct, method, …). Default: 'function'.",
      },
      maxHops: {
        type: 'number',
        description: "Traversal depth for 'relations' (1-5, default 1).",
      },
      topK: {
        type: 'number',
        description: "Number of top results for 'hotspots' and 'bridges' (default 20).",
      },
      maxSamples: {
        type: 'number',
        description:
          "For 'bridges': sample N source nodes for betweenness on large graphs (0/omit = exact).",
      },
      minSize: {
        type: 'number',
        description: "Minimum community size for 'modules' (default 2).",
      },
      edgeTypes: {
        type: 'array',
        items: { type: 'string' },
        description:
          'Filter by edge type (e.g. ["CALLS","IMPORTS","CONTAINS","USES_TYPE"]). Empty/omitted = all edge types.',
      },
      projectId: {
        type: 'string',
        description:
          'Project tenant_id. Takes precedence over cwd. If both are omitted the project is auto-detected from cwd; graph errors rather than guessing.',
      },
      cwd: {
        type: 'string',
        description:
          'Absolute path of your current working directory. Pass this so the server can auto-detect the project over HTTP (same as search/grep/list). Ignored when projectId is provided.',
      },
    },
    required: ['action'],
  },
};
