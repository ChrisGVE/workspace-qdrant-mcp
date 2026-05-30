/**
 * `graph` MCP tool — code-relationship graph navigation.
 *
 * Surfaces the daemon's GraphService (edges built from tree-sitter symbol
 * relations during ingestion) to MCP clients. Read-only. Actions:
 *   - stats     : node/edge counts by type (GetGraphStats)
 *   - relations : callers/callees N hops from a symbol (QueryRelated)
 *   - impact    : what transitively depends on a symbol (ImpactAnalysis)
 *   - hotspots  : most central symbols by PageRank (ComputePageRank)
 *   - modules   : code communities/clusters (DetectCommunities)
 *
 * Tenant resolution mirrors `workspace_index indexing_status`: an explicit
 * `projectId` wins; otherwise the first active project from the daemon is used
 * (works inside the dockerized MCP where cwd-based detection isn't available).
 */

import { createHash } from 'node:crypto';

import type { DaemonClient } from '../clients/daemon-client.js';
import type {
  ImpactAnalysisRequest,
  PageRankRequest,
  CommunityRequest,
  BetweennessRequest,
  QueryRelatedRequest,
} from '../clients/grpc-types.js';

type JsonObject = Record<string, unknown>;

function str(args: JsonObject, key: string): string | undefined {
  const v = args[key];
  return typeof v === 'string' && v.trim().length > 0 ? v : undefined;
}

function num(args: JsonObject, key: string): number | undefined {
  const v = args[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function strArray(args: JsonObject, key: string): string[] | undefined {
  const v = args[key];
  if (Array.isArray(v)) {
    const out = v.filter((x): x is string => typeof x === 'string');
    return out.length > 0 ? out : undefined;
  }
  return undefined;
}

/**
 * SHA256(tenant_id|file_path|symbol_name|symbol_type)[..32 hex chars].
 * Must match Rust's `compute_node_id` so QueryRelated finds the node.
 */
function computeNodeId(
  tenantId: string,
  filePath: string,
  symbolName: string,
  symbolType: string
): string {
  return createHash('sha256')
    .update(`${tenantId}|${filePath}|${symbolName}|${symbolType}`)
    .digest('hex')
    .slice(0, 32);
}

async function resolveTenant(args: JsonObject, daemonClient: DaemonClient): Promise<string> {
  const explicit = str(args, 'projectId') ?? str(args, 'tenantId');
  if (explicit) return explicit;
  const projects = await daemonClient.listProjects({ active_only: true });
  const first = projects.projects[0];
  if (!first) {
    throw new Error(
      'No active project found. Pass `projectId` (the tenant_id) explicitly, or register/activate a project first.'
    );
  }
  return first.project_id;
}

export async function handleGraph(
  rawArgs: Record<string, unknown> | undefined,
  daemonClient: DaemonClient | undefined
): Promise<unknown> {
  if (!daemonClient) {
    throw new Error('graph requires a connected daemon client (gRPC unavailable)');
  }
  const args = rawArgs ?? {};
  const action = str(args, 'action') ?? 'stats';
  const tenant = await resolveTenant(args, daemonClient);
  const edgeTypes = strArray(args, 'edgeTypes');

  switch (action) {
    case 'stats': {
      const r = await daemonClient.getGraphStats({ tenant_id: tenant });
      return { success: true, action, tenant_id: tenant, ...r };
    }

    case 'impact':
    case 'usages': {
      // Both wrap ImpactAnalysis (reverse reachability over the graph):
      //   impact → "blast radius if I change X" (what breaks)
      //   usages → "where/by what is X used" (find usages)
      // Same data, framed for the two questions. Precision improves once the
      // LSP call-hierarchy pass resolves CALLS edges (see daemon graph_ingest).
      const symbol = str(args, 'symbol');
      if (!symbol) throw new Error(`graph action '${action}' requires \`symbol\``);
      const filePath = str(args, 'filePath');
      const req: ImpactAnalysisRequest = {
        tenant_id: tenant,
        symbol_name: symbol,
        ...(filePath ? { file_path: filePath } : {}),
      };
      const r = await daemonClient.impactAnalysis(req);
      return { success: true, action, tenant_id: tenant, symbol, ...r };
    }

    case 'hotspots': {
      const req: PageRankRequest = {
        tenant_id: tenant,
        top_k: num(args, 'topK') ?? 20,
        ...(edgeTypes ? { edge_types: edgeTypes } : {}),
      };
      const r = await daemonClient.computePageRank(req);
      return { success: true, action, tenant_id: tenant, ...r };
    }

    case 'bridges': {
      // Betweenness centrality — symbols that sit on many shortest paths
      // ("bridges"/bottlenecks connecting otherwise-separate clusters).
      const maxSamples = num(args, 'maxSamples');
      const req: BetweennessRequest = {
        tenant_id: tenant,
        top_k: num(args, 'topK') ?? 20,
        ...(maxSamples !== undefined ? { max_samples: maxSamples } : {}),
        ...(edgeTypes ? { edge_types: edgeTypes } : {}),
      };
      const r = await daemonClient.computeBetweenness(req);
      return { success: true, action, tenant_id: tenant, ...r };
    }

    case 'modules': {
      const minSize = num(args, 'minSize');
      const req: CommunityRequest = {
        tenant_id: tenant,
        ...(minSize !== undefined ? { min_community_size: minSize } : {}),
        ...(edgeTypes ? { edge_types: edgeTypes } : {}),
      };
      const r = await daemonClient.detectCommunities(req);
      return { success: true, action, tenant_id: tenant, ...r };
    }

    case 'relations': {
      const symbol = str(args, 'symbol');
      const filePath = str(args, 'filePath');
      if (!symbol || !filePath) {
        throw new Error(
          "graph action 'relations' requires `symbol` and `filePath` " +
            "(plus optional `symbolType`, default 'function'). Get these from a `search` result's metadata."
        );
      }
      const symbolType = str(args, 'symbolType') ?? 'function';
      const nodeId = computeNodeId(tenant, filePath, symbol, symbolType);
      const req: QueryRelatedRequest = {
        tenant_id: tenant,
        node_id: nodeId,
        max_hops: num(args, 'maxHops') ?? 1,
        ...(edgeTypes ? { edge_types: edgeTypes } : {}),
      };
      const r = await daemonClient.queryRelated(req);
      return { success: true, action, tenant_id: tenant, symbol, node_id: nodeId, ...r };
    }

    default:
      throw new Error(
        `Unknown graph action: '${action}'. Use one of: stats, relations, impact, usages, hotspots, bridges, modules.`
      );
  }
}
