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
 * Tenant resolution mirrors `search`/`grep`/`list`: an explicit `projectId`
 * wins; otherwise the caller's `cwd` is resolved to its project (so `graph`
 * operates on the SAME project as the other tools). It does NOT fall back to
 * "first active project" — that silently returned a different project's graph
 * when the cwd didn't match — it errors instead, asking for `projectId`/`cwd`.
 */

import { createHash } from 'node:crypto';

import type { DaemonClient } from '../clients/daemon-client.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { getEffectiveCwd } from '../utils/request-context.js';
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

async function resolveTenant(
  args: JsonObject,
  projectDetector: ProjectDetector
): Promise<string> {
  const explicit = str(args, 'projectId') ?? str(args, 'tenantId');
  if (explicit) return explicit;
  // Resolve the caller's cwd to its project exactly like `search`/`grep`/`list`
  // (`getEffectiveCwd()` honours the `cwd` arg / X-MCP-Host-Cwd header). This is
  // what keeps `graph` on the same project as the rest of the tools.
  // `fallbackToSoleProject` covers the single-project convenience case.
  const detected = await projectDetector.getProjectInfo(getEffectiveCwd(), false, {
    fallbackToSoleProject: true,
  });
  if (detected?.projectId) return detected.projectId;
  // Deliberately NO "first active project" fallback: with multiple projects and
  // an unresolvable cwd it picked an arbitrary (wrong) project and returned its
  // graph silently. Fail loudly instead.
  throw new Error(
    'Could not resolve a project for `graph`. Pass `projectId` (the tenant_id), ' +
      'or pass `cwd` (your absolute working directory) so the project can be ' +
      'auto-detected. (graph no longer guesses the first active project.)'
  );
}

export async function handleGraph(
  rawArgs: Record<string, unknown> | undefined,
  daemonClient: DaemonClient | undefined,
  projectDetector: ProjectDetector
): Promise<unknown> {
  if (!daemonClient) {
    throw new Error('graph requires a connected daemon client (gRPC unavailable)');
  }
  const args = rawArgs ?? {};
  const action = str(args, 'action') ?? 'stats';
  const tenant = await resolveTenant(args, projectDetector);
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
