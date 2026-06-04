/**
 * Base point identity computation for workspace-qdrant-mcp.
 *
 * Must produce identical output to the Rust implementation in
 * wqm-common/src/hashing.rs for the same inputs.
 */
/**
 * Normalize a file path for stable ID generation.
 * Uses forward slashes and strips trailing slashes.
 */
export declare function normalizePathForId(path: string): string;
/**
 * Compute the base point hash: SHA256(tenant_id|relative_path|file_hash)[:32]
 *
 * The base point uniquely identifies a specific VERSION of a specific file,
 * independent of branch. Identical content at the same path shares one
 * base_point across all branches, enabling content-hash dedup.
 *
 * @param tenantId - derived from git remote URL hash (git) or path hash (non-git)
 * @param relativePath - file path relative to project root (normalized)
 * @param fileHash - SHA256 of file content or git blob SHA
 * @returns 32-char hex string
 */
export declare function computeBasePoint(tenantId: string, relativePath: string, fileHash: string): string;
/**
 * Compute a Qdrant point ID from a base point and chunk index.
 *
 * Formula: SHA256(base_point|chunk_index)[:32]
 *
 * @param basePoint - the base point hash
 * @param chunkIndex - zero-based chunk index
 * @returns 32-char hex string
 */
export declare function computePointId(basePoint: string, chunkIndex: number): string;
//# sourceMappingURL=base-point.d.ts.map