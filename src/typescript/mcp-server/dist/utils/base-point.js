/**
 * Base point identity computation for workspace-qdrant-mcp.
 *
 * Must produce identical output to the Rust implementation in
 * wqm-common/src/hashing.rs for the same inputs.
 */
import { createHash } from 'crypto';
/**
 * Normalize a file path for stable ID generation.
 * Uses forward slashes and strips trailing slashes.
 */
export function normalizePathForId(path) {
    return path.replace(/\\/g, '/').replace(/\/+$/, '');
}
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
export function computeBasePoint(tenantId, relativePath, fileHash) {
    const normalized = normalizePathForId(relativePath);
    const input = `${tenantId}|${normalized}|${fileHash}`;
    return createHash('sha256').update(input).digest('hex').slice(0, 32);
}
/**
 * Compute a Qdrant point ID from a base point and chunk index.
 *
 * Formula: SHA256(base_point|chunk_index)[:32]
 *
 * @param basePoint - the base point hash
 * @param chunkIndex - zero-based chunk index
 * @returns 32-char hex string
 */
export function computePointId(basePoint, chunkIndex) {
    const input = `${basePoint}|${chunkIndex}`;
    return createHash('sha256').update(input).digest('hex').slice(0, 32);
}
//# sourceMappingURL=base-point.js.map