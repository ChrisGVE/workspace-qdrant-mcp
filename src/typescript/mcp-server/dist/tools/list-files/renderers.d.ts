/**
 * Rendering functions for the list tool's three output formats.
 *
 * Provides tree, summary, and flat renderers that convert a FolderNode
 * tree (or flat file list) into a human-readable string representation.
 */
import type { TrackedFileEntry } from '../../clients/tracked-files-queries/index.js';
import type { FolderNode } from '../list-files-types.js';
export interface RenderResult {
    text: string;
    count: number;
}
export declare function renderTree(root: FolderNode, maxDepth: number, limit: number): RenderResult;
export declare function renderSummary(root: FolderNode, maxDepth: number, limit: number): RenderResult;
export declare function renderFlat(files: TrackedFileEntry[], limit: number): RenderResult;
//# sourceMappingURL=renderers.d.ts.map