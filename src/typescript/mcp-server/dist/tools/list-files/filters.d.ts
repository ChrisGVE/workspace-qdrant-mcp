/**
 * File filtering utilities for the list tool.
 *
 * Provides glob-based filtering and folder counting helpers.
 */
import type { TrackedFileEntry } from '../../clients/tracked-files-queries/index.js';
import type { FolderNode } from '../list-files-types.js';
export declare function countFolders(node: FolderNode): number;
/**
 * Simple glob filter on relative paths.
 * Supports * (any non-/ chars) and ** (any path segment including /).
 */
export declare function filterByGlob(files: TrackedFileEntry[], pattern: string): TrackedFileEntry[];
export declare function globToRegex(pattern: string): RegExp;
//# sourceMappingURL=filters.d.ts.map