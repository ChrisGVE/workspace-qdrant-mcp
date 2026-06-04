/**
 * Directory tree construction from flat file path lists.
 *
 * Builds a FolderNode tree from tracked file entries, handling
 * basePath stripping and submodule detection.
 */
import type { TrackedFileEntry, SubmoduleEntry } from '../../clients/tracked-files-queries/index.js';
import type { FolderNode } from '../list-files-types.js';
/**
 * Build a folder tree from flat file paths.
 *
 * If basePath is set, strips it from relativePath before inserting.
 */
export declare function buildTree(files: TrackedFileEntry[], submodules: SubmoduleEntry[], basePath: string): FolderNode;
//# sourceMappingURL=tree-builder.d.ts.map