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
export function buildTree(
  files: TrackedFileEntry[],
  submodules: SubmoduleEntry[],
  basePath: string,
): FolderNode {
  const root: FolderNode = {
    name: basePath || '.',
    children: new Map(),
    files: [],
    totalFiles: 0,
  };

  // Build set of submodule paths for fast lookup
  const submoduleMap = new Map<string, SubmoduleEntry>();
  for (const sm of submodules) {
    submoduleMap.set(sm.submodulePath, sm);
  }

  for (const file of files) {
    let relPath = file.relativePath;

    // Strip basePath prefix
    if (basePath && relPath.startsWith(basePath + '/')) {
      relPath = relPath.slice(basePath.length + 1);
    }

    const segments = relPath.split('/');
    const fileName = segments.pop()!;
    let current = root;

    // Walk/create folder path
    let pathSoFar = basePath;
    for (const segment of segments) {
      pathSoFar = pathSoFar ? `${pathSoFar}/${segment}` : segment;

      if (!current.children.has(segment)) {
        const node: FolderNode = {
          name: segment,
          children: new Map(),
          files: [],
          totalFiles: 0,
        };

        // Check if this folder is a submodule root
        const sm = submoduleMap.get(pathSoFar);
        if (sm) {
          node.submodule = { repoName: sm.repoName };
        }

        current.children.set(segment, node);
      }

      current = current.children.get(segment)!;

      // If we hit a submodule, don't go deeper
      if (current.submodule) break;
    }

    // Only add the file if we didn't stop at a submodule
    if (!current.submodule) {
      current.files.push({
        name: fileName,
        extension: file.extension,
        language: file.language,
        isTest: file.isTest,
      });
    }
  }

  // Compute totalFiles bottom-up
  computeTotalFiles(root);

  return root;
}

function computeTotalFiles(node: FolderNode): number {
  let total = node.files.length;
  for (const child of node.children.values()) {
    total += computeTotalFiles(child);
  }
  node.totalFiles = total;
  return total;
}
