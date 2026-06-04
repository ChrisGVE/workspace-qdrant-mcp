/**
 * Directory tree construction from flat file path lists.
 *
 * Builds a FolderNode tree from tracked file entries, handling
 * basePath stripping and submodule detection.
 */
/** Create a submodule lookup map keyed by path. */
function buildSubmoduleMap(submodules) {
    const map = new Map();
    for (const sm of submodules) {
        map.set(sm.submodulePath, sm);
    }
    return map;
}
/** Insert a single file entry into the folder tree. */
function insertFile(root, file, basePath, submoduleMap) {
    let relPath = file.relativePath;
    if (basePath && relPath.startsWith(basePath + '/')) {
        relPath = relPath.slice(basePath.length + 1);
    }
    const segments = relPath.split('/');
    const fileName = segments.pop();
    let current = root;
    let pathSoFar = basePath;
    for (const segment of segments) {
        pathSoFar = pathSoFar ? `${pathSoFar}/${segment}` : segment;
        if (!current.children.has(segment)) {
            const node = { name: segment, children: new Map(), files: [], totalFiles: 0 };
            const sm = submoduleMap.get(pathSoFar);
            if (sm)
                node.submodule = { repoName: sm.repoName };
            current.children.set(segment, node);
        }
        current = current.children.get(segment);
        if (current.submodule)
            break;
    }
    if (!current.submodule) {
        current.files.push({
            name: fileName,
            extension: file.extension,
            language: file.language,
            isTest: file.isTest,
        });
    }
}
/**
 * Build a folder tree from flat file paths.
 *
 * If basePath is set, strips it from relativePath before inserting.
 */
export function buildTree(files, submodules, basePath) {
    const root = { name: basePath || '.', children: new Map(), files: [], totalFiles: 0 };
    const submoduleMap = buildSubmoduleMap(submodules);
    for (const file of files) {
        insertFile(root, file, basePath, submoduleMap);
    }
    computeTotalFiles(root);
    return root;
}
function computeTotalFiles(node) {
    let total = node.files.length;
    for (const child of node.children.values()) {
        total += computeTotalFiles(child);
    }
    node.totalFiles = total;
    return total;
}
//# sourceMappingURL=tree-builder.js.map