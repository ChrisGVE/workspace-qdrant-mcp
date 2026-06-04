/**
 * Retrieve tool argument builder — parse raw MCP tool arguments into RetrieveOptions
 */
/** Build retrieve options from raw tool arguments */
export function buildRetrieveOptions(args) {
    const options = {};
    const documentId = args?.['documentId'];
    if (documentId)
        options.documentId = documentId;
    const collection = args?.['collection'];
    if (collection === 'projects' ||
        collection === 'libraries' ||
        collection === 'rules' ||
        collection === 'scratchpad') {
        options.collection = collection;
    }
    const filter = args?.['filter'];
    if (filter)
        options.filter = filter;
    const limit = args?.['limit'];
    if (limit !== undefined)
        options.limit = limit;
    const offset = args?.['offset'];
    if (offset !== undefined)
        options.offset = offset;
    const projectId = args?.['projectId'];
    if (projectId)
        options.projectId = projectId;
    const libraryName = args?.['libraryName'];
    if (libraryName)
        options.libraryName = libraryName;
    return options;
}
//# sourceMappingURL=retrieve.js.map