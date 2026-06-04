/**
 * Store tool argument builder — parse raw MCP tool arguments into StoreOptions
 */
// ── Validation ────────────────────────────────────────────────────────────
function validateStoreArgs(args) {
    const content = args?.['content'];
    if (!content) {
        throw new Error('Content is required for store operation');
    }
    const forProject = args?.['forProject'];
    const libraryName = args?.['libraryName'];
    if (!forProject && !libraryName) {
        throw new Error('libraryName is required - store tool is for libraries collection only. ' +
            'Use forProject: true to store to the current project\'s library.');
    }
    return content;
}
// ── Option extractors ─────────────────────────────────────────────────────
function extractTargetOptions(args, options, sessionState) {
    const libraryName = args?.['libraryName'];
    if (libraryName)
        options.libraryName = libraryName;
    const forProject = args?.['forProject'];
    if (forProject) {
        options.forProject = true;
        if (sessionState.projectId)
            options.projectId = sessionState.projectId;
    }
}
function extractMetadataOptions(args, options) {
    const title = args?.['title'];
    if (title)
        options.title = title;
    const url = args?.['url'];
    if (url)
        options.url = url;
    const filePath = args?.['filePath'];
    if (filePath)
        options.filePath = filePath;
    const sourceType = args?.['sourceType'];
    if (sourceType === 'user_input' ||
        sourceType === 'web' ||
        sourceType === 'file' ||
        sourceType === 'scratchbook' ||
        sourceType === 'note') {
        options.sourceType = sourceType;
    }
    const metadata = args?.['metadata'];
    if (metadata)
        options.metadata = metadata;
}
/**
 * Build store options from raw tool arguments.
 * Store tool is for libraries collection ONLY per spec.
 */
export function buildStoreOptions(args, sessionState) {
    const content = validateStoreArgs(args);
    const options = { content };
    extractTargetOptions(args, options, sessionState);
    extractMetadataOptions(args, options);
    return options;
}
//# sourceMappingURL=store.js.map