/**
 * DaemonEmbedder - Embedding generation via daemon service
 *
 * Provides a clean API for embedding generation that delegates to the daemon's
 * EmbeddingService via gRPC. This ensures consistency with the Python implementation
 * and uses the same FastEmbed model (all-MiniLM-L6-v2) across all components.
 *
 * Features:
 * - Dense embeddings via FastEmbed (384 dimensions)
 * - Sparse vectors via BM25 for keyword search
 * - Caching handled by daemon (LRU cache)
 * - Graceful degradation when daemon unavailable
 */
// Expected embedding dimensions for all-MiniLM-L6-v2
const EXPECTED_DIMENSIONS = 384;
/**
 * DaemonEmbedder - Generates embeddings via daemon's EmbeddingService
 *
 * Usage:
 * ```typescript
 * const embedder = new DaemonEmbedder(daemonClient);
 *
 * // Dense embedding
 * const dense = await embedder.generateEmbedding("search query");
 * console.log(dense.embedding); // 384-dimensional vector
 *
 * // Sparse vector for keyword search
 * const sparse = await embedder.generateSparseVector("search query");
 * console.log(sparse.indices, sparse.values);
 *
 * // Both for hybrid search
 * const { dense, sparse } = await embedder.generateHybridVectors("search query");
 * ```
 */
export class DaemonEmbedder {
    daemonClient;
    _config;
    constructor(daemonClient, config = {}) {
        this.daemonClient = daemonClient;
        this._config = {
            retryOnFailure: config.retryOnFailure ?? true,
            maxRetries: config.maxRetries ?? 2,
        };
    }
    /**
     * Get current configuration
     */
    get config() {
        return this._config;
    }
    /**
     * Generate dense embedding for text
     *
     * Uses FastEmbed all-MiniLM-L6-v2 model via daemon
     * Returns 384-dimensional normalized vector
     *
     * @throws Error if daemon unavailable or embedding generation fails
     */
    async generateEmbedding(text) {
        if (!text.trim()) {
            throw new Error('Text cannot be empty');
        }
        const response = await this.daemonClient.embedText({ text });
        if (!response.success) {
            throw new Error(response.error_message ?? 'Embedding generation failed');
        }
        // Validate dimensions
        if (response.dimensions !== EXPECTED_DIMENSIONS) {
            console.warn(`Unexpected embedding dimensions: expected ${EXPECTED_DIMENSIONS}, got ${response.dimensions}`);
        }
        return {
            embedding: response.embedding,
            dimensions: response.dimensions,
            model: response.model_name,
        };
    }
    /**
     * Generate sparse vector for text
     *
     * Uses BM25-style tokenization via daemon
     * Returns index-value pairs for keyword search
     *
     * @throws Error if daemon unavailable or generation fails
     */
    async generateSparseVector(text) {
        if (!text.trim()) {
            return { indices: [], values: [], vocabSize: 0 };
        }
        const response = await this.daemonClient.generateSparseVector({ text });
        if (!response.success) {
            throw new Error(response.error_message ?? 'Sparse vector generation failed');
        }
        // Convert Record<number, number> to parallel arrays
        const entries = Object.entries(response.indices_values);
        const indices = entries.map(([k]) => Number(k));
        const values = entries.map(([, v]) => v);
        return {
            indices,
            values,
            vocabSize: response.vocab_size,
        };
    }
    /**
     * Generate both dense and sparse vectors for hybrid search
     *
     * Convenience method that generates both vector types in parallel
     */
    async generateHybridVectors(text) {
        const [dense, sparse] = await Promise.all([
            this.generateEmbedding(text),
            this.generateSparseVector(text),
        ]);
        return { dense, sparse };
    }
    /**
     * Check if embedder is ready (daemon connected)
     */
    isReady() {
        return this.daemonClient.isConnected();
    }
    /**
     * Get embedding model info
     */
    getModelInfo() {
        return {
            name: 'all-MiniLM-L6-v2',
            dimensions: EXPECTED_DIMENSIONS,
        };
    }
}
//# sourceMappingURL=daemon-embedder.js.map