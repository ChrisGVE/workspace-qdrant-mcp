/**
 * Construct a `QdrantClient` from MCP-server config fields.
 *
 * Four call sites (search, retrieve, rules, health-monitor) used to repeat
 * the same six-line boilerplate of building a `clientConfig` literal and
 * conditionally setting `apiKey`. This factory centralizes that.
 *
 * It does *not* memoize: a fresh `QdrantClient` is returned on every call
 * so per-test vitest mocks of the constructor keep working. Caching here
 * would silence three of the four "Api key is used with unsecure
 * connection." log lines per MCP session, but the test-isolation cost is
 * not worth the small log-noise win.
 */

import { QdrantClient } from '@qdrant/js-client-rest';

export interface QdrantClientOptions {
  /** Qdrant base URL, e.g. `http://qdrant:6333`. */
  url: string;
  /**
   * Optional API key. `undefined` is accepted (callers spread it from a
   * possibly-unset config field) and treated identically to a missing key.
   */
  apiKey?: string | undefined;
  /** Request timeout in milliseconds. Defaults to 5000 in this factory. */
  timeout?: number | undefined;
}

/** Build a `QdrantClient` from the connection options. */
export function getQdrantClient(opts: QdrantClientOptions): QdrantClient {
  const clientConfig: { url: string; apiKey?: string; timeout?: number } = {
    url: opts.url,
    timeout: opts.timeout ?? 5000,
  };
  if (opts.apiKey) {
    clientConfig.apiKey = opts.apiKey;
  }
  return new QdrantClient(clientConfig);
}
