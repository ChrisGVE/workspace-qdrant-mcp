# Dockerfile for workspace-qdrant-mcp MCP server (TypeScript)
#
# The MCP server communicates via stdin/stdout (stdio transport).
# It connects to an external Qdrant instance for vector storage.
#
# Required: QDRANT_URL (default: http://host.docker.internal:6333)
# Optional: QDRANT_API_KEY (for Qdrant Cloud)
#
# Usage:
#   docker build -t workspace-qdrant-mcp .
#   docker run -i -e QDRANT_URL=http://host.docker.internal:6333 workspace-qdrant-mcp

# ---- Build Stage ----
FROM node:20-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Assets needed by generate-config-defaults.ts
# (resolves PROJECT_ROOT as 4 levels up from src/typescript/mcp-server/scripts/)
COPY assets/ assets/

# Pre-built native .node addon (wqm-common-node Rust/napi-rs bridge, Linux x64 glibc)
COPY src/rust/common-node/index.js src/rust/common-node/index.js
COPY src/rust/common-node/index.d.ts src/rust/common-node/index.d.ts
ARG VERSION=v0.0.1
RUN curl -fsSL \
    "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/${VERSION}/wqm-common-node.linux-x64-gnu.node" \
    -o src/rust/common-node/wqm-common-node.linux-x64-gnu.node

# Install dependencies (cached layer)
COPY src/typescript/mcp-server/package*.json src/typescript/mcp-server/
RUN cd src/typescript/mcp-server && npm ci

# Copy source and build TypeScript
COPY src/typescript/mcp-server/ src/typescript/mcp-server/
RUN cd src/typescript/mcp-server && npm run build

# Prune dev dependencies
RUN cd src/typescript/mcp-server && npm prune --omit=dev

# ---- Production Stage ----
FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Native .node addon for production (Linux x64 glibc)
COPY src/rust/common-node/index.js src/rust/common-node/index.js
COPY src/rust/common-node/index.d.ts src/rust/common-node/index.d.ts
ARG VERSION=v0.0.1
RUN curl -fsSL \
    "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/${VERSION}/wqm-common-node.linux-x64-gnu.node" \
    -o src/rust/common-node/wqm-common-node.linux-x64-gnu.node \
    && apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy compiled TypeScript output and production node_modules
COPY --from=builder /build/src/typescript/mcp-server/dist src/typescript/mcp-server/dist/
COPY --from=builder /build/src/typescript/mcp-server/node_modules src/typescript/mcp-server/node_modules/
COPY --from=builder /build/src/typescript/mcp-server/package.json src/typescript/mcp-server/package.json

# Run as non-root
RUN groupadd --system --gid 65534 mcpuser \
    && useradd --system --uid 65534 --gid 65534 --no-create-home mcpuser \
    && chown -R mcpuser:mcpuser /app
USER mcpuser

ENV NODE_ENV=production
ENV WQM_LOG_LEVEL=info

LABEL org.opencontainers.image.title="workspace-qdrant-mcp"
LABEL org.opencontainers.image.description="Project-scoped semantic workspace memory MCP server with Qdrant hybrid search"
LABEL org.opencontainers.image.source="https://github.com/ChrisGVE/workspace-qdrant-mcp"
LABEL org.opencontainers.image.licenses="MIT"

# MCP stdio transport: the server reads from stdin, writes to stdout
WORKDIR /app/src/typescript/mcp-server
ENTRYPOINT ["node", "dist/index.js"]
