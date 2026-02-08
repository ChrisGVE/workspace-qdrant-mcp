// AUTO-GENERATED from assets/default_configuration.yaml — do not edit manually.
// Re-generate with: npx tsx scripts/generate-config-defaults.ts

import type { ServerConfig } from './config.js';

/**
 * Default configuration values extracted from the canonical YAML source.
 *
 * Fields that are MCP-server-specific (e.g. database.path) and have no
 * corresponding YAML entry use inline defaults.
 */
export const DEFAULT_CONFIG: ServerConfig = {
  database: {
    // Not in YAML — platform-specific, resolved at runtime by getStateDirectory()
    path: '~/.workspace-qdrant/state.db',
  },
  qdrant: {
    url: "http://localhost:6333",
    timeout: 30000,
  },
  daemon: {
    grpcPort: 50051,
    queuePollIntervalMs: 500,
    queueBatchSize: 10,
  },
  watching: {
    // Simplified glob subset of YAML watching.allowed_extensions for MCP clients
    patterns: ['*.py', '*.rs', '*.md', '*.js', '*.ts'],
    ignorePatterns: [
        "*.pyc",
        "*.class",
        "*.o",
        "*.obj",
        "*.lock",
        "*.min.js",
        "*.min.css",
        "*.map",
        "*.bundle.js",
        "*.chunk.js",
        "node_modules/*",
        "target/*",
        "build/*",
        "dist/*",
        "out/*",
        ".git/*",
        "__pycache__/*",
        ".venv/*",
        "venv/*",
        ".env/*",
        ".tox/*",
        ".mypy_cache/*",
        ".pytest_cache/*",
        ".ruff_cache/*",
        ".gradle/*",
        ".next/*",
        ".nuxt/*",
        ".svelte-kit/*",
        ".astro/*",
        "Pods/*",
        "DerivedData/*",
        ".build/*",
        ".swiftpm/*",
        ".fastembed_cache/*",
        ".terraform/*",
        ".terragrunt-cache/*",
        "coverage/*",
        ".nyc_output/*",
        ".cargo/*",
        ".rustup/*",
        "vendor/*",
        ".bundle/*",
        ".cache/*",
        ".tmp/*",
        "tmp/*",
        ".DS_Store/*",
        ".idea/*",
        ".vscode/*",
        ".settings/*",
        ".project/*",
        ".classpath/*",
        "bin/*",
        "obj/*",
        ".zig-cache/*",
        "zig-out/*",
        "elm-stuff/*",
        ".stack-work/*",
        "_build/*",
        "deps/*",
        ".dart_tool/*",
        ".pub-cache/*"
    ],
  },
  collections: {
    memoryCollectionName: "memory",
  },
  environment: {},
  memory: {
    limits: {
      maxLabelLength: 15,
      maxTitleLength: 50,
      maxTagLength: 20,
      maxTagsPerRule: 5,
    },
  },
};
