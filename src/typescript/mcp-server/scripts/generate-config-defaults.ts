#!/usr/bin/env tsx
/**
 * Build script: Generate TypeScript DEFAULT_CONFIG from canonical YAML source.
 *
 * Reads assets/default_configuration.yaml and produces
 * src/types/generated-defaults.ts with a type-checked DEFAULT_CONFIG constant.
 *
 * Run: npx tsx scripts/generate-config-defaults.ts
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse as parseYaml } from 'yaml';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..', '..', '..', '..');
const YAML_PATH = resolve(PROJECT_ROOT, 'assets', 'default_configuration.yaml');
const OUTPUT_PATH = resolve(__dirname, '..', 'src', 'types', 'generated-defaults.ts');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Parse a YAML duration string (e.g. "30s", "5m", "1h") into milliseconds. */
function durationToMs(value: string | number): number {
  if (typeof value === 'number') return value;
  const match = value.match(/^(\d+(?:\.\d+)?)\s*(ms|s|m|h)$/);
  if (!match) throw new Error(`Cannot parse duration: "${value}"`);
  const num = parseFloat(match[1]!);
  switch (match[2]) {
    case 'ms':
      return num;
    case 's':
      return num * 1_000;
    case 'm':
      return num * 60_000;
    case 'h':
      return num * 3_600_000;
    default:
      throw new Error(`Unknown unit in duration: "${value}"`);
  }
}

/** Safely read a nested YAML value. */
function get(obj: Record<string, unknown>, path: string): unknown {
  let cur: unknown = obj;
  for (const key of path.split('.')) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[key];
  }
  return cur;
}

/** Convert directory names to glob ignore patterns (e.g. "node_modules" → "node_modules/*"). */
function dirToGlob(dir: string): string {
  return `${dir}/*`;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const yamlContent = readFileSync(YAML_PATH, 'utf-8');
const yaml = parseYaml(yamlContent) as Record<string, unknown>;

// --- Extract values with validation ---

const qdrantUrl = get(yaml, 'qdrant.url') as string | undefined;
const qdrantTimeout = get(yaml, 'qdrant.timeout') as string | number | undefined;
const grpcPort = get(yaml, 'grpc.port') as number | undefined;
const pollIntervalMs = get(yaml, 'queue_processor.poll_interval_ms') as number | undefined;
const batchSize = get(yaml, 'queue_processor.batch_size') as number | undefined;
const excludeDirs = get(yaml, 'watching.exclude_directories') as string[] | undefined;
const excludePatterns = get(yaml, 'watching.exclude_patterns') as string[] | undefined;

// Build ignore patterns from exclude_directories + exclude_patterns
const ignorePatterns: string[] = [
  ...(excludePatterns ?? []),
  ...(excludeDirs ?? []).map(dirToGlob),
];

// Validate required fields sourced from YAML
const errors: string[] = [];
if (qdrantUrl === undefined) errors.push('qdrant.url');
if (qdrantTimeout === undefined) errors.push('qdrant.timeout');
if (grpcPort === undefined) errors.push('grpc.port');
if (pollIntervalMs === undefined) errors.push('queue_processor.poll_interval_ms');
if (batchSize === undefined) errors.push('queue_processor.batch_size');

if (errors.length > 0) {
  console.error(`Missing required YAML fields: ${errors.join(', ')}`);
  process.exit(1);
}

// MCP-server-specific defaults not present in the shared daemon YAML.
// These are owned by the TypeScript server layer only.
const rulesCollName = 'rules';
const maxLabelLen = 15;
const maxTitleLen = 50;
const maxTagLen = 20;
const maxTagsPerRule = 5;

// --- Build the generated file content ---

const timeoutMs = durationToMs(qdrantTimeout!);

const output = `// AUTO-GENERATED from assets/default_configuration.yaml — do not edit manually.
// Re-generate with: npx tsx scripts/generate-config-defaults.ts

import type { ServerConfig } from './config.js';
import { getDatabasePath } from '../utils/paths.js';

/**
 * Default configuration values extracted from the canonical YAML source.
 *
 * Fields that are MCP-server-specific (e.g. database.path) and have no
 * corresponding YAML entry use inline defaults.
 */
export const DEFAULT_CONFIG: ServerConfig = {
  database: {
    path: getDatabasePath(),
  },
  qdrant: {
    url: ${JSON.stringify(qdrantUrl)},
    timeout: ${timeoutMs},
  },
  daemon: {
    grpcPort: ${grpcPort},
    queuePollIntervalMs: ${pollIntervalMs},
    queueBatchSize: ${batchSize},
  },
  watching: {
    // Simplified glob subset of YAML watching.allowed_extensions for MCP clients
    patterns: ['*.py', '*.rs', '*.md', '*.js', '*.ts'],
    ignorePatterns: ${JSON.stringify(ignorePatterns, null, 4).replace(/\n/g, '\n    ')},
  },
  collections: {
    rulesCollectionName: ${JSON.stringify(rulesCollName)},
  },
  environment: {},
  rules: {
    limits: {
      maxLabelLength: ${maxLabelLen},
      maxTitleLength: ${maxTitleLen},
      maxTagLength: ${maxTagLen},
      maxTagsPerRule: ${maxTagsPerRule},
    },
  },
};
`;

writeFileSync(OUTPUT_PATH, output, 'utf-8');
console.log(`Generated ${OUTPUT_PATH}`);
