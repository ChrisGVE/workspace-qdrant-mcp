#!/usr/bin/env node

import { randomUUID } from 'node:crypto';
import { createHash } from 'node:crypto';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';

import { Client } from '../../src/typescript/mcp-server/node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js';
import { StreamableHTTPClientTransport } from '../../src/typescript/mcp-server/node_modules/@modelcontextprotocol/sdk/dist/esm/client/streamableHttp.js';

const DEFAULT_ATTEMPTS = 30;
const DEFAULT_POLL_SECONDS = 2;
const LIBRARY_QUERY = 'How should the system recover after credentials stop working?';
const PROJECT_QUERY = 'How should the application recover when credentials expire?';

function parseArgs(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;

    const raw = token.slice(2);
    const eqIndex = raw.indexOf('=');
    if (eqIndex >= 0) {
      const key = raw.slice(0, eqIndex);
      const value = raw.slice(eqIndex + 1);
      options[key] = value;
      continue;
    }

    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      options[raw] = 'true';
      continue;
    }

    options[raw] = next;
    i += 1;
  }
  return options;
}

function asNumber(value, fallback) {
  if (value === undefined || value === '') return fallback;
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function getTextContent(toolResult) {
  const text = (toolResult?.content ?? [])
    .filter((item) => item && item.type === 'text' && typeof item.text === 'string')
    .map((item) => item.text)
    .join('\n');
  return text.trim();
}

function parseJsonToolResult(toolResult) {
  const text = getTextContent(toolResult);
  if (!text) {
    throw new Error('Tool result did not contain any text content.');
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`Expected JSON tool output, got: ${text}`);
  }
}

function computeProjectId(projectDir) {
  const canonical = String(projectDir).replace(/\\/g, '/');
  const hash = createHash('sha256').update(canonical, 'utf8').digest('hex');
  return `local_${hash.slice(0, 12)}`;
}

function convertToWqmPath(pathValue) {
  if (!pathValue) return pathValue;

  const raw = String(pathValue).replace(/\\/g, '/');

  if (/^\/mnt\/[a-z]\//i.test(raw)) {
    return raw;
  }

  if (raw.startsWith('/')) {
    return raw;
  }

  const driveMatch = raw.match(/^([A-Za-z]):\/(.*)$/);
  if (driveMatch) {
    const drive = driveMatch[1].toLowerCase();
    const rest = driveMatch[2].replace(/^\/+/, '');
    return `/mnt/${drive}/${rest}`;
  }

  const absolute = resolve(pathValue).replace(/\\/g, '/');
  const absoluteDriveMatch = absolute.match(/^([A-Za-z]):\/(.*)$/);
  if (absoluteDriveMatch) {
    const drive = absoluteDriveMatch[1].toLowerCase();
    const rest = absoluteDriveMatch[2].replace(/^\/+/, '');
    return `/mnt/${drive}/${rest}`;
  }

  return absolute;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHealth(mcpUrl, attempts, pollSeconds) {
  const healthUrl = new URL(mcpUrl);
  healthUrl.pathname = healthUrl.pathname.replace(/\/mcp$/, '/healthz');
  for (let i = 0; i < attempts; i += 1) {
    try {
      const response = await fetch(healthUrl, { method: 'GET' });
      if (response.ok) return;
    } catch {
      // Keep polling.
    }
    await sleep(pollSeconds * 1000);
  }
  throw new Error(`MCP health check did not return 200 after ${attempts} attempts: ${healthUrl}`);
}

async function connectClient(mcpUrl, token) {
  const transport = new StreamableHTTPClientTransport(new URL(mcpUrl), {
    requestInit: {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    },
  });
  const client = new Client({ name: 'semantic-smoke-client', version: '0.1.0' });
  await client.connect(transport);
  return { client, transport };
}

function ensureProjectFiles(projectDir) {
  rmSync(projectDir, { recursive: true, force: true });
  mkdirSync(join(projectDir, 'src'), { recursive: true });

  const readme = [
    '# Session recovery',
    '',
    'When credentials stop working, the system should refresh the session, retry with exponential backoff, and only ask for interactive sign-in after the retry budget has been exhausted.',
  ].join('\n');

  const recoveryPy = [
    'def recover_session():',
    '    """Recover access after the session has gone stale."""',
    '    return "retry-and-reauth"',
  ].join('\n');

  writeFileSync(join(projectDir, 'README.md'), `${readme}\n`, 'utf8');
  writeFileSync(join(projectDir, 'src', 'recovery.py'), `${recoveryPy}\n`, 'utf8');

  let gitInit = spawnSync('git', ['-C', projectDir, 'init', '-q', '-b', 'main'], {
    stdio: 'inherit',
  });
  if (gitInit.status !== 0) {
    gitInit = spawnSync('git', ['-C', projectDir, 'init', '-q'], {
      stdio: 'inherit',
    });
    if (gitInit.status !== 0) {
      throw new Error(`git init failed with exit code ${gitInit.status ?? 1}`);
    }
  }
}

function buildProjectDir(repoDir, providedDir) {
  if (providedDir && providedDir.trim()) {
    return resolve(providedDir);
  }

  return resolve(
    repoDir,
    '.wqm-fork',
    'semantic-smoke',
    `project-${randomUUID().replace(/-/g, '')}`
  );
}

async function runLibrarySmoke({
  mcpUrl,
  token,
  libraryName,
  attempts,
  pollSeconds,
}) {
  await waitForHealth(mcpUrl, attempts, pollSeconds);
  const { client, transport } = await connectClient(mcpUrl, token);
  try {
    const storeResult = await client.callTool({
      name: 'store',
      arguments: {
        content:
          'When a session becomes stale, refresh the identity, retry with exponential backoff, and prompt the user for sign-in only after the retry budget is exhausted.',
        libraryName,
        title: 'Session recovery',
        sourceType: 'note',
      },
    });
    if (storeResult.isError) {
      throw new Error(`Library store call failed: ${getTextContent(storeResult)}`);
    }

    const searchArgs = {
      query: LIBRARY_QUERY,
      collection: 'libraries',
      libraryName,
      scope: 'global',
      mode: 'semantic',
      limit: 5,
      scoreThreshold: 0,
    };

    for (let i = 1; i <= attempts; i += 1) {
      const searchResult = await client.callTool({
        name: 'search',
        arguments: searchArgs,
      });
      if (searchResult.isError) {
        throw new Error(`Library semantic search failed: ${getTextContent(searchResult)}`);
      }

      const payload = parseJsonToolResult(searchResult);
      if (payload.results?.length > 0) {
        console.log('Semantic library smoke passed.');
        console.log(JSON.stringify(payload, null, 2));
        return;
      }

      await sleep(pollSeconds * 1000);
    }

    throw new Error(`No semantic hits found in the libraries collection after ${attempts} attempts.`);
  } finally {
    await transport.close().catch(() => {});
  }
}

async function runProjectSmoke({
  mcpUrl,
  token,
  projectDir,
  projectName,
  attempts,
  pollSeconds,
}) {
  await waitForHealth(mcpUrl, attempts, pollSeconds);
  ensureProjectFiles(projectDir);
  const wqmProjectDir = convertToWqmPath(projectDir);
  const projectId = computeProjectId(wqmProjectDir);

  const { client, transport } = await connectClient(mcpUrl, token);
  try {
    const registerResult = await client.callTool({
      name: 'store',
      arguments: {
        type: 'project',
        path: wqmProjectDir,
        name: projectName,
      },
    });

    let registerPayload = null;
    const registerText = getTextContent(registerResult);
    if (registerResult.isError) {
      if (!/timed out|timeout/i.test(registerText)) {
        throw new Error(`Project registration failed: ${registerText}`);
      }
      console.warn('Project registration timed out, but the backend may still finish asynchronously.');
    } else {
      registerPayload = parseJsonToolResult(registerResult);
      if (registerPayload.project_id && registerPayload.project_id !== projectId) {
        throw new Error(
          `Project registration returned unexpected project_id ${registerPayload.project_id}; expected ${projectId}`
        );
      }
    }

    const readmePath = join(projectDir, 'README.md');
    const readmeUpdate = [
      '# Session recovery',
      '',
      'When credentials stop working, the system should refresh the session, retry with exponential backoff, and only ask for interactive sign-in after the retry budget has been exhausted.',
      '',
      'Post-registration update: credentials can expire, so the application should refresh the session and keep retrying with backoff before prompting for sign-in.',
    ].join('\n');
    writeFileSync(readmePath, `${readmeUpdate}\n`, 'utf8');

    const searchArgs = {
      query: PROJECT_QUERY,
      collection: 'projects',
      scope: 'project',
      mode: 'semantic',
      projectId,
      limit: 5,
      scoreThreshold: 0,
    };

    for (let i = 1; i <= attempts; i += 1) {
      const searchResult = await client.callTool({
        name: 'search',
        arguments: searchArgs,
      });
      if (searchResult.isError) {
        throw new Error(`Project semantic search failed: ${getTextContent(searchResult)}`);
      }

      const payload = parseJsonToolResult(searchResult);
      if (payload.results?.length > 0) {
        console.log('Semantic project smoke passed.');
        console.log(`Project ID: ${projectId}`);
        console.log(JSON.stringify(payload, null, 2));
        return;
      }

      await sleep(pollSeconds * 1000);
    }

    throw new Error(`No semantic hits found in the projects collection after ${attempts} attempts.`);
  } finally {
    await transport.close().catch(() => {});
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const mode = String(args.mode ?? '').trim().toLowerCase();
  const repoDir = resolve(String(args['repo-dir'] ?? process.cwd()));
  const mcpUrl = String(args['mcp-url'] ?? '').trim();
  const token = String(args.token ?? '').trim();
  const attempts = asNumber(args.attempts, DEFAULT_ATTEMPTS);
  const pollSeconds = asNumber(args['poll-seconds'], DEFAULT_POLL_SECONDS);

  if (!mcpUrl) {
    throw new Error('--mcp-url is required.');
  }
  if (!token) {
    throw new Error('--token is required.');
  }

  if (mode === 'library') {
    const libraryName = String(args['library-name'] ?? 'semantic-smoke').trim();
    await runLibrarySmoke({ mcpUrl, token, libraryName, attempts, pollSeconds });
    return;
  }

  if (mode === 'project') {
    const projectName = String(args['project-name'] ?? 'semantic-project').trim();
    const projectDir = buildProjectDir(repoDir, String(args['project-dir'] ?? '').trim());
    await runProjectSmoke({
      mcpUrl,
      token,
      projectDir,
      projectName,
      attempts,
      pollSeconds,
    });
    return;
  }

  throw new Error(`Unknown mode: ${mode}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
