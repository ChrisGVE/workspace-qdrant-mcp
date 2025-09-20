/**
 * Quick K6 Performance Test for CI/CD
 *
 * Lightweight performance test for continuous integration
 * Tests core MCP tools with minimal load to verify sub-200ms targets
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Lightweight metrics for CI
const responseTime = new Trend('response_time');
const sub200msRate = new Rate('sub_200ms_rate');

// Quick test configuration - minimal load for CI
export const options = {
  vus: 5,
  duration: '30s',
  thresholds: {
    'http_req_duration': ['p(95)<200'],
    'sub_200ms_rate': ['rate>=0.90'], // 90% threshold for CI
    'http_req_failed': ['rate<0.05'], // 5% error rate for CI
  },
};

const MCP_SERVER_URL = __ENV.MCP_SERVER_URL || 'http://127.0.0.1:8000';

// Essential MCP tools for quick testing
const QUICK_TESTS = [
  { tool: 'list_workspace_collections', params: {} },
  { tool: 'create_collection', params: {
    name: `ci_test_${Date.now()}`,
    config: { vectors: { size: 384, distance: 'Cosine' } }
  }},
  { tool: 'add_document_tool', params: {
    collection: 'default',
    content: 'CI test document',
    metadata: { type: 'ci_test' },
    document_id: `ci_doc_${Date.now()}`
  }},
  { tool: 'search_by_metadata_tool', params: {
    collection: 'default',
    metadata_filter: { type: 'ci_test' },
    limit: 5
  }},
  { tool: 'list_watched_folders', params: {} },
];

function callMCP(toolName, params) {
  const start = Date.now();

  const payload = {
    jsonrpc: '2.0',
    id: Math.random().toString(36).substr(2, 9),
    method: 'tools/call',
    params: { name: toolName, arguments: params }
  };

  const response = http.post(`${MCP_SERVER_URL}/mcp/call`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '3s',
  });

  const duration = Date.now() - start;
  responseTime.add(duration);
  sub200msRate.add(duration < 200);

  check(response, {
    [`${toolName}: status 200`]: (r) => r.status === 200,
    [`${toolName}: < 200ms`]: () => duration < 200,
  });

  return response;
}

export default function() {
  // Test a random subset of tools each iteration
  const testCount = Math.floor(Math.random() * 3) + 2; // 2-4 tests per iteration
  const tests = QUICK_TESTS.sort(() => 0.5 - Math.random()).slice(0, testCount);

  tests.forEach(({ tool, params }) => {
    callMCP(tool, params);
    sleep(0.1); // Brief pause between calls
  });

  sleep(Math.random() * 0.5); // Random pause between iterations
}

export function setup() {
  console.log('🚀 Quick Performance Test - CI Mode');
  const health = http.get(`${MCP_SERVER_URL}/health`, { timeout: '3s' });
  if (health.status !== 200) {
    throw new Error(`Server not ready: ${health.status}`);
  }
  console.log('✅ Server ready for testing');
}

export function teardown() {
  console.log('🏁 Quick performance test completed');
}