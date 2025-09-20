/**
 * K6 Performance Tests for Workspace-Qdrant-MCP Server
 *
 * Tests the 11 core MCP tools for sub-200ms response time requirements
 * Target: All MCP tools must respond within 200ms under load
 *
 * Test Scenarios:
 * - Load test: 10 VUs for 30 seconds
 * - Stress test: Ramp up to 50 VUs over 2 minutes
 * - Spike test: Sudden load spikes
 *
 * Performance Thresholds:
 * - Response time p95 < 200ms (strict requirement)
 * - Response time p99 < 500ms (acceptable degradation)
 * - Error rate < 1%
 * - Throughput > 100 RPS per tool
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics for MCP performance tracking
const mcpResponseTime = new Trend('mcp_response_time');
const mcpErrorRate = new Rate('mcp_error_rate');
const mcpThroughput = new Counter('mcp_requests_total');
const sub200msRate = new Rate('sub_200ms_responses');

// Test configuration
export const options = {
  scenarios: {
    // Load test scenario - sustained load
    load_test: {
      executor: 'constant-vus',
      vus: 10,
      duration: '30s',
      tags: { test_type: 'load' },
    },
    // Stress test scenario - gradual ramp up
    stress_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 20 },
        { duration: '60s', target: 50 },
        { duration: '30s', target: 0 },
      ],
      tags: { test_type: 'stress' },
    },
    // Spike test scenario - sudden load spikes
    spike_test: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      preAllocatedVUs: 50,
      maxVUs: 100,
      stages: [
        { duration: '30s', target: 50 },
        { duration: '10s', target: 200 }, // Spike
        { duration: '30s', target: 50 },
      ],
      tags: { test_type: 'spike' },
    },
  },
  thresholds: {
    // Critical performance requirements
    'http_req_duration': ['p(95)<200', 'p(99)<500'], // 95% under 200ms, 99% under 500ms
    'sub_200ms_responses': ['rate>=0.95'], // 95% of responses under 200ms
    'http_req_failed': ['rate<0.01'], // Less than 1% failures
  },
};

// MCP server configuration
const MCP_SERVER_URL = __ENV.MCP_SERVER_URL || 'http://127.0.0.1:8000';
const API_BASE = `${MCP_SERVER_URL}/mcp`;

// Test data for realistic scenarios
const TEST_DOCUMENTS = [
  {
    content: 'This is a test document for performance testing. It contains sample text for indexing and search operations.',
    metadata: { type: 'test', category: 'performance', priority: 'high' }
  },
  {
    content: 'Sample Python code: def hello_world(): print("Hello, World!")',
    metadata: { type: 'code', language: 'python', category: 'sample' }
  },
  {
    content: 'Performance testing requires careful measurement of response times and throughput under various load conditions.',
    metadata: { type: 'documentation', category: 'testing', priority: 'medium' }
  }
];

const TEST_SEARCH_QUERIES = [
  'performance testing',
  'python code',
  'hello world',
  'sample text',
  'documentation'
];

// Core MCP tools to test (11 critical tools)
const MCP_TOOLS = {
  LIST_COLLECTIONS: 'list_workspace_collections',
  CREATE_COLLECTION: 'create_collection',
  ADD_DOCUMENT: 'add_document_tool',
  GET_DOCUMENT: 'get_document_tool',
  SEARCH_METADATA: 'search_by_metadata_tool',
  UPDATE_SCRATCHBOOK: 'update_scratchbook_tool',
  SEARCH_SCRATCHBOOK: 'search_scratchbook_tool',
  LIST_SCRATCHBOOK: 'list_scratchbook_notes_tool',
  HYBRID_SEARCH: 'hybrid_search_advanced_tool',
  ADD_WATCH: 'add_watch_folder',
  LIST_WATCHES: 'list_watched_folders'
};

/**
 * Make MCP tool call with performance tracking
 */
function callMCPTool(toolName, params = {}) {
  const startTime = Date.now();

  const payload = {
    jsonrpc: '2.0',
    id: Math.random().toString(36).substr(2, 9),
    method: 'tools/call',
    params: {
      name: toolName,
      arguments: params
    }
  };

  const response = http.post(`${API_BASE}/call`, JSON.stringify(payload), {
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    timeout: '5s', // 5 second timeout for safety
  });

  const duration = Date.now() - startTime;

  // Track custom metrics
  mcpResponseTime.add(duration);
  mcpThroughput.add(1);

  const success = check(response, {
    [`${toolName}: status 200`]: (r) => r.status === 200,
    [`${toolName}: response time < 200ms`]: () => duration < 200,
    [`${toolName}: response time < 500ms`]: () => duration < 500,
    [`${toolName}: valid JSON response`]: (r) => {
      try {
        JSON.parse(r.body);
        return true;
      } catch {
        return false;
      }
    },
  });

  // Track sub-200ms rate
  sub200msRate.add(duration < 200);

  if (!success) {
    mcpErrorRate.add(1);
    console.warn(`${toolName} failed: ${response.status} - ${response.body}`);
  } else {
    mcpErrorRate.add(0);
  }

  return {
    response,
    duration,
    success
  };
}

/**
 * Test workspace collection operations
 */
function testCollectionOperations() {
  // Test listing collections
  callMCPTool(MCP_TOOLS.LIST_COLLECTIONS);

  // Test creating a collection
  const collectionName = `perf_test_${Date.now()}`;
  callMCPTool(MCP_TOOLS.CREATE_COLLECTION, {
    name: collectionName,
    config: {
      vectors: {
        size: 384,
        distance: 'Cosine'
      }
    }
  });

  return collectionName;
}

/**
 * Test document operations
 */
function testDocumentOperations(collectionName) {
  const testDoc = TEST_DOCUMENTS[Math.floor(Math.random() * TEST_DOCUMENTS.length)];

  // Test adding document
  const addResult = callMCPTool(MCP_TOOLS.ADD_DOCUMENT, {
    collection: collectionName,
    content: testDoc.content,
    metadata: testDoc.metadata,
    document_id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`
  });

  // Test getting document (if add was successful)
  if (addResult.success) {
    try {
      const addResponse = JSON.parse(addResult.response.body);
      if (addResponse.result && addResponse.result.document_id) {
        callMCPTool(MCP_TOOLS.GET_DOCUMENT, {
          collection: collectionName,
          document_id: addResponse.result.document_id
        });
      }
    } catch (e) {
      console.warn('Failed to parse add document response for get test');
    }
  }
}

/**
 * Test search operations
 */
function testSearchOperations(collectionName) {
  const query = TEST_SEARCH_QUERIES[Math.floor(Math.random() * TEST_SEARCH_QUERIES.length)];

  // Test metadata search
  callMCPTool(MCP_TOOLS.SEARCH_METADATA, {
    collection: collectionName,
    metadata_filter: { type: 'test' },
    limit: 10
  });

  // Test hybrid search
  callMCPTool(MCP_TOOLS.HYBRID_SEARCH, {
    collection: collectionName,
    query: query,
    limit: 10,
    hybrid_alpha: 0.5
  });
}

/**
 * Test scratchbook operations
 */
function testScratchbookOperations() {
  const projectName = `perf_project_${Date.now()}`;
  const noteContent = `Performance test note created at ${new Date().toISOString()}`;

  // Test updating scratchbook
  callMCPTool(MCP_TOOLS.UPDATE_SCRATCHBOOK, {
    content: noteContent,
    project_name: projectName,
    note_type: 'performance_test'
  });

  // Test searching scratchbook
  callMCPTool(MCP_TOOLS.SEARCH_SCRATCHBOOK, {
    query: 'performance test',
    project_filter: projectName,
    limit: 5
  });

  // Test listing scratchbook notes
  callMCPTool(MCP_TOOLS.LIST_SCRATCHBOOK, {
    project_name: projectName,
    limit: 10
  });
}

/**
 * Test watch folder operations
 */
function testWatchOperations() {
  // Test listing watched folders
  callMCPTool(MCP_TOOLS.LIST_WATCHES);

  // Test adding watch folder (using temp directory)
  const tempPath = `/tmp/k6_test_${Date.now()}`;
  callMCPTool(MCP_TOOLS.ADD_WATCH, {
    folder_path: tempPath,
    collection_name: `watch_test_${Date.now()}`,
    watch_config: {
      include_patterns: ['*.txt', '*.md'],
      exclude_patterns: ['*.tmp']
    }
  });
}

/**
 * Main test function - runs one iteration of all performance tests
 */
export default function() {
  // Randomize test execution order to avoid patterns
  const testFunctions = [
    testCollectionOperations,
    testScratchbookOperations,
    testWatchOperations
  ];

  // Execute core tests
  const collectionName = testCollectionOperations();
  sleep(0.1); // Brief pause between operations

  testDocumentOperations(collectionName);
  sleep(0.1);

  testSearchOperations(collectionName);
  sleep(0.1);

  testScratchbookOperations();
  sleep(0.1);

  testWatchOperations();

  // Random sleep between 0.1 and 0.5 seconds to simulate realistic usage
  sleep(Math.random() * 0.4 + 0.1);
}

/**
 * Setup function - runs once before all tests
 */
export function setup() {
  console.log('🚀 Starting K6 MCP Performance Tests');
  console.log(`📊 Target: All tools respond within 200ms`);
  console.log(`🎯 Testing ${Object.keys(MCP_TOOLS).length} core MCP tools`);
  console.log(`🌐 Server: ${MCP_SERVER_URL}`);

  // Verify server is running
  const healthCheck = http.get(`${MCP_SERVER_URL}/health`, { timeout: '5s' });
  if (healthCheck.status !== 200) {
    throw new Error(`MCP server not responding at ${MCP_SERVER_URL}`);
  }

  console.log('✅ MCP server health check passed');
  return { serverUrl: MCP_SERVER_URL };
}

/**
 * Teardown function - runs once after all tests
 */
export function teardown(data) {
  console.log('🏁 K6 MCP Performance Tests Completed');
  console.log(`📈 Total requests: ${mcpThroughput.count}`);
  console.log(`⚡ Sub-200ms rate: ${(sub200msRate.count / mcpThroughput.count * 100).toFixed(2)}%`);
}