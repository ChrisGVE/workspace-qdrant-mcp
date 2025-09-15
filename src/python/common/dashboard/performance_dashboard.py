"""
Web-based performance monitoring dashboard for metadata filtering operations.

This module provides a real-time web dashboard for monitoring search performance,
accuracy metrics, and baseline compliance. Features interactive charts, alerts,
and comprehensive multi-tenant performance visualization.

Key Features:
    - Real-time performance metrics with live updates
    - Baseline comparison visualization (2.18ms, 94.2% precision)
    - Multi-tenant search performance monitoring
    - Interactive charts and graphs
    - Performance alert notifications
    - Historical trend analysis
    - Export capabilities for reports

Task 233.6: Web dashboard for multi-tenant search performance monitoring.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
import uvicorn

from ..core.performance_monitoring import MetadataFilteringPerformanceMonitor
from ..core.hybrid_search import HybridSearchEngine


class PerformanceDashboardServer:
    """
    FastAPI-based performance dashboard server.

    Provides real-time monitoring interface for metadata filtering performance
    with WebSocket support for live updates and interactive visualizations.
    """

    def __init__(
        self,
        performance_monitor: MetadataFilteringPerformanceMonitor,
        title: str = "Metadata Filtering Performance Dashboard"
    ):
        """Initialize dashboard server."""
        self.performance_monitor = performance_monitor
        self.title = title
        self.app = FastAPI(title=title)
        self.active_websockets: List[WebSocket] = []

        # Setup routes and websockets
        self._setup_routes()
        self._setup_websockets()

        logger.info("Performance dashboard server initialized")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self._render_dashboard_template()

        @self.app.get("/api/performance/status")
        async def get_performance_status():
            """Get current performance status."""
            try:
                status = self.performance_monitor.get_performance_status()
                return JSONResponse(status)
            except Exception as e:
                logger.error(f"Failed to get performance status: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/performance/dashboard")
        async def get_dashboard_data():
            """Get real-time dashboard data."""
            try:
                dashboard_data = self.performance_monitor.dashboard.get_real_time_dashboard()
                return JSONResponse(dashboard_data)
            except Exception as e:
                logger.error(f"Failed to get dashboard data: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/performance/baseline")
        async def get_baseline_config():
            """Get baseline configuration."""
            return JSONResponse(self.performance_monitor.baseline.to_dict())

        @self.app.get("/api/performance/accuracy")
        async def get_accuracy_summary():
            """Get accuracy summary."""
            try:
                summary = self.performance_monitor.accuracy_tracker.get_accuracy_summary()
                return JSONResponse(summary)
            except Exception as e:
                logger.error(f"Failed to get accuracy summary: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/performance/benchmarks")
        async def get_benchmark_history():
            """Get benchmark history."""
            try:
                benchmarks = self.performance_monitor.benchmark_suite.get_benchmark_history()
                benchmark_data = []

                for benchmark in benchmarks:
                    benchmark_data.append({
                        "benchmark_id": benchmark.benchmark_id,
                        "timestamp": benchmark.timestamp.isoformat(),
                        "test_name": benchmark.test_name,
                        "avg_response_time": benchmark.avg_response_time,
                        "p95_response_time": benchmark.p95_response_time,
                        "avg_precision": benchmark.avg_precision,
                        "avg_recall": benchmark.avg_recall,
                        "passes_baseline": benchmark.passes_baseline(self.performance_monitor.baseline),
                        "performance_regression": benchmark.performance_regression,
                        "accuracy_regression": benchmark.accuracy_regression
                    })

                return JSONResponse(benchmark_data)
            except Exception as e:
                logger.error(f"Failed to get benchmark history: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/performance/alerts")
        async def get_performance_alerts():
            """Get recent performance alerts."""
            try:
                alerts = self.performance_monitor.accuracy_tracker.get_recent_accuracy_alerts(hours=24)
                return JSONResponse(alerts)
            except Exception as e:
                logger.error(f"Failed to get alerts: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/performance/simulate")
        async def simulate_performance_data():
            """Simulate performance data for dashboard testing."""
            import random

            # Simulate real-time metrics
            for i in range(10):
                response_time = random.uniform(1.5, 4.0)  # 1.5-4ms range
                precision = random.uniform(89.0, 97.0)   # 89-97% range
                recall = random.uniform(85.0, 95.0)      # 85-95% range

                self.performance_monitor.dashboard.record_real_time_metric(
                    operation_type=f"simulated_search_{i}",
                    response_time=response_time,
                    accuracy_metrics={"precision": precision, "recall": recall},
                    metadata={"simulation": True, "timestamp": datetime.now().isoformat()}
                )

            return JSONResponse({"message": "Simulated data generated", "data_points": 10})

    def _setup_websockets(self):
        """Setup WebSocket endpoints for real-time updates."""

        @self.app.websocket("/ws/performance")
        async def websocket_performance_updates(websocket: WebSocket):
            """WebSocket endpoint for real-time performance updates."""
            await websocket.accept()
            self.active_websockets.append(websocket)

            try:
                while True:
                    # Send performance data every 5 seconds
                    dashboard_data = self.performance_monitor.dashboard.get_real_time_dashboard()

                    await websocket.send_text(json.dumps({
                        "type": "dashboard_update",
                        "data": dashboard_data,
                        "timestamp": datetime.now().isoformat()
                    }))

                    await asyncio.sleep(5)

            except WebSocketDisconnect:
                self.active_websockets.remove(websocket)
                logger.info("WebSocket client disconnected")

    def _render_dashboard_template(self) -> str:
        """Render the main dashboard HTML template."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .status-excellent { @apply bg-green-100 border-green-500 text-green-700; }
        .status-good { @apply bg-yellow-100 border-yellow-500 text-yellow-700; }
        .status-degraded { @apply bg-red-100 border-red-500 text-red-700; }
        .metric-card { @apply bg-white rounded-lg shadow-md p-6 border-l-4; }
        .chart-container { @apply bg-white rounded-lg shadow-md p-4; }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">Metadata Filtering Performance Dashboard</h1>
            <p class="text-gray-600">Real-time monitoring of search performance and accuracy metrics</p>
            <div class="flex items-center mt-2">
                <div id="connection-status" class="flex items-center">
                    <div class="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                    <span class="text-sm text-gray-600">Connected</span>
                </div>
                <div class="ml-4 text-sm text-gray-500" id="last-update">
                    Last updated: <span id="update-time">--:--:--</span>
                </div>
            </div>
        </div>

        <!-- Performance Overview -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <!-- Response Time -->
            <div class="metric-card border-blue-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 mb-1">Avg Response Time</p>
                        <p class="text-2xl font-bold text-gray-800" id="avg-response-time">--ms</p>
                    </div>
                    <div class="text-blue-500">
                        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                </div>
                <div class="mt-2">
                    <span class="text-xs text-gray-500">Target: 2.18ms</span>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div id="response-time-bar" class="bg-blue-500 h-2 rounded-full" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <!-- Precision -->
            <div class="metric-card border-green-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 mb-1">Search Precision</p>
                        <p class="text-2xl font-bold text-gray-800" id="avg-precision">--%</p>
                    </div>
                    <div class="text-green-500">
                        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                </div>
                <div class="mt-2">
                    <span class="text-xs text-gray-500">Target: 94.2%</span>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div id="precision-bar" class="bg-green-500 h-2 rounded-full" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <!-- Recall -->
            <div class="metric-card border-yellow-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 mb-1">Search Recall</p>
                        <p class="text-2xl font-bold text-gray-800" id="avg-recall">--%</p>
                    </div>
                    <div class="text-yellow-500">
                        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                        </svg>
                    </div>
                </div>
                <div class="mt-2">
                    <span class="text-xs text-gray-500">Target: 92.0%</span>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div id="recall-bar" class="bg-yellow-500 h-2 rounded-full" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <!-- Overall Status -->
            <div class="metric-card border-purple-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 mb-1">System Status</p>
                        <p class="text-lg font-bold text-gray-800" id="system-status">Unknown</p>
                    </div>
                    <div class="text-purple-500">
                        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 00-2-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                        </svg>
                    </div>
                </div>
                <div class="mt-2">
                    <div id="status-indicator" class="px-2 py-1 rounded text-xs font-medium">
                        Monitoring...
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Response Time Trend -->
            <div class="chart-container">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">Response Time Trend</h3>
                <canvas id="response-time-chart" width="400" height="200"></canvas>
            </div>

            <!-- Accuracy Metrics -->
            <div class="chart-container">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">Accuracy Metrics</h3>
                <canvas id="accuracy-chart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Baseline Comparison -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">Baseline Comparison</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="text-center">
                    <div class="text-3xl font-bold mb-2" id="response-time-comparison">--</div>
                    <div class="text-sm text-gray-600">Response Time vs 2.18ms target</div>
                    <div class="mt-2" id="response-time-status">
                        <span class="px-2 py-1 rounded text-xs font-medium">Checking...</span>
                    </div>
                </div>
                <div class="text-center">
                    <div class="text-3xl font-bold mb-2" id="precision-comparison">--</div>
                    <div class="text-sm text-gray-600">Precision vs 94.2% target</div>
                    <div class="mt-2" id="precision-status">
                        <span class="px-2 py-1 rounded text-xs font-medium">Checking...</span>
                    </div>
                </div>
                <div class="text-center">
                    <div class="text-3xl font-bold mb-2" id="recall-comparison">--</div>
                    <div class="text-sm text-gray-600">Recall vs 92.0% target</div>
                    <div class="mt-2" id="recall-status">
                        <span class="px-2 py-1 rounded text-xs font-medium">Checking...</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Alerts -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">Recent Performance Alerts</h3>
            <div id="alerts-container">
                <div class="text-gray-500 text-center py-4">
                    <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-900 mx-auto mb-2"></div>
                    Loading alerts...
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard JavaScript
        class PerformanceDashboard {
            constructor() {
                this.websocket = null;
                this.responseTimeChart = null;
                this.accuracyChart = null;
                this.init();
            }

            async init() {
                this.setupCharts();
                this.connectWebSocket();
                await this.loadInitialData();
                await this.loadAlerts();
            }

            setupCharts() {
                // Response Time Chart
                const rtCtx = document.getElementById('response-time-chart').getContext('2d');
                this.responseTimeChart = new Chart(rtCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Response Time (ms)',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.1
                        }, {
                            label: 'Target (2.18ms)',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            borderDash: [5, 5],
                            pointRadius: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 5
                            }
                        }
                    }
                });

                // Accuracy Chart
                const accCtx = document.getElementById('accuracy-chart').getContext('2d');
                this.accuracyChart = new Chart(accCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Precision (%)',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            backgroundColor: 'rgba(34, 197, 94, 0.1)',
                            tension: 0.1
                        }, {
                            label: 'Recall (%)',
                            data: [],
                            borderColor: 'rgb(251, 191, 36)',
                            backgroundColor: 'rgba(251, 191, 36, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                min: 80,
                                max: 100
                            }
                        }
                    }
                });
            }

            connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/performance`;

                this.websocket = new WebSocket(wsUrl);

                this.websocket.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    if (message.type === 'dashboard_update') {
                        this.updateDashboard(message.data);
                    }
                };

                this.websocket.onclose = () => {
                    document.getElementById('connection-status').innerHTML =
                        '<div class="w-3 h-3 bg-red-500 rounded-full mr-2"></div><span class="text-sm text-gray-600">Disconnected</span>';

                    // Attempt reconnection after 5 seconds
                    setTimeout(() => this.connectWebSocket(), 5000);
                };
            }

            async loadInitialData() {
                try {
                    const response = await fetch('/api/performance/dashboard');
                    const data = await response.json();
                    this.updateDashboard(data);
                } catch (error) {
                    console.error('Failed to load initial data:', error);
                }
            }

            async loadAlerts() {
                try {
                    const response = await fetch('/api/performance/alerts');
                    const alerts = await response.json();
                    this.updateAlerts(alerts);
                } catch (error) {
                    console.error('Failed to load alerts:', error);
                }
            }

            updateDashboard(data) {
                if (!data.performance_overview) return;

                const perf = data.performance_overview;
                const acc = data.accuracy_overview?.overall_accuracy || {};
                const baseline = data.baseline_comparison || {};

                // Update metrics
                document.getElementById('avg-response-time').textContent =
                    `${perf.avg_response_time?.toFixed(2) || '--'}ms`;
                document.getElementById('avg-precision').textContent =
                    `${acc.avg_precision?.toFixed(1) || '--'}%`;
                document.getElementById('avg-recall').textContent =
                    `${acc.avg_recall?.toFixed(1) || '--'}%`;

                // Update status
                document.getElementById('system-status').textContent =
                    perf.status?.toUpperCase() || 'UNKNOWN';

                const statusIndicator = document.getElementById('status-indicator');
                statusIndicator.textContent = perf.status?.toUpperCase() || 'MONITORING';
                statusIndicator.className = `px-2 py-1 rounded text-xs font-medium ${this.getStatusClass(perf.status)}`;

                // Update progress bars
                this.updateProgressBar('response-time-bar', perf.avg_response_time, 5.0);
                this.updateProgressBar('precision-bar', acc.avg_precision, 100);
                this.updateProgressBar('recall-bar', acc.avg_recall, 100);

                // Update baseline comparison
                this.updateBaselineComparison(perf, acc);

                // Update charts
                this.updateCharts(data);

                // Update timestamp
                document.getElementById('update-time').textContent =
                    new Date().toLocaleTimeString();
            }

            getStatusClass(status) {
                switch(status) {
                    case 'excellent': return 'status-excellent';
                    case 'good': return 'status-good';
                    case 'degraded': return 'status-degraded';
                    default: return 'bg-gray-100 border-gray-500 text-gray-700';
                }
            }

            updateProgressBar(elementId, value, max) {
                const element = document.getElementById(elementId);
                if (element && value !== undefined) {
                    const percentage = Math.min((value / max) * 100, 100);
                    element.style.width = `${percentage}%`;
                }
            }

            updateBaselineComparison(perf, acc) {
                // Response time comparison
                if (perf.avg_response_time !== undefined) {
                    const rtRatio = perf.avg_response_time / 2.18;
                    document.getElementById('response-time-comparison').textContent =
                        `${rtRatio.toFixed(2)}x`;

                    const rtStatus = document.getElementById('response-time-status');
                    if (perf.avg_response_time <= 2.18) {
                        rtStatus.innerHTML = '<span class="status-excellent">✓ Meeting Target</span>';
                    } else if (perf.avg_response_time <= 3.0) {
                        rtStatus.innerHTML = '<span class="status-good">△ Acceptable</span>';
                    } else {
                        rtStatus.innerHTML = '<span class="status-degraded">✗ Below Target</span>';
                    }
                }

                // Precision comparison
                if (acc.avg_precision !== undefined) {
                    const precisionDiff = acc.avg_precision - 94.2;
                    document.getElementById('precision-comparison').textContent =
                        `${precisionDiff >= 0 ? '+' : ''}${precisionDiff.toFixed(1)}%`;

                    const precisionStatus = document.getElementById('precision-status');
                    if (acc.avg_precision >= 94.2) {
                        precisionStatus.innerHTML = '<span class="status-excellent">✓ Meeting Target</span>';
                    } else if (acc.avg_precision >= 90.0) {
                        precisionStatus.innerHTML = '<span class="status-good">△ Above Minimum</span>';
                    } else {
                        precisionStatus.innerHTML = '<span class="status-degraded">✗ Below Minimum</span>';
                    }
                }

                // Recall comparison
                if (acc.avg_recall !== undefined) {
                    const recallDiff = acc.avg_recall - 92.0;
                    document.getElementById('recall-comparison').textContent =
                        `${recallDiff >= 0 ? '+' : ''}${recallDiff.toFixed(1)}%`;

                    const recallStatus = document.getElementById('recall-status');
                    if (acc.avg_recall >= 92.0) {
                        recallStatus.innerHTML = '<span class="status-excellent">✓ Meeting Target</span>';
                    } else if (acc.avg_recall >= 85.0) {
                        recallStatus.innerHTML = '<span class="status-good">△ Above Minimum</span>';
                    } else {
                        recallStatus.innerHTML = '<span class="status-degraded">✗ Below Minimum</span>';
                    }
                }
            }

            updateCharts(data) {
                const trends = data.trends || {};
                const responseTrend = trends.response_time_trend || [];

                if (responseTrend.length > 0) {
                    // Update response time chart
                    const labels = responseTrend.map((_, i) => `${i + 1}`);
                    this.responseTimeChart.data.labels = labels;
                    this.responseTimeChart.data.datasets[0].data = responseTrend;
                    this.responseTimeChart.data.datasets[1].data = new Array(responseTrend.length).fill(2.18);
                    this.responseTimeChart.update();
                }

                // Update accuracy chart (would need accuracy trend data)
                const acc = data.accuracy_overview?.overall_accuracy || {};
                if (acc.avg_precision !== undefined && acc.avg_recall !== undefined) {
                    // For now, just show current values
                    const currentLabel = new Date().toLocaleTimeString();
                    this.accuracyChart.data.labels.push(currentLabel);
                    this.accuracyChart.data.datasets[0].data.push(acc.avg_precision);
                    this.accuracyChart.data.datasets[1].data.push(acc.avg_recall);

                    // Keep only last 20 points
                    if (this.accuracyChart.data.labels.length > 20) {
                        this.accuracyChart.data.labels.shift();
                        this.accuracyChart.data.datasets[0].data.shift();
                        this.accuracyChart.data.datasets[1].data.shift();
                    }

                    this.accuracyChart.update();
                }
            }

            updateAlerts(alerts) {
                const container = document.getElementById('alerts-container');

                if (!alerts || alerts.length === 0) {
                    container.innerHTML = '<div class="text-gray-500 text-center py-4">No recent alerts</div>';
                    return;
                }

                const alertsHtml = alerts.map(alert => {
                    const alertClass = alert.severity === 'critical' ? 'border-red-500 bg-red-50' : 'border-yellow-500 bg-yellow-50';
                    const iconColor = alert.severity === 'critical' ? 'text-red-500' : 'text-yellow-500';

                    return `
                        <div class="border-l-4 ${alertClass} p-4 mb-3">
                            <div class="flex items-center">
                                <div class="${iconColor} mr-3">
                                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path>
                                    </svg>
                                </div>
                                <div class="flex-1">
                                    <div class="font-medium text-gray-800">${alert.type.replace('_', ' ').toUpperCase()}</div>
                                    <div class="text-sm text-gray-600 mt-1">
                                        ${alert.type === 'precision_regression' ? 'Search precision' : 'Search recall'}
                                        ${alert.measurement.toFixed(1)}% below ${alert.type === 'precision_regression' ? 'precision' : 'recall'} baseline
                                    </div>
                                    <div class="text-xs text-gray-500 mt-1">
                                        ${new Date(alert.timestamp).toLocaleString()} • Collection: ${alert.collection}
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                container.innerHTML = alertsHtml;
            }
        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new PerformanceDashboard();
        });
    </script>
</body>
</html>
        """.replace("{{ title }}", self.title)

    async def broadcast_update(self, data: Dict):
        """Broadcast update to all connected WebSocket clients."""
        if not self.active_websockets:
            return

        message = json.dumps({
            "type": "performance_update",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

        # Send to all connected clients
        disconnected = []
        for websocket in self.active_websockets:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.active_websockets.remove(websocket)

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the dashboard server."""
        logger.info(f"Starting performance dashboard server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


async def create_dashboard_server(
    performance_monitor: MetadataFilteringPerformanceMonitor,
    title: str = "Metadata Filtering Performance Dashboard"
) -> PerformanceDashboardServer:
    """Create and configure dashboard server."""
    return PerformanceDashboardServer(performance_monitor, title)


# Export main class
__all__ = ["PerformanceDashboardServer", "create_dashboard_server"]