#!/usr/bin/env python3
"""
Health Check Server for Workspace Qdrant MCP

This script provides comprehensive health checks for the application,
including database connectivity, memory usage, and service availability.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import aiohttp
import psutil
from aiohttp import web

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthCheckService:
    """Health check service for monitoring application and system health."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checks_history: List[Dict] = []
        self.max_history = 100
        
    async def check_database_connection(self) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            # Replace with actual database connection check
            # This is a placeholder for Qdrant connection
            async with aiohttp.ClientSession() as session:
                qdrant_url = self.config.get('qdrant_url', 'http://localhost:6333')
                async with session.get(f'{qdrant_url}/cluster') as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'healthy',
                            'response_time_ms': response.headers.get('X-Response-Time', 0),
                            'details': {
                                'qdrant_status': data.get('status', 'unknown'),
                                'peer_count': len(data.get('peers', []))
                            }
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'error': f'HTTP {response.status}'
                        }
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                'status': 'healthy' if memory.percent < 85 else 'warning' if memory.percent < 95 else 'critical',
                'usage_percent': memory.percent,
                'available_bytes': memory.available,
                'used_bytes': memory.used,
                'total_bytes': memory.total
            }
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return {
                'status': 'healthy' if usage_percent < 80 else 'warning' if usage_percent < 90 else 'critical',
                'usage_percent': usage_percent,
                'free_bytes': disk.free,
                'used_bytes': disk.used,
                'total_bytes': disk.total
            }
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            return {
                'status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 95 else 'critical',
                'usage_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_average': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
            }
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_application_endpoints(self) -> Dict[str, Any]:
        """Check application-specific endpoints."""
        endpoints_to_check = self.config.get('endpoints', [
            'http://localhost:8000/health',
            'http://localhost:8001/metrics',
        ])
        
        endpoint_results = {}
        overall_status = 'healthy'
        
        for endpoint in endpoints_to_check:
            try:
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response_time = (time.time() - start_time) * 1000  # Convert to ms
                        
                        status = 'healthy' if response.status < 400 else 'unhealthy'
                        if status == 'unhealthy':
                            overall_status = 'unhealthy'
                        
                        endpoint_results[endpoint] = {
                            'status': status,
                            'response_code': response.status,
                            'response_time_ms': round(response_time, 2)
                        }
                        
            except asyncio.TimeoutError:
                endpoint_results[endpoint] = {
                    'status': 'unhealthy',
                    'error': 'timeout'
                }
                overall_status = 'unhealthy'
            except Exception as e:
                endpoint_results[endpoint] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'endpoints': endpoint_results
        }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        start_time = time.time()
        
        # Run all checks concurrently
        checks = await asyncio.gather(
            self.check_database_connection(),
            self.check_memory_usage(),
            self.check_disk_usage(),
            self.check_cpu_usage(),
            self.check_application_endpoints(),
            return_exceptions=True
        )
        
        # Process results
        db_check, memory_check, disk_check, cpu_check, endpoints_check = checks
        
        # Determine overall status
        all_statuses = [
            db_check.get('status', 'unhealthy'),
            memory_check.get('status', 'unhealthy'),
            disk_check.get('status', 'unhealthy'),
            cpu_check.get('status', 'unhealthy'),
            endpoints_check.get('status', 'unhealthy')
        ]
        
        if 'unhealthy' in all_statuses:
            overall_status = 'unhealthy'
        elif 'critical' in all_statuses:
            overall_status = 'critical'
        elif 'warning' in all_statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': overall_status,
            'checks': {
                'database': db_check,
                'memory': memory_check,
                'disk': disk_check,
                'cpu': cpu_check,
                'endpoints': endpoints_check
            },
            'execution_time_ms': round((time.time() - start_time) * 1000, 2)
        }
        
        # Store in history
        self.checks_history.append(result)
        if len(self.checks_history) > self.max_history:
            self.checks_history.pop(0)
        
        return result

# Web handlers
async def health_handler(request):
    """Main health check endpoint."""
    health_service = request.app['health_service']
    result = await health_service.run_all_checks()
    
    # Set HTTP status code based on health status
    status_map = {
        'healthy': 200,
        'warning': 200,
        'critical': 503,
        'unhealthy': 503
    }
    
    return web.Response(
        text=json.dumps(result, indent=2),
        status=status_map.get(result['status'], 503),
        content_type='application/json'
    )

async def health_summary_handler(request):
    """Summary health check endpoint (lightweight)."""
    health_service = request.app['health_service']
    
    # Quick checks only
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu = psutil.cpu_percent()
    
    overall_status = 'healthy'
    if memory.percent > 95 or (disk.used / disk.total) * 100 > 95 or cpu > 95:
        overall_status = 'critical'
    elif memory.percent > 85 or (disk.used / disk.total) * 100 > 85 or cpu > 85:
        overall_status = 'warning'
    
    return web.Response(
        text=json.dumps({
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat()
        }),
        status=200 if overall_status != 'critical' else 503,
        content_type='application/json'
    )

async def readiness_handler(request):
    """Kubernetes readiness probe endpoint."""
    health_service = request.app['health_service']
    
    # Check if application is ready to serve traffic
    try:
        db_check = await health_service.check_database_connection()
        if db_check['status'] != 'healthy':
            return web.Response(text='Not Ready', status=503)
        
        return web.Response(text='Ready', status=200)
    except Exception:
        return web.Response(text='Not Ready', status=503)

async def liveness_handler(request):
    """Kubernetes liveness probe endpoint."""
    # Simple liveness check - if we can respond, we're alive
    return web.Response(text='Alive', status=200)

async def metrics_handler(request):
    """Prometheus metrics endpoint."""
    health_service = request.app['health_service']
    
    if not health_service.checks_history:
        await health_service.run_all_checks()
    
    latest_check = health_service.checks_history[-1]
    
    metrics = []
    
    # Status metrics (0=unhealthy, 1=warning, 2=healthy, 3=critical)
    status_values = {'unhealthy': 0, 'warning': 1, 'healthy': 2, 'critical': 3}
    
    for check_name, check_data in latest_check['checks'].items():
        status_value = status_values.get(check_data.get('status', 'unhealthy'), 0)
        metrics.append(f'health_check_status{{check="{check_name}"}} {status_value}')
    
    # Resource metrics
    if 'memory' in latest_check['checks']:
        memory_data = latest_check['checks']['memory']
        metrics.append(f'health_check_memory_usage_percent {memory_data.get("usage_percent", 0)}')
    
    if 'disk' in latest_check['checks']:
        disk_data = latest_check['checks']['disk']
        metrics.append(f'health_check_disk_usage_percent {disk_data.get("usage_percent", 0)}')
    
    if 'cpu' in latest_check['checks']:
        cpu_data = latest_check['checks']['cpu']
        metrics.append(f'health_check_cpu_usage_percent {cpu_data.get("usage_percent", 0)}')
    
    # Overall health
    overall_status_value = status_values.get(latest_check['status'], 0)
    metrics.append(f'health_check_overall_status {overall_status_value}')
    
    metrics.append(f'health_check_execution_time_ms {latest_check["execution_time_ms"]}')
    
    return web.Response(
        text='\n'.join(metrics) + '\n',
        content_type='text/plain'
    )

def create_app(config: Dict[str, Any]) -> web.Application:
    """Create and configure the web application."""
    app = web.Application()
    
    # Initialize health service
    health_service = HealthCheckService(config)
    app['health_service'] = health_service
    
    # Setup routes
    app.router.add_get('/health', health_handler)
    app.router.add_get('/health/summary', health_summary_handler)
    app.router.add_get('/ready', readiness_handler)
    app.router.add_get('/alive', liveness_handler)
    app.router.add_get('/metrics', metrics_handler)
    
    return app

async def main():
    """Main function to run the health check server."""
    config = {
        'qdrant_url': 'http://localhost:6333',
        'endpoints': [
            'http://localhost:8000/health',
            'http://localhost:8001/metrics',
        ],
        'port': 8080,
        'host': '0.0.0.0'
    }
    
    app = create_app(config)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, config['host'], config['port'])
    await site.start()
    
    logger.info(f"Health check server started on {config['host']}:{config['port']}")
    logger.info("Available endpoints:")
    logger.info("  /health - Comprehensive health check")
    logger.info("  /health/summary - Lightweight health summary")
    logger.info("  /ready - Readiness probe")
    logger.info("  /alive - Liveness probe")
    logger.info("  /metrics - Prometheus metrics")
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down health check server...")
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())