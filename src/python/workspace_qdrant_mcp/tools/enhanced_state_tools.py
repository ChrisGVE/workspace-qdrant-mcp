"""
Enhanced State Management Tools for MCP Server.

This module provides MCP tool endpoints for the enhanced SQLite state management
features including backup/restore, migrations, performance monitoring, and 
database integrity checks.

Available Tools:
    - create_database_backup: Create a database backup with metadata
    - restore_database_backup: Restore database from a backup
    - list_database_backups: List available database backups
    - get_performance_statistics: Get database performance metrics
    - verify_database_integrity: Comprehensive database integrity check
    - apply_database_migration: Apply database schema migration
    - cleanup_performance_logs: Clean up old performance logs
    - get_database_health: Overall database health report
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger
from common.core.sqlite_state_manager import SQLiteStateManager
from workspace_qdrant_mcp.core.state_enhancements import (
    EnhancedStateManager,
    create_enhanced_state_manager
)


# Global enhanced state manager instance
_enhanced_manager: Optional[EnhancedStateManager] = None


async def _get_enhanced_manager() -> EnhancedStateManager:
    """Get or create the enhanced state manager instance."""
    global _enhanced_manager
    if _enhanced_manager is None:
        # Use default database path or configuration
        _enhanced_manager = await create_enhanced_state_manager()
    return _enhanced_manager


async def create_database_backup(description: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a database backup with metadata.
    
    Args:
        description: Optional description for the backup
        
    Returns:
        Dictionary containing backup ID and metadata
    """
    try:
        manager = await _get_enhanced_manager()
        backup_id = await manager.create_backup(description)
        
        # Get backup details
        backup_list = await manager.get_backup_list()
        backup_info = next((b for b in backup_list if b['backup_id'] == backup_id), None)
        
        return {
            'success': True,
            'backup_id': backup_id,
            'backup_info': backup_info,
            'message': f'Database backup created successfully: {backup_id}'
        }
        
    except Exception as e:
        logger.error(f"Failed to create database backup: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to create database backup'
        }


async def restore_database_backup(backup_id: str) -> Dict[str, Any]:
    """
    Restore database from a backup.
    
    Args:
        backup_id: ID of the backup to restore
        
    Returns:
        Dictionary containing restore status
    """
    try:
        manager = await _get_enhanced_manager()
        success = await manager.restore_backup(backup_id)
        
        if success:
            return {
                'success': True,
                'backup_id': backup_id,
                'message': f'Database successfully restored from backup: {backup_id}'
            }
        else:
            return {
                'success': False,
                'backup_id': backup_id,
                'message': f'Failed to restore from backup: {backup_id}'
            }
            
    except Exception as e:
        logger.error(f"Failed to restore database backup {backup_id}: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to restore database backup: {backup_id}'
        }


async def list_database_backups() -> Dict[str, Any]:
    """
    List all available database backups.
    
    Returns:
        Dictionary containing list of backups with metadata
    """
    try:
        manager = await _get_enhanced_manager()
        backups = await manager.get_backup_list()
        
        return {
            'success': True,
            'backups': backups,
            'total_backups': len(backups),
            'message': f'Found {len(backups)} database backups'
        }
        
    except Exception as e:
        logger.error(f"Failed to list database backups: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to list database backups'
        }


async def get_performance_statistics(hours: int = 24) -> Dict[str, Any]:
    """
    Get database performance statistics for the specified time period.
    
    Args:
        hours: Number of hours to include in statistics (default: 24)
        
    Returns:
        Dictionary containing performance statistics
    """
    try:
        manager = await _get_enhanced_manager()
        stats = await manager.get_performance_stats(hours)
        
        return {
            'success': True,
            'statistics': stats,
            'time_period_hours': hours,
            'message': f'Retrieved performance statistics for last {hours} hours'
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance statistics: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to get performance statistics'
        }


async def verify_database_integrity() -> Dict[str, Any]:
    """
    Perform comprehensive database integrity check.
    
    Returns:
        Dictionary containing integrity check results
    """
    try:
        manager = await _get_enhanced_manager()
        integrity_results = await manager.verify_database_integrity()
        
        # Determine overall health
        issues_found = []
        if 'sqlite_integrity' in integrity_results:
            if not all('ok' in result.lower() for result in integrity_results['sqlite_integrity']):
                issues_found.append('SQLite integrity issues detected')
        
        if 'foreign_key_violations' in integrity_results:
            if integrity_results['foreign_key_violations']:
                issues_found.append(f"{len(integrity_results['foreign_key_violations'])} foreign key violations")
        
        if 'orphaned_records' in integrity_results:
            for check_name, count in integrity_results['orphaned_records'].items():
                if count > 0:
                    issues_found.append(f"{count} {check_name}")
        
        overall_health = 'Good' if not issues_found else 'Issues Found'
        
        return {
            'success': True,
            'integrity_results': integrity_results,
            'overall_health': overall_health,
            'issues_found': issues_found,
            'message': f'Database integrity check completed - {overall_health}'
        }
        
    except Exception as e:
        logger.error(f"Failed to verify database integrity: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to verify database integrity'
        }


async def apply_database_migration(target_version: int, create_backup: bool = True) -> Dict[str, Any]:
    """
    Apply database schema migration to target version.
    
    Args:
        target_version: Target schema version to migrate to
        create_backup: Whether to create backup before migration (default: True)
        
    Returns:
        Dictionary containing migration status
    """
    try:
        manager = await _get_enhanced_manager()
        
        # Validate migration first
        is_valid = await manager.validate_migration(target_version)
        if not is_valid:
            return {
                'success': False,
                'target_version': target_version,
                'message': f'Migration to version {target_version} validation failed'
            }
        
        # Apply migration
        success = await manager.apply_migration(target_version, create_backup)
        
        if success:
            return {
                'success': True,
                'target_version': target_version,
                'backup_created': create_backup,
                'message': f'Successfully migrated database to version {target_version}'
            }
        else:
            return {
                'success': False,
                'target_version': target_version,
                'message': f'Failed to migrate database to version {target_version}'
            }
            
    except Exception as e:
        logger.error(f"Failed to apply database migration to version {target_version}: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to apply database migration to version {target_version}'
        }


async def cleanup_performance_logs(days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Clean up old performance logs.
    
    Args:
        days_to_keep: Number of days of logs to keep (default: 30)
        
    Returns:
        Dictionary containing cleanup results
    """
    try:
        manager = await _get_enhanced_manager()
        deleted_count = await manager.cleanup_performance_logs(days_to_keep)
        
        return {
            'success': True,
            'days_to_keep': days_to_keep,
            'deleted_count': deleted_count,
            'message': f'Cleaned up {deleted_count} old performance log entries'
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup performance logs: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to cleanup performance logs'
        }


async def get_database_health() -> Dict[str, Any]:
    """
    Get comprehensive database health report including statistics and integrity.
    
    Returns:
        Dictionary containing comprehensive health information
    """
    try:
        manager = await _get_enhanced_manager()
        
        # Get various health metrics
        health_report = {}
        
        # Basic database statistics
        try:
            stats = await manager.base_manager.get_database_stats()
            health_report['database_stats'] = stats
        except Exception as e:
            health_report['database_stats'] = {'error': str(e)}
        
        # Performance statistics
        try:
            perf_stats = await manager.get_performance_stats(hours=24)
            health_report['performance_stats'] = perf_stats
        except Exception as e:
            health_report['performance_stats'] = {'error': str(e)}
        
        # Integrity check
        try:
            integrity = await manager.verify_database_integrity()
            health_report['integrity_check'] = integrity
        except Exception as e:
            health_report['integrity_check'] = {'error': str(e)}
        
        # Backup information
        try:
            backups = await manager.get_backup_list()
            recent_backups = [b for b in backups[:5]]  # Last 5 backups
            health_report['recent_backups'] = recent_backups
            health_report['total_backups'] = len(backups)
        except Exception as e:
            health_report['recent_backups'] = []
            health_report['total_backups'] = 0
            health_report['backup_error'] = str(e)
        
        # Determine overall health status
        critical_issues = []
        warnings = []
        
        # Check for critical issues
        if 'error' in health_report['database_stats']:
            critical_issues.append('Database statistics unavailable')
        
        if 'error' in health_report['integrity_check']:
            critical_issues.append('Integrity check failed')
        elif 'sqlite_integrity' in health_report['integrity_check']:
            sqlite_results = health_report['integrity_check']['sqlite_integrity']
            if not all('ok' in result.lower() for result in sqlite_results):
                critical_issues.append('SQLite integrity issues detected')
        
        # Check for warnings
        if health_report['total_backups'] == 0:
            warnings.append('No database backups available')
        elif health_report['total_backups'] < 3:
            warnings.append('Few database backups available')
        
        if 'error' in health_report['performance_stats']:
            warnings.append('Performance statistics unavailable')
        
        # Overall status
        if critical_issues:
            overall_status = 'Critical'
        elif warnings:
            overall_status = 'Warning'
        else:
            overall_status = 'Healthy'
        
        return {
            'success': True,
            'overall_status': overall_status,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'health_report': health_report,
            'message': f'Database health check completed - Status: {overall_status}'
        }
        
    except Exception as e:
        logger.error(f"Failed to get database health: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to get database health information'
        }


async def execute_atomic_bulk_operation(operations: List[Dict[str, Any]], operation_name: str = "bulk_operation") -> Dict[str, Any]:
    """
    Execute multiple database operations atomically.
    
    Args:
        operations: List of operation dictionaries with type, table, data, and conditions
        operation_name: Name for the operation (for performance tracking)
        
    Returns:
        Dictionary containing operation results
    """
    try:
        manager = await _get_enhanced_manager()
        success = await manager.atomic_bulk_operation(operations, operation_name)
        
        return {
            'success': success,
            'operation_name': operation_name,
            'operations_count': len(operations),
            'message': f'Atomic bulk operation {"succeeded" if success else "failed"}: {len(operations)} operations'
        }
        
    except Exception as e:
        logger.error(f"Failed to execute atomic bulk operation: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to execute atomic bulk operation'
        }


# Tool registration functions for MCP server integration
def get_enhanced_state_tools() -> Dict[str, Any]:
    """Get all enhanced state management tools for MCP registration."""
    return {
        'create_database_backup': {
            'function': create_database_backup,
            'description': 'Create a database backup with metadata',
            'parameters': {
                'description': {'type': 'string', 'description': 'Optional description for the backup'}
            }
        },
        'restore_database_backup': {
            'function': restore_database_backup,
            'description': 'Restore database from a backup',
            'parameters': {
                'backup_id': {'type': 'string', 'description': 'ID of the backup to restore', 'required': True}
            }
        },
        'list_database_backups': {
            'function': list_database_backups,
            'description': 'List all available database backups',
            'parameters': {}
        },
        'get_performance_statistics': {
            'function': get_performance_statistics,
            'description': 'Get database performance statistics for the specified time period',
            'parameters': {
                'hours': {'type': 'integer', 'description': 'Number of hours to include in statistics (default: 24)', 'default': 24}
            }
        },
        'verify_database_integrity': {
            'function': verify_database_integrity,
            'description': 'Perform comprehensive database integrity check',
            'parameters': {}
        },
        'apply_database_migration': {
            'function': apply_database_migration,
            'description': 'Apply database schema migration to target version',
            'parameters': {
                'target_version': {'type': 'integer', 'description': 'Target schema version to migrate to', 'required': True},
                'create_backup': {'type': 'boolean', 'description': 'Whether to create backup before migration (default: True)', 'default': True}
            }
        },
        'cleanup_performance_logs': {
            'function': cleanup_performance_logs,
            'description': 'Clean up old performance logs',
            'parameters': {
                'days_to_keep': {'type': 'integer', 'description': 'Number of days of logs to keep (default: 30)', 'default': 30}
            }
        },
        'get_database_health': {
            'function': get_database_health,
            'description': 'Get comprehensive database health report including statistics and integrity',
            'parameters': {}
        },
        'execute_atomic_bulk_operation': {
            'function': execute_atomic_bulk_operation,
            'description': 'Execute multiple database operations atomically',
            'parameters': {
                'operations': {'type': 'array', 'description': 'List of operation dictionaries with type, table, data, and conditions', 'required': True},
                'operation_name': {'type': 'string', 'description': 'Name for the operation (for performance tracking)', 'default': 'bulk_operation'}
            }
        }
    }