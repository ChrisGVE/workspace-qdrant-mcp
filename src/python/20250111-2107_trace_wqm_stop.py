#!/usr/bin/env python3
"""Trace the execution of wqm service stop to find why unload isn't called."""

import asyncio
import sys
import logging
from pathlib import Path

# Add the CLI to the path
sys.path.insert(0, '/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python')

from wqm_cli.cli.commands.service import ServiceManager

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

async def trace_stop_execution():
    """Trace the execution of the stop command to find where it's failing."""
    print("🔍 TRACING WQM SERVICE STOP EXECUTION")
    print("=" * 60)
    
    # Create a service manager instance
    service_manager = ServiceManager()
    
    print(f"Service manager created for: {service_manager.service_name}")
    print(f"Platform detected: {service_manager.system}")
    
    # Call the stop_service method directly
    print("\n📞 Calling service_manager.stop_service()...")
    try:
        result = await service_manager.stop_service()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(trace_stop_execution())