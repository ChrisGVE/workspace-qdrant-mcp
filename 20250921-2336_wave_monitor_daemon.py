#!/usr/bin/env python3
"""
Next Wave Deployment Monitor
Task Orchestrator - Continuous Monitoring for Task 267 Completion

Monitors task dependencies and triggers immediate agent deployment
when blocking tasks complete.
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class WaveMonitorDaemon:
    """Monitors task completion and triggers next wave deployment"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.monitor_running = False
        self.deployment_log = []

    async def check_task_status(self, task_id: str) -> Optional[str]:
        """Check current status of a specific task"""
        try:
            cmd = [
                "task-master", "get_task",
                "--id", task_id,
                "--projectRoot", str(self.project_root)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get("data", {}).get("status")
            return None

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            print(f"❌ Error checking task {task_id}: {e}")
            return None

    async def deploy_immediate_wave(self):
        """Deploy the 4 immediate wave tasks in parallel"""
        print("🚀 DEPLOYING IMMEDIATE WAVE - 4 PARALLEL TASKS")

        wave_tasks = [
            {
                "id": "253",
                "title": "OS-Standard Directory Usage",
                "agent": "system-architect",
                "priority": "medium"
            },
            {
                "id": "254",
                "title": "Embedded Pattern System",
                "agent": "rust-engineer",
                "priority": "medium"
            },
            {
                "id": "256",
                "title": "gRPC Communication Layer",
                "agent": "distributed-systems-engineer",
                "priority": "high"
            },
            {
                "id": "257",
                "title": "SQLite State Management",
                "agent": "database-engineer",
                "priority": "high"
            }
        ]

        # Deploy high priority tasks first
        high_priority = [t for t in wave_tasks if t["priority"] == "high"]
        for task in high_priority:
            await self.deploy_task_agent(task)
            await asyncio.sleep(2)  # Brief delay between deployments

        # Deploy medium priority tasks
        medium_priority = [t for t in wave_tasks if t["priority"] == "medium"]
        for task in medium_priority:
            await self.deploy_task_agent(task)
            await asyncio.sleep(2)

        # Update task statuses to in-progress
        for task in wave_tasks:
            await self.update_task_status(task["id"], "in-progress")

        self.log_deployment("IMMEDIATE_WAVE", wave_tasks)
        print("✅ IMMEDIATE WAVE DEPLOYED: 4 parallel tasks executing")

        # Start monitoring for second wave trigger (task 254 completion)
        asyncio.create_task(self.monitor_second_wave())

    async def deploy_second_wave(self):
        """Deploy the 2 second wave tasks when task 254 completes"""
        print("🚀 DEPLOYING SECOND WAVE - 2 ADDITIONAL TASKS")

        second_wave_tasks = [
            {
                "id": "255",
                "title": "LSP Integration and Code Intelligence",
                "agent": "lsp-integration-expert",
                "priority": "high"
            },
            {
                "id": "260",
                "title": "Project Detection and Multi-Tenancy",
                "agent": "project-detection-specialist",
                "priority": "medium"
            }
        ]

        for task in second_wave_tasks:
            await self.deploy_task_agent(task)
            await self.update_task_status(task["id"], "in-progress")
            await asyncio.sleep(2)

        self.log_deployment("SECOND_WAVE", second_wave_tasks)
        print("✅ SECOND WAVE DEPLOYED: 2 additional parallel tasks executing")

    async def deploy_task_agent(self, task: Dict):
        """Deploy a specialized agent for a specific task"""
        agent_prompt = f"""
TASK ASSIGNMENT FOR {task['agent'].upper()}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 TASK ID: {task['id']}
🎯 OBJECTIVE: {task['title']}
⚡ PRIORITY: {task['priority'].upper()}

🎯 EXECUTION REQUIREMENTS:
1. Use sequential-thinking to break down the task systematically
2. Read existing codebase context (2000+ lines) before making changes
3. Follow git discipline with atomic commits after each change
4. Test all implementations comprehensively
5. Update task-master progress during implementation
6. Do NOT mark task as done until 100% verified complete

📐 SCOPE BOUNDARIES:
- Focus ONLY on task {task['id']} requirements
- Do NOT modify unrelated system components
- Coordinate with orchestrator for any cross-task dependencies

🔄 PROCESS WORKFLOW:
1. task-master get_task --id={task['id']} (understand requirements)
2. Use sequential-thinking to create implementation plan
3. Read relevant codebase sections for context
4. Implement changes with atomic commits
5. Test each change thoroughly
6. Update task-master with progress notes
7. Verify 100% completion before exiting

📊 REPORTING REQUIREMENTS:
- task-master update_task --id={task['id']} --prompt="progress updates"
- Immediate notification if blocked or needs clarification
- task-master set_task_status --id={task['id']} --status=done ONLY when complete

🚨 CRITICAL REMINDER:
- Use task-master tools for ALL task management
- Follow sequential-thinking for complex problem solving
- Make atomic commits following project git discipline
- Test everything - no exceptions
- Confirm completion with orchestrator before marking done

BEGIN EXECUTION OF TASK {task['id']}: {task['title']}
"""

        print(f"🤖 Deploying {task['agent']} for Task {task['id']}: {task['title']}")

        # In a real deployment, this would spawn a new Claude agent instance
        # For now, we'll create deployment records
        deployment_record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task["id"],
            "agent": task["agent"],
            "prompt": agent_prompt,
            "status": "deployed"
        }

        # Write deployment record to file for manual agent execution
        deployment_file = self.project_root / f"20250921-2336_agent_deployment_{task['id']}.md"
        with open(deployment_file, 'w') as f:
            f.write(f"# Agent Deployment Record - Task {task['id']}\n\n")
            f.write(f"**Agent**: {task['agent']}\n")
            f.write(f"**Task**: {task['title']}\n")
            f.write(f"**Priority**: {task['priority']}\n")
            f.write(f"**Deployed**: {deployment_record['timestamp']}\n\n")
            f.write("## Agent Instructions\n\n")
            f.write(agent_prompt)

        return deployment_record

    async def update_task_status(self, task_id: str, status: str):
        """Update task status in task-master"""
        try:
            cmd = [
                "task-master", "set_task_status",
                "--id", task_id,
                "--status", status,
                "--projectRoot", str(self.project_root)
            ]
            subprocess.run(cmd, check=True, timeout=30)
            print(f"📊 Updated Task {task_id} status: {status}")
        except Exception as e:
            print(f"❌ Failed to update Task {task_id} status: {e}")

    async def monitor_second_wave(self):
        """Monitor for task 254 completion to trigger second wave"""
        print("👁️  Starting second wave monitoring (Task 254 completion)")

        while True:
            status_254 = await self.check_task_status("254")
            if status_254 == "done":
                print("🎯 TRIGGER DETECTED: Task 254 completed!")
                await self.deploy_second_wave()
                break

            await asyncio.sleep(60)  # Check every minute

    async def monitor_task_267(self):
        """Main monitoring loop for task 267 completion"""
        print("👁️  Starting Task 267 completion monitoring...")
        print(f"📍 Project: {self.project_root}")
        print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")

        self.monitor_running = True
        check_count = 0

        while self.monitor_running:
            check_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")

            # Check task 267 status
            status_267 = await self.check_task_status("267")

            if status_267 == "done":
                print(f"🎯 TRIGGER DETECTED at {current_time}!")
                print("🚀 Task 267 completed - deploying immediate wave!")
                await self.deploy_immediate_wave()
                break
            elif status_267:
                print(f"⏳ {current_time} - Check #{check_count}: Task 267 status = {status_267}")
            else:
                print(f"❌ {current_time} - Check #{check_count}: Failed to get Task 267 status")

            # Wait 30 seconds before next check
            await asyncio.sleep(30)

    def log_deployment(self, wave_type: str, tasks: List[Dict]):
        """Log deployment information"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "wave_type": wave_type,
            "tasks_deployed": len(tasks),
            "task_details": tasks
        }
        self.deployment_log.append(log_entry)

        # Write to log file
        log_file = self.project_root / "20250921-2336_wave_deployment_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.deployment_log, f, indent=2)

    def stop_monitoring(self):
        """Stop the monitoring daemon"""
        self.monitor_running = False
        print("🛑 Wave monitor daemon stopped")

async def main():
    """Main entry point for wave monitoring daemon"""
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"

    print("🚀 NEXT WAVE DEPLOYMENT MONITOR STARTING")
    print("=" * 50)

    daemon = WaveMonitorDaemon(project_root)

    try:
        await daemon.monitor_task_267()
    except KeyboardInterrupt:
        print("\n🛑 Monitor interrupted by user")
        daemon.stop_monitoring()
    except Exception as e:
        print(f"❌ Monitor error: {e}")
    finally:
        print("📊 Final deployment log:")
        for entry in daemon.deployment_log:
            print(f"  {entry['timestamp']}: {entry['wave_type']} - {entry['tasks_deployed']} tasks")

if __name__ == "__main__":
    asyncio.run(main())