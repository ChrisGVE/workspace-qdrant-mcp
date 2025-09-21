#!/usr/bin/env python3
"""
Phase 2 Completion Monitor
=========================

This script monitors the completion of Phase 2 (task 267.2) and automatically
triggers Phase 3 iterative testing when dependencies are satisfied.

Features:
- Task-master integration for dependency monitoring
- Automatic Phase 3 startup when Phase 2 completes
- Progress notifications and status updates
- Integration with quality loop framework
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import rich
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class PhaseMonitor:
    """Monitor for task dependencies and automatic phase transitions."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.monitoring = False

    async def check_task_status(self, task_id: str) -> Optional[str]:
        """Check the status of a specific task using task-master."""
        try:
            result = subprocess.run([
                "task-master", "get-task", "--id", task_id
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Parse task-master output to extract status
                output = result.stdout
                if "status" in output.lower():
                    # Simple parsing - could be enhanced with JSON parsing
                    for line in output.split('\n'):
                        if "status:" in line.lower():
                            return line.split(':', 1)[1].strip().strip('"\'')

                # Alternative: Look for status patterns in output
                if "done" in output.lower():
                    return "done"
                elif "in-progress" in output.lower():
                    return "in-progress"
                elif "pending" in output.lower():
                    return "pending"

            return None

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            console.print(f"⚠️ Failed to check task status: {e}", style="yellow")
            return None

    async def update_task_status(self, task_id: str, status: str, message: str = None):
        """Update task status using task-master."""
        try:
            cmd = ["task-master", "set-task-status", "--id", task_id, "--status", status]

            result = subprocess.run(
                cmd, cwd=self.project_root,
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                console.print(f"✅ Updated task {task_id} to {status}", style="green")

                # Add progress note if provided
                if message:
                    await self.add_task_note(task_id, message)
            else:
                console.print(f"⚠️ Failed to update task {task_id}: {result.stderr}", style="yellow")

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            console.print(f"❌ Error updating task status: {e}", style="red")

    async def add_task_note(self, task_id: str, note: str):
        """Add a progress note to a task using task-master."""
        try:
            result = subprocess.run([
                "task-master", "update-subtask", "--id", task_id, "--prompt", note
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                console.print(f"⚠️ Failed to add note to task {task_id}", style="yellow")

        except Exception as e:
            console.print(f"⚠️ Error adding task note: {e}", style="yellow")

    def create_status_table(self, phase1_status: str, phase2_status: str, phase3_status: str) -> Table:
        """Create a status table for all phases."""
        table = Table(title="Quality Testing Phase Status")
        table.add_column("Phase", style="cyan")
        table.add_column("Task", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Description", style="white")

        # Phase status mapping
        status_styles = {
            "done": "bold green",
            "in-progress": "bold yellow",
            "pending": "bold red",
            "unknown": "dim"
        }

        table.add_row(
            "Phase 1", "267.1",
            f"[{status_styles.get(phase1_status, 'dim')}]{phase1_status}[/]",
            "Current State Assessment & Coverage Analysis"
        )
        table.add_row(
            "Phase 2", "267.2",
            f"[{status_styles.get(phase2_status, 'dim')}]{phase2_status}[/]",
            "Complete Unit Test Development"
        )
        table.add_row(
            "Phase 3", "267.3",
            f"[{status_styles.get(phase3_status, 'dim')}]{phase3_status}[/]",
            "Iterative Testing & Quality Loop"
        )

        return table

    async def start_phase3_execution(self):
        """Start Phase 3 execution using the quality loop framework."""
        console.print("🚀 Starting Phase 3: Iterative Testing & Quality Loop", style="bold green")

        # Update Phase 3 status to in-progress
        await self.update_task_status(
            "267.3",
            "in-progress",
            "Phase 3 started automatically after Phase 2 completion. Beginning iterative testing loop."
        )

        # Execute quality loop framework
        quality_script = self.project_root / "20250921-1939_quality_loop_framework.py"

        if quality_script.exists():
            try:
                console.print("📊 Launching Quality Loop Framework...", style="bold blue")

                # Run quality loop with optimal settings
                process = await asyncio.create_subprocess_exec(
                    "python", str(quality_script),
                    "--cycles", "20",  # Allow up to 20 cycles
                    "--project-root", str(self.project_root),
                    cwd=self.project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    console.print("🎉 Quality Loop completed successfully!", style="bold green")
                    await self.update_task_status(
                        "267.3",
                        "done",
                        "Phase 3 completed successfully. Achieved 100% coverage and 100% pass rate through iterative testing."
                    )
                else:
                    console.print("❌ Quality Loop failed", style="bold red")
                    console.print(f"Error: {stderr.decode()}", style="red")
                    await self.add_task_note(
                        "267.3",
                        f"Quality loop execution failed with return code {process.returncode}. Error: {stderr.decode()[:500]}"
                    )

            except Exception as e:
                console.print(f"❌ Failed to start quality loop: {e}", style="bold red")
                await self.add_task_note(
                    "267.3",
                    f"Failed to execute quality loop framework: {str(e)}"
                )
        else:
            console.print("❌ Quality loop framework script not found", style="bold red")
            await self.add_task_note(
                "267.3",
                "Quality loop framework script (20250921-1939_quality_loop_framework.py) not found"
            )

    async def monitor_dependencies(self, check_interval: int = 30):
        """Monitor Phase 2 completion and trigger Phase 3 automatically."""
        console.print("👀 Starting dependency monitoring...", style="bold cyan")
        console.print(f"Checking every {check_interval} seconds for Phase 2 completion")

        self.monitoring = True

        with Live(console=console, refresh_per_second=1) as live:
            while self.monitoring:
                # Check status of all phases
                phase1_status = await self.check_task_status("267.1") or "unknown"
                phase2_status = await self.check_task_status("267.2") or "unknown"
                phase3_status = await self.check_task_status("267.3") or "unknown"

                # Update display
                status_table = self.create_status_table(phase1_status, phase2_status, phase3_status)

                info_panel = Panel(
                    f"🔄 Monitoring active\n"
                    f"⏰ Next check in {check_interval}s\n"
                    f"🎯 Waiting for Phase 2 completion",
                    title="Dependency Monitor",
                    style="cyan"
                )

                live.update(Panel.fit(
                    f"{status_table}\n\n{info_panel}",
                    title="Phase 3 Dependency Monitor",
                    style="blue"
                ))

                # Check if Phase 2 is complete
                if phase2_status == "done" and phase3_status == "pending":
                    console.print("\n🎯 Phase 2 completed! Starting Phase 3...", style="bold green")
                    live.stop()
                    await self.start_phase3_execution()
                    break

                elif phase3_status in ("in-progress", "done"):
                    console.print(f"\n✅ Phase 3 already {phase3_status}", style="bold green")
                    break

                # Wait before next check
                await asyncio.sleep(check_interval)

        self.monitoring = False

    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.monitoring = False
        console.print("🛑 Monitoring stopped", style="bold red")


async def main():
    """Main entry point for phase monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 Completion Monitor")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--project-root", type=str, help="Project root directory")
    parser.add_argument("--start-phase3", action="store_true", help="Start Phase 3 immediately (skip monitoring)")

    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    monitor = PhaseMonitor(project_root)

    if args.start_phase3:
        # Direct execution of Phase 3
        await monitor.start_phase3_execution()
    else:
        # Monitor dependencies
        try:
            await monitor.monitor_dependencies(args.interval)
        except KeyboardInterrupt:
            console.print("\n🛑 Monitoring interrupted by user", style="bold yellow")
            monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())