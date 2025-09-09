"""Service management commands for memexd daemon.

This module implements cross-platform service installation and management
for the memexd daemon with pure daemon architecture and priority-based
resource management.

Commands:
    wqm service install               # Install memexd as user service
    wqm service uninstall             # Remove user service
    wqm service start                 # Start user service
    wqm service stop                  # Stop user service
    wqm service restart               # Restart user service
    wqm service status                # Show user service status
    wqm service logs                  # Show user service logs
    
Note: All services are installed as user-level services only.
"""

import asyncio
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...core.daemon_manager import DaemonManager, get_daemon_manager
from ...observability import get_logger
from ..utils import handle_async_command

# Initialize app and logger
service_app = typer.Typer(name="service", help="User service management")
console = Console()
logger = get_logger(__name__)


class ServiceManager:
    """Cross-platform service manager for memexd daemon."""

    def __init__(self):
        self.system = platform.system().lower()
        self.service_name = "memexd"
        self.daemon_binary = "memexd"  # Priority-based daemon

    async def install_service(
        self,
        config_file: Optional[Path] = None,
        log_level: str = "info",
        auto_start: bool = True,
    ) -> Dict[str, Any]:
        """Install memexd as a user service."""
        try:
            if self.system == "darwin":
                return await self._install_macos_service(
                    config_file, log_level, auto_start
                )
            elif self.system == "linux":
                return await self._install_linux_service(
                    config_file, log_level, auto_start
                )
            elif self.system == "windows":
                return await self._install_windows_service(
                    config_file, log_level, auto_start
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {self.system}",
                    "platform": self.system,
                }
        except Exception as e:
            logger.error("Service installation failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Installation failed: {e}",
                "platform": self.system,
            }

    async def _install_macos_service(
        self,
        config_file: Optional[Path],
        log_level: str,
        auto_start: bool,
    ) -> Dict[str, Any]:
        """Install macOS launchd service."""
        # Find daemon binary
        daemon_path = await self._find_daemon_binary()
        if not daemon_path:
            return {
                "success": False,
                "error": "memexd binary not found. Build it first with: cargo build --release --bin memexd",
            }

        # Create plist content for user service
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_path = plist_dir / f"{service_id}.plist"
        logger.debug(f"Installing user service: {plist_path}")

        # Create plist directory if it doesn't exist and validate permissions
        try:
            plist_dir.mkdir(parents=True, exist_ok=True)
            # Test write permissions by creating a temporary file
            test_file = plist_dir / ".wqm_service_test"
            test_file.touch()
            test_file.unlink()  # Clean up immediately
            logger.debug(f"Directory permissions validated: {plist_dir}")
        except PermissionError as e:
            logger.error(f"Cannot create or write to directory {plist_dir}: {e}")
            suggestion = (
                f"Cannot create user service directory {plist_dir}.\n"
                "This is unexpected for user-level installation.\n"
                "Please check your home directory permissions."
            )
            return {
                "success": False,
                "error": f"Cannot create service directory: {e}",
                "suggestion": suggestion,
                "plist_dir": str(plist_dir),
            }

        # Setup XDG-compliant configuration directory
        # Check XDG_CONFIG_HOME first, fall back to system default
        xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config_home:
            config_dir = Path(xdg_config_home) / "workspace-qdrant"
        else:
            # Fall back to system canonical location
            if self.system == "darwin":
                config_dir = Path.home() / ".config" / "workspace-qdrant"
            elif self.system == "linux":
                config_dir = Path.home() / ".config" / "workspace-qdrant"
            else:  # windows
                config_dir = Path.home() / ".config" / "workspace-qdrant"
        
        config_file_default = config_dir / "workspace_qdrant_config.toml"
        
        # Create XDG config directory if it doesn't exist
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created XDG config directory: {config_dir}")
        except Exception as e:
            logger.error(f"Failed to create XDG config directory: {e}")
            return {
                "success": False,
                "error": f"Cannot create XDG config directory: {e}",
            }
        
        # If no config file specified, use/create default in XDG location
        if not config_file:
            config_file = config_file_default
            
        # Copy default config if target doesn't exist
        if not config_file.exists():
            # Look for default config in project directory first
            project_config = Path.cwd() / "workspace_qdrant_config.toml"
            if project_config.exists():
                import shutil
                shutil.copy2(project_config, config_file)
                logger.debug(f"Copied project config to: {config_file}")
            else:
                # Create a minimal default config
                default_config = '''# Workspace Qdrant MCP Configuration
# TOML format for memexd daemon

# Logging configuration
log_file = "/Users/{}/Library/Logs/memexd.log"

# Processing engine configuration
max_concurrent_tasks = 4
default_timeout_ms = 30000
enable_preemption = true
chunk_size = 1000
enable_lsp = true
log_level = "info"
enable_metrics = true
metrics_interval_secs = 60

# Auto-ingestion configuration
[auto_ingestion]
enabled = true
auto_create_watches = true
include_common_files = true
include_source_files = true
target_collection_suffix = "scratchbook"
max_files_per_batch = 5
batch_delay_seconds = 2.0
max_file_size_mb = 50
recursive_depth = 5
debounce_seconds = 10

# Workspace directory (auto-detected from current working directory)
project_path = "{}"
'''.format(Path.home().name, Path.cwd())
                
                config_file.write_text(default_config)
                logger.debug(f"Created default config: {config_file}")

        # Build daemon arguments
        daemon_args = [str(daemon_path)]
        # Always include config file
        daemon_args.extend(["--config", str(config_file)])
        daemon_args.extend(["--log-level", log_level])
        # Add foreground mode for user services (launchd manages the process)
        daemon_args.append("--foreground")

        # Create plist content with priority-based resource management
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{service_id}</string>
    
    <key>ProgramArguments</key>
    <array>
        {chr(10).join(f"        <string>{arg}</string>" for arg in daemon_args)}
    </array>
    
    <key>RunAtLoad</key>
    <{"true" if auto_start else "false"}/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    
    <key>ProcessType</key>
    <string>Background</string>
    
    <key>Nice</key>
    <integer>5</integer>
    
    <key>LowPriorityIO</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>{self._get_log_path("memexd.log")}</string>
    
    <key>StandardErrorPath</key>
    <string>{self._get_log_path("memexd.error.log")}</string>
    
    <key>WorkingDirectory</key>
    <string>/tmp</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>RUST_LOG</key>
        <string>{log_level}</string>
        <key>MEMEXD_PRIORITY_MODE</key>
        <string>enabled</string>
        <key>MEMEXD_HIGH_PRIORITY_QUEUE_SIZE</key>
        <string>100</string>
        <key>MEMEXD_LOW_PRIORITY_QUEUE_SIZE</key>
        <string>1000</string>
        <key>WQM_LOG_FILE</key>
        <string>false</string>
    </dict>
    
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>4096</integer>
        <key>NumberOfProcesses</key>
        <integer>100</integer>
    </dict>
    
    <key>ThrottleInterval</key>
    <integer>10</integer>
    
    <key>ExitTimeOut</key>
    <integer>30</integer>
</dict>
</plist>"""

        # Write plist file
        try:
            plist_path.write_text(plist_content)
        except PermissionError:
            suggestion = (
                "Permission denied writing to user directory. This is unexpected.\n"
                "Please check if ~/Library/LaunchAgents/ is writable:\n"
                f"  ls -la {plist_dir.parent}"
            )
            return {
                "success": False,
                "error": f"Permission denied writing to {plist_path}.",
                "suggestion": suggestion,
                "plist_path": str(plist_path),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write plist file: {e}",
                "plist_path": str(plist_path),
            }

        # Load service
        try:
            cmd = ["launchctl", "load", str(plist_path)]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Failed to load service: {stderr.decode()}",
                    "plist_path": str(plist_path),
                }

            return {
                "success": True,
                "service_id": service_id,
                "plist_path": str(plist_path),
                "daemon_path": str(daemon_path),
                "auto_start": auto_start,
                "message": f"Service {service_id} installed successfully",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load service: {e}",
                "plist_path": str(plist_path),
            }

    async def _install_linux_service(
        self,
        config_file: Optional[Path],
        log_level: str,
        auto_start: bool,
    ) -> Dict[str, Any]:
        """Install Linux systemd service."""
        # Find daemon binary
        daemon_path = await self._find_daemon_binary()
        if not daemon_path:
            return {
                "success": False,
                "error": "memexd binary not found. Build it first with: cargo build --release --bin memexd",
            }

        service_name = f"{self.service_name}.service"
        
        # User service location
        service_dir = Path.home() / ".config" / "systemd" / "user"
        service_path = service_dir / service_name
        systemctl_args = ["systemctl", "--user"]

        # Create service directory if it doesn't exist
        service_dir.mkdir(parents=True, exist_ok=True)

        # Build daemon arguments
        exec_start = str(daemon_path)
        if config_file:
            exec_start += f" --config {config_file}"
        exec_start += f" --log-level {log_level}"

        # Create systemd service file with priority-based resource management
        service_content = f"""[Unit]
Description=Memory eXchange Daemon - Document processing and embedding service
Documentation=https://github.com/workspace-qdrant-mcp
After=network.target qdrant.service
Wants=network.target

[Service]
Type=simple
ExecStart={exec_start}
ExecReload=/bin/kill -HUP $MAINPID
ExecStop=/bin/kill -TERM $MAINPID
TimeoutStopSec=30
Restart=on-failure
RestartSec=5
StartLimitBurst=3
StartLimitInterval=60

# Resource management for priority-based processing
Nice=5
IOSchedulingClass=2
IOSchedulingPriority=7
CPUSchedulingPolicy=0

# Resource limits
LimitNOFILE=4096
LimitNPROC=100
MemoryHigh=1G
MemoryMax=2G

# Security and isolation
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/tmp /var/log

# Environment variables
Environment=RUST_LOG={log_level}
Environment=MEMEXD_PRIORITY_MODE=enabled
Environment=MEMEXD_HIGH_PRIORITY_QUEUE_SIZE=100
Environment=MEMEXD_LOW_PRIORITY_QUEUE_SIZE=1000
Environment=MEMEXD_RESOURCE_THROTTLE=enabled

# Working directory
WorkingDirectory=/tmp

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=memexd

[Install]
WantedBy=default.target
"""

        # Write service file
        try:
            service_path.write_text(service_content)

            # Reload systemd
            cmd = systemctl_args + ["daemon-reload"]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await result.communicate()

            # Enable service if auto_start is requested
            if auto_start:
                cmd = systemctl_args + ["enable", service_name]
                result = await asyncio.create_subprocess_exec(
                    *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = await result.communicate()

                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Failed to enable service: {stderr.decode()}",
                        "service_path": str(service_path),
                    }

            return {
                "success": True,
                "service_name": service_name,
                "service_path": str(service_path),
                "daemon_path": str(daemon_path),
                "auto_start": auto_start,
                "message": f"Service {service_name} installed successfully",
            }

        except PermissionError:
            suggestion = (
                "Permission denied writing to user directory. This is unexpected.\n"
                "Please check if ~/.config/systemd/user/ is writable:\n"
                f"  ls -la {service_dir.parent}"
            )
            return {
                "success": False,
                "error": f"Permission denied writing to {service_path}.",
                "suggestion": suggestion,
                "service_path": str(service_path),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to install service: {e}",
                "service_path": str(service_path),
            }

    async def _install_windows_service(
        self, config_file: Optional[Path], log_level: str, auto_start: bool
    ) -> Dict[str, Any]:
        """Install Windows service."""
        # TODO: Implement Windows service installation
        return {
            "success": False,
            "error": "Windows service installation not yet implemented",
            "platform": "windows",
        }

    async def uninstall_service(self) -> Dict[str, Any]:
        """Uninstall user service."""
        try:
            if self.system == "darwin":
                return await self._uninstall_macos_service()
            elif self.system == "linux":
                return await self._uninstall_linux_service()
            elif self.system == "windows":
                return await self._uninstall_windows_service()
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {self.system}",
                }
        except Exception as e:
            return {"success": False, "error": f"Uninstallation failed: {e}"}

    async def _uninstall_macos_service(self) -> Dict[str, Any]:
        """Uninstall macOS service."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"
        plist_path = (
            Path.home() / "Library" / "LaunchAgents" / f"{service_id}.plist"
        )

        if not plist_path.exists():
            return {"success": False, "error": f"Service not found at {plist_path}"}

        # Unload service
        try:
            cmd = ["launchctl", "unload", str(plist_path)]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await result.communicate()

            # Remove plist file
            plist_path.unlink()

            return {
                "success": True,
                "service_id": service_id,
                "plist_path": str(plist_path),
                "message": f"Service {service_id} uninstalled successfully",
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to uninstall service: {e}"}

    async def _uninstall_linux_service(self) -> Dict[str, Any]:
        """Uninstall Linux service."""
        service_name = f"{self.service_name}.service"
        service_path = Path.home() / ".config" / "systemd" / "user" / service_name
        systemctl_args = ["systemctl", "--user"]

        if not service_path.exists():
            return {"success": False, "error": f"Service not found at {service_path}"}

        try:
            # Stop and disable service
            for action in ["stop", "disable"]:
                cmd = systemctl_args + [action, service_name]
                result = await asyncio.create_subprocess_exec(
                    *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                await result.communicate()

            # Remove service file
            service_path.unlink()

            # Reload systemd
            cmd = systemctl_args + ["daemon-reload"]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await result.communicate()

            return {
                "success": True,
                "service_name": service_name,
                "service_path": str(service_path),
                "message": f"Service {service_name} uninstalled successfully",
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to uninstall service: {e}"}

    async def _uninstall_windows_service(self) -> Dict[str, Any]:
        """Uninstall Windows service."""
        return {
            "success": False,
            "error": "Windows service uninstallation not yet implemented",
        }

    async def start_service(self) -> Dict[str, Any]:
        """Start the user service."""
        try:
            if self.system == "darwin":
                return await self._start_macos_service()
            elif self.system == "linux":
                return await self._start_linux_service()
            elif self.system == "windows":
                return await self._start_windows_service()
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {self.system}",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to start service: {e}"}

    async def _start_macos_service(self) -> Dict[str, Any]:
        """Start macOS service."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"

        cmd = ["launchctl", "start", service_id]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Failed to start service: {stderr.decode()}",
            }

        return {
            "success": True,
            "service_id": service_id,
            "message": f"Service {service_id} started successfully",
        }

    async def _start_linux_service(self) -> Dict[str, Any]:
        """Start Linux service."""
        service_name = f"{self.service_name}.service"
        systemctl_args = ["systemctl", "--user"]

        cmd = systemctl_args + ["start", service_name]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Failed to start service: {stderr.decode()}",
            }

        return {
            "success": True,
            "service_name": service_name,
            "message": f"Service {service_name} started successfully",
        }

    async def _start_windows_service(self) -> Dict[str, Any]:
        """Start Windows service."""
        return {"success": False, "error": "Windows service start not yet implemented"}

    async def stop_service(self) -> Dict[str, Any]:
        """Stop the user service."""
        try:
            if self.system == "darwin":
                return await self._stop_macos_service()
            elif self.system == "linux":
                return await self._stop_linux_service()
            elif self.system == "windows":
                return await self._stop_windows_service()
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {self.system}",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to stop service: {e}"}

    async def _stop_macos_service(self) -> Dict[str, Any]:
        """Stop macOS service."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"

        cmd = ["launchctl", "stop", service_id]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        # launchctl stop may return non-zero even on success
        return {
            "success": True,
            "service_id": service_id,
            "message": f"Service {service_id} stop command sent",
        }

    async def _stop_linux_service(self) -> Dict[str, Any]:
        """Stop Linux service."""
        service_name = f"{self.service_name}.service"
        systemctl_args = ["systemctl", "--user"]

        cmd = systemctl_args + ["stop", service_name]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Failed to stop service: {stderr.decode()}",
            }

        return {
            "success": True,
            "service_name": service_name,
            "message": f"Service {service_name} stopped successfully",
        }

    async def _stop_windows_service(self) -> Dict[str, Any]:
        """Stop Windows service."""
        return {"success": False, "error": "Windows service stop not yet implemented"}

    async def get_service_status(self) -> Dict[str, Any]:
        """Get user service status."""
        try:
            if self.system == "darwin":
                return await self._get_macos_service_status()
            elif self.system == "linux":
                return await self._get_linux_service_status()
            elif self.system == "windows":
                return await self._get_windows_service_status()
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {self.system}",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to get service status: {e}"}

    async def _get_macos_service_status(self) -> Dict[str, Any]:
        """Get macOS service status."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"

        # Check if service is loaded
        cmd = ["launchctl", "list"]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        output = stdout.decode()
        service_loaded = service_id in output

        # Get service info if loaded
        status = "unknown"
        pid = None
        if service_loaded:
            # Extract PID and status from launchctl list output
            for line in output.split("\n"):
                if service_id in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            pid_str = parts[0]
                            if pid_str != "-":
                                pid = int(pid_str)
                                status = "running"
                            else:
                                status = "stopped"
                        except ValueError:
                            status = "error"
                    break

        return {
            "success": True,
            "service_id": service_id,
            "status": "loaded" if service_loaded else "not_loaded",
            "running": status == "running",
            "pid": pid,
            "platform": "macOS",
        }

    async def _get_linux_service_status(self) -> Dict[str, Any]:
        """Get Linux service status."""
        service_name = f"{self.service_name}.service"
        systemctl_args = ["systemctl", "--user"]

        # Get service status
        cmd = systemctl_args + ["is-active", service_name]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        is_active = stdout.decode().strip() == "active"

        # Get detailed status
        cmd = systemctl_args + ["status", service_name, "--no-pager", "--lines=0"]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        status_output = stdout.decode()

        # Parse status information
        status = "unknown"
        pid = None
        enabled = False

        if "active (running)" in status_output:
            status = "running"
        elif "inactive (dead)" in status_output:
            status = "stopped"
        elif "failed" in status_output:
            status = "failed"

        # Extract PID
        import re

        pid_match = re.search(r"Main PID: (\d+)", status_output)
        if pid_match:
            pid = int(pid_match.group(1))

        # Check if enabled
        cmd = systemctl_args + ["is-enabled", service_name]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        enabled = stdout.decode().strip() == "enabled"

        return {
            "success": True,
            "service_name": service_name,
            "status": status,
            "running": is_active,
            "pid": pid,
            "enabled": enabled,
            "platform": "Linux",
        }

    async def _get_windows_service_status(self) -> Dict[str, Any]:
        """Get Windows service status."""
        return {"success": False, "error": "Windows service status not yet implemented"}

    async def get_service_logs(
        self, lines: int = 50
    ) -> Dict[str, Any]:
        """Get user service logs."""
        try:
            if self.system == "darwin":
                return await self._get_macos_service_logs(lines)
            elif self.system == "linux":
                return await self._get_linux_service_logs(lines)
            elif self.system == "windows":
                return await self._get_windows_service_logs(lines)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {self.system}",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to get service logs: {e}"}

    async def _get_macos_service_logs(self, lines: int) -> Dict[str, Any]:
        """Get macOS service logs."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"

        # Try to read log files from user directory
        user_log_dir = Path.home() / "Library" / "Logs"
        log_files = [
            str(user_log_dir / "memexd.log"),
            str(user_log_dir / "memexd.error.log")
        ]
        logs = []

        for log_file in log_files:
            log_path = Path(log_file)
            if log_path.exists():
                try:
                    with open(log_path, "r") as f:
                        content = f.readlines()[-lines:]
                    logs.extend([f"[{log_file}] {line.rstrip()}" for line in content])
                except Exception as e:
                    logs.append(f"[{log_file}] Error reading log: {e}")

        if not logs:
            # Try system log
            cmd = [
                "log",
                "show",
                "--predicate",
                f"subsystem == '{service_id}'",
                "--last",
                f"{lines}",
            ]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logs = stdout.decode().split("\n")
            else:
                logs = ["No logs available"]

        return {
            "success": True,
            "service_id": service_id,
            "logs": logs,
            "lines_requested": lines,
            "platform": "macOS",
        }

    async def _get_linux_service_logs(
        self, lines: int
    ) -> Dict[str, Any]:
        """Get Linux service logs."""
        service_name = f"{self.service_name}.service"
        cmd = ["journalctl", "--user", "-u", service_name, "-n", str(lines), "--no-pager"]

        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            return {"success": False, "error": f"Failed to get logs: {stderr.decode()}"}

        logs = stdout.decode().split("\n")

        return {
            "success": True,
            "service_name": service_name,
            "logs": logs,
            "lines_requested": lines,
            "platform": "Linux",
        }

    async def _get_windows_service_logs(self, lines: int) -> Dict[str, Any]:
        """Get Windows service logs."""
        return {"success": False, "error": "Windows service logs not yet implemented"}

    async def _find_daemon_binary(self) -> Optional[Path]:
        """Find the memexd binary."""
        # Look in common build locations
        project_root = Path.cwd()
        rust_engine_path = project_root / "rust-engine"

        # Check for built binary in target directory
        target_dirs = [
            rust_engine_path / "target" / "release",
            rust_engine_path / "target" / "debug",
        ]

        binary_name = self.daemon_binary
        if self.system == "windows":
            binary_name += ".exe"

        for target_dir in target_dirs:
            binary_path = target_dir / binary_name
            if binary_path.exists():
                return binary_path

        # Check if it's in PATH
        try:
            result = await asyncio.create_subprocess_exec(
                "which" if self.system != "windows" else "where",
                binary_name,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                binary_path = Path(stdout.decode().strip().split("\n")[0])
                if binary_path.exists():
                    return binary_path
        except Exception:
            pass

        return None

    def _get_log_path(self, filename: str) -> str:
        """Get appropriate log path for user service."""
        # User service - use user's log directory
        user_log_dir = Path.home() / "Library" / "Logs"
        user_log_dir.mkdir(parents=True, exist_ok=True)
        return str(user_log_dir / filename)


# Create service manager instance
service_manager = ServiceManager()


@service_app.command("install")
def install_service(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Logging level"),
    auto_start: bool = typer.Option(
        True, "--auto-start/--no-auto-start", help="Start service automatically on boot"
    ),
) -> None:
    """Install memexd as a user service with priority-based resource management."""

    async def _install():
        config_path = Path(config_file) if config_file else None
        result = await service_manager.install_service(
            config_path, log_level, auto_start
        )

        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service installed successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}\n"
                    f"Binary: {result.get('daemon_path', 'memexd')}\n"
                    f"Auto-start: {'Yes' if auto_start else 'No'}\n"
                    f"Service type: User service\n"
                    f"Priority mode: Enabled\n"
                    f"Resource management: Active",
                    title="Service Installation",
                    style="green",
                )
            )

            if auto_start:
                console.print("💡 The service will start automatically on system boot.")
            else:
                console.print(
                    "💡 Use 'wqm service start' to start the service manually."
                )
        else:
            error_text = f"❌ Installation failed!\n\nError: {result['error']}\nPlatform: {result.get('platform', 'unknown')}"
            
            # Add suggestion if available
            if 'suggestion' in result:
                error_text += f"\n\n💡 Suggestion:\n{result['suggestion']}"
            
            console.print(
                Panel.fit(
                    error_text,
                    title="Installation Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_install())


@service_app.command("uninstall")
def uninstall_service(
    force: bool = typer.Option(False, "--force", help="Force uninstallation"),
) -> None:
    """Uninstall memexd user service."""

    async def _uninstall():
        if not force:
            confirm = typer.confirm(
                "Are you sure you want to uninstall the memexd service?"
            )
            if not confirm:
                console.print("Uninstallation cancelled.")
                return

        result = await service_manager.uninstall_service()

        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service uninstalled successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}",
                    title="Service Uninstallation",
                    style="green",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"❌ Uninstallation failed!\n\nError: {result['error']}",
                    title="Uninstallation Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_uninstall())


@service_app.command("start")
def start_service() -> None:
    """Start the memexd user service."""

    async def _start():
        result = await service_manager.start_service()

        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service started successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}\n"
                    f"Priority-based processing: Active\n"
                    f"Resource management: Enabled",
                    title="Service Start",
                    style="green",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"❌ Failed to start service!\n\nError: {result['error']}",
                    title="Start Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_start())


@service_app.command("stop")
def stop_service() -> None:
    """Stop the memexd user service."""

    async def _stop():
        result = await service_manager.stop_service()

        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service stopped successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}",
                    title="Service Stop",
                    style="green",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"❌ Failed to stop service!\n\nError: {result['error']}",
                    title="Stop Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_stop())


@service_app.command("restart")
def restart_service() -> None:
    """Restart the memexd user service."""

    async def _restart():
        # Stop first
        stop_result = await service_manager.stop_service()
        if not stop_result["success"]:
            console.print(f"⚠️  Warning: Failed to stop service: {stop_result['error']}")

        # Wait a moment
        await asyncio.sleep(2)

        # Start
        start_result = await service_manager.start_service()

        if start_result["success"]:
            platform_name = platform.system().lower()
            
            console.print(
                Panel.fit(
                    f"✅ Service restarted successfully!\n\n"
                    f"Service: {start_result.get('service_id', start_result.get('service_name'))}\n"
                    f"Type: User service\n"
                    f"Platform: {platform_name.title()}\n"
                    f"Configuration: Active and loaded\n"
                    f"Priority-based processing: Enabled",
                    title="Service Restart Complete",
                    style="green",
                )
            )
            
            # Show additional help for configuration changes
            console.print("\nService is now running with updated configuration.")
            console.print("Use 'wqm service status' to check service health.")
        else:
            console.print(
                Panel.fit(
                    f"❌ Failed to restart service!\n\n"
                    f"Stop: {'Success' if stop_result['success'] else 'Failed'}\n"
                    f"Start: Failed - {start_result['error']}\n\n"
                    f"Try running 'wqm service status' to diagnose issues.",
                    title="Restart Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_restart())


@service_app.command("status")
def get_status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status"),
) -> None:
    """Show memexd user service status."""

    async def _status():
        result = await service_manager.get_service_status()

        if result["success"]:
            service_name = result.get(
                "service_id", result.get("service_name", "memexd")
            )
            status = result.get("status", "unknown")
            running = result.get("running", False)
            pid = result.get("pid")
            platform = result.get("platform", service_manager.system.title())

            # Create status table
            table = Table(title=f"Service Status - {service_name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            # Status with color coding
            if running:
                status_text = Text("Running", style="green")
            elif status == "stopped":
                status_text = Text("Stopped", style="yellow")
            elif status == "failed":
                status_text = Text("Failed", style="red")
            else:
                status_text = Text(status.title(), style="dim")

            table.add_row("Status", str(status_text))
            table.add_row("Platform", platform)

            if pid:
                table.add_row("PID", str(pid))

            if "enabled" in result:
                table.add_row(
                    "Auto-start", "Enabled" if result["enabled"] else "Disabled"
                )

            # Add priority and resource management info
            table.add_row("Priority Mode", "Enabled" if running else "N/A")
            table.add_row("Resource Management", "Active" if running else "N/A")

            console.print(table)

            if verbose and running:
                console.print(
                    "\n💡 Service is running with priority-based resource management:"
                )
                console.print(
                    "   • High-priority queue: MCP operations and current project"
                )
                console.print("   • Low-priority queue: Background folder ingestion")
                console.print(
                    "   • Resource throttling: Enabled for low-priority tasks"
                )
        else:
            console.print(
                Panel.fit(
                    f"❌ Failed to get service status!\n\nError: {result['error']}",
                    title="Status Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_status())


@service_app.command("logs")
def get_logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of log lines to show"),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow logs in real-time"
    ),
) -> None:
    """Show memexd user service logs."""

    async def _logs():
        result = await service_manager.get_service_logs(lines)

        if result["success"]:
            service_name = result.get(
                "service_id", result.get("service_name", "memexd")
            )
            logs = result.get("logs", [])

            console.print(
                Panel.fit(
                    f"Showing last {len(logs)} lines for {service_name}",
                    title="Service Logs",
                    style="blue",
                )
            )

            for log_line in logs:
                if log_line.strip():  # Skip empty lines
                    console.print(log_line)

            if follow:
                console.print("\n[dim]Following logs... Press Ctrl+C to stop[/dim]")
                # TODO: Implement log following
        else:
            console.print(
                Panel.fit(
                    f"❌ Failed to get service logs!\n\nError: {result['error']}",
                    title="Logs Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_logs())


if __name__ == "__main__":
    service_app()
