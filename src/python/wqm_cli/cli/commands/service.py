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

from common.core.daemon_manager import DaemonManager, get_daemon_manager
from common.observability import get_logger
from common.utils.project_detection import ProjectDetector
from ..utils import create_command_app, handle_async_command

# Initialize app and logger
service_app = create_command_app(
    name="service",
    help_text="User service management for memexd daemon",
    no_args_is_help=True,
)
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
        # Add launchd-specific PID file to avoid conflicts with manual instances
        launchd_pid_file = f"/tmp/memexd-launchd.pid"
        daemon_args.extend(["--pid-file", launchd_pid_file])
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
        # Find daemon binary
        daemon_path = await self._find_daemon_binary()
        if not daemon_path:
            return {
                "success": False,
                "error": "memexd.exe binary not found. Build it first with: cargo build --release --bin memexd",
            }

        service_name = f"memexd-{self.service_name}"
        
        # Setup Windows-appropriate configuration directory
        import os
        local_appdata = os.environ.get('LOCALAPPDATA', str(Path.home() / "AppData" / "Local"))
        config_dir = Path(local_appdata) / "workspace-qdrant"
        config_file_default = config_dir / "workspace_qdrant_config.toml"
        
        # Create config directory if it doesn't exist
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created Windows config directory: {config_dir}")
        except Exception as e:
            logger.error(f"Failed to create Windows config directory: {e}")
            return {
                "success": False,
                "error": f"Cannot create config directory: {e}",
            }
        
        # If no config file specified, use/create default in Windows location
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
                # Create a minimal default config for Windows
                default_config = f'''# Workspace Qdrant MCP Configuration
# TOML format for memexd daemon

# Logging configuration (Windows paths)
log_file = "{self._get_log_path("memexd.log").replace(chr(92), chr(92) + chr(92))}"

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
project_path = "{str(Path.cwd()).replace(chr(92), chr(92) + chr(92))}"
'''
                
                config_file.write_text(default_config)
                logger.debug(f"Created default Windows config: {config_file}")

        # Build daemon arguments
        daemon_args = [str(daemon_path)]
        daemon_args.extend(["--config", str(config_file)])
        daemon_args.extend(["--log-level", log_level])
        # Windows service-specific PID file
        windows_pid_file = str(Path(os.environ.get('TEMP', 'C:\\temp')) / "memexd-service.pid")
        daemon_args.extend(["--pid-file", windows_pid_file])

        # Build service command string for Windows
        service_cmd = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in daemon_args)
        
        try:
            # Check if service already exists
            check_cmd = ["sc", "query", service_name]
            result = await asyncio.create_subprocess_exec(
                *check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return {
                    "success": False,
                    "error": f"Service {service_name} already exists. Uninstall it first.",
                    "service_name": service_name,
                }

            # Create Windows service using sc create
            create_cmd = [
                "sc", "create", service_name,
                f"binPath={service_cmd}",
                "type=own",
                "start=demand",  # Manual start by default, enable if auto_start
                f"DisplayName=Memory eXchange Daemon ({self.service_name})",
                "obj=LocalSystem"  # Run as LocalSystem for user services
            ]
            
            result = await asyncio.create_subprocess_exec(
                *create_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                return {
                    "success": False,
                    "error": f"Failed to create Windows service: {error_msg}",
                    "service_name": service_name,
                }

            # Configure service description
            desc_cmd = [
                "sc", "description", service_name,
                "Document processing and embedding service with priority-based resource management"
            ]
            await asyncio.create_subprocess_exec(*desc_cmd)

            # Set service to auto-start if requested
            if auto_start:
                auto_cmd = ["sc", "config", service_name, "start=auto"]
                result = await asyncio.create_subprocess_exec(
                    *auto_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode != 0:
                    logger.warning(f"Failed to set auto-start: {stderr.decode()}")

            return {
                "success": True,
                "service_name": service_name,
                "daemon_path": str(daemon_path),
                "config_file": str(config_file),
                "auto_start": auto_start,
                "message": f"Windows service {service_name} installed successfully",
            }

        except Exception as e:
            logger.error("Windows service installation failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Installation failed: {e}",
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
        service_name = f"memexd-{self.service_name}"
        
        try:
            # Check if service exists
            check_cmd = ["sc", "query", service_name]
            result = await asyncio.create_subprocess_exec(
                *check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Service {service_name} not found or already uninstalled",
                    "service_name": service_name,
                }

            # Stop service if it's running before uninstalling
            stop_cmd = ["sc", "stop", service_name]
            result = await asyncio.create_subprocess_exec(
                *stop_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await result.communicate()
            # Don't fail if stop fails - service might already be stopped

            # Wait a moment for service to stop
            await asyncio.sleep(2)

            # Delete the service
            delete_cmd = ["sc", "delete", service_name]
            result = await asyncio.create_subprocess_exec(
                *delete_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                return {
                    "success": False,
                    "error": f"Failed to delete Windows service: {error_msg}",
                    "service_name": service_name,
                }

            # Clean up Windows service-specific PID file if it exists
            import os
            windows_pid_file = Path(os.environ.get('TEMP', 'C:\\temp')) / "memexd-service.pid"
            if windows_pid_file.exists():
                try:
                    windows_pid_file.unlink()
                    logger.debug(f"Removed Windows service PID file: {windows_pid_file}")
                except OSError as e:
                    logger.warning(f"Could not remove PID file: {e}")

            return {
                "success": True,
                "service_name": service_name,
                "message": f"Windows service {service_name} uninstalled successfully",
            }

        except Exception as e:
            logger.error("Windows service uninstallation failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Uninstallation failed: {e}",
                "service_name": service_name,
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
        """Start macOS service with comprehensive cleanup and validation."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"

        try:
            # First check if service is installed
            status_check = await self._get_macos_service_status()
            if not status_check.get("loaded", False):
                return {
                    "success": False,
                    "error": "Service is not installed",
                    "suggestion": "Run 'wqm service install' first to install the service",
                    "help_command": "wqm service install",
                }

            # Step 1: Clean up any stale resources before starting
            await self._cleanup_service_resources()

            # Step 2: Give resources time to fully clean up
            await asyncio.sleep(1)

            # Step 3: Check if service is loaded, if not load it first
            plist_dir = Path.home() / "Library" / "LaunchAgents"
            plist_path = plist_dir / f"{service_id}.plist"
            
            # Try to start first, if it fails because service is unloaded, load it
            cmd = ["launchctl", "start", service_id]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            # If start failed and plist exists, try loading first then starting
            if result.returncode != 0 and plist_path.exists():
                error_msg = stderr.decode().strip()
                if "Could not find specified service" in error_msg or "service not found" in error_msg.lower():
                    # Service is unloaded, need to load it first
                    logger.debug("Service unloaded, loading first...")
                    load_cmd = ["launchctl", "load", str(plist_path)]
                    load_result = await asyncio.create_subprocess_exec(
                        *load_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    load_stdout, load_stderr = await load_result.communicate()
                    
                    if load_result.returncode == 0:
                        # Loading succeeded, service should now be running due to KeepAlive
                        # Give it a moment to start
                        await asyncio.sleep(2)
                    else:
                        return {
                            "success": False,
                            "error": f"Failed to load service: {load_stderr.decode().strip()}",
                            "service_id": service_id,
                        }
                else:
                    # Different error, not related to unloaded service
                    pass

            if result.returncode != 0 and not ("Could not find specified service" in stderr.decode() or "service not found" in stderr.decode().lower()):
                error_msg = stderr.decode().strip()
                # Enhanced error reporting with diagnostic information
                logger.debug(f"launchctl start failed: {error_msg}")
                
                # Check if the service is actually running despite the error
                status_check_after = await self._get_macos_service_status()
                if status_check_after.get("status") == "running":
                    return {
                        "success": True,
                        "service_id": service_id,
                        "message": f"Service {service_id} is running (launchctl reported error but service is active)",
                        "warning": f"launchctl start returned error but service is running: {error_msg}"
                    }
                
                # Provide better error messages based on common failure scenarios
                if result.returncode == 3:
                    return {
                        "success": False,
                        "error": "Service failed to start - service definition not found",
                        "suggestion": "Try reinstalling the service with 'wqm service uninstall' followed by 'wqm service install'",
                        "technical_details": f"launchctl returned: {error_msg}",
                    }
                elif result.returncode == 5:
                    return {
                        "success": False,
                        "error": "Service failed to start - input/output error",
                        "suggestion": "This usually indicates a configuration problem. Check that the daemon binary exists and is executable",
                        "technical_details": f"launchctl returned: {error_msg}",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Service failed to start (exit code {result.returncode})",
                        "suggestion": "Check service logs with 'wqm service logs' for more details",
                        "technical_details": error_msg if error_msg else "No additional error message",
                    }

            # Step 4: Verify the service actually started
            await asyncio.sleep(2)  # Give service time to initialize
            status_check = await self._get_macos_service_status()
            
            if status_check.get("status") == "running":
                return {
                    "success": True,
                    "service_id": service_id,
                    "message": f"Service {service_id} started successfully and is running",
                    "status": status_check
                }
            else:
                return {
                    "success": False,
                    "error": f"Service started but is not running. Status: {status_check.get('status', 'unknown')}",
                    "debug_info": status_check
                }

        except Exception as e:
            logger.error(f"Exception in _start_macos_service: {e}")
            return {
                "success": False,
                "error": f"Exception starting service: {e}",
            }

    async def _cleanup_service_resources(self) -> None:
        """Clean up stale service resources before starting."""
        try:
            # Clean up any stale PID files
            pid_files = ["/tmp/memexd.pid", "/tmp/memexd-launchd.pid", "/tmp/memexd-manual.pid"]
            for pid_file in pid_files:
                if Path(pid_file).exists():
                    try:
                        # Check if the PID in the file is still a running memexd process
                        with open(pid_file, 'r') as f:
                            pid_str = f.read().strip()
                            if pid_str.isdigit():
                                pid = int(pid_str)
                                
                                # Check if process exists and is memexd
                                check_cmd = ["ps", "-p", str(pid), "-o", "comm="]
                                result = await asyncio.create_subprocess_exec(
                                    *check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                                )
                                stdout, _ = await result.communicate()
                                
                                if result.returncode == 0 and b"memexd" in stdout:
                                    logger.info(f"Found running memexd process {pid}, killing it")
                                    kill_cmd = ["kill", "-TERM", str(pid)]
                                    await asyncio.create_subprocess_exec(*kill_cmd)
                                    await asyncio.sleep(1)  # Give it time to terminate
                                
                                # Remove the stale PID file
                                Path(pid_file).unlink()
                                logger.debug(f"Removed stale PID file: {pid_file}")
                    
                    except (ValueError, OSError) as e:
                        logger.debug(f"Error cleaning PID file {pid_file}: {e}")
                        # Remove malformed PID file
                        try:
                            Path(pid_file).unlink()
                        except OSError:
                            pass

            # Clean up non-launchd memexd processes only
            try:
                # Find all memexd processes
                pgrep_cmd = ["pgrep", "-f", "memexd"]
                result = await asyncio.create_subprocess_exec(
                    *pgrep_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                
                if result.returncode == 0:
                    pids = stdout.decode().strip().split('\n')
                    killed_processes = []
                    
                    for pid_str in pids:
                        if pid_str.strip().isdigit():
                            pid = int(pid_str.strip())
                            
                            # Check if this is a launchd process by examining its arguments
                            ps_cmd = ["ps", "-p", str(pid), "-o", "args="]
                            ps_result = await asyncio.create_subprocess_exec(
                                *ps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                            )
                            ps_stdout, _ = await ps_result.communicate()
                            
                            if ps_result.returncode == 0:
                                args = ps_stdout.decode().strip()
                                # Only kill processes that are NOT using the launchd PID file
                                if "--pid-file /tmp/memexd-launchd.pid" not in args and "memexd" in args:
                                    logger.debug(f"Terminating non-launchd memexd process {pid}: {args[:100]}...")
                                    kill_cmd = ["kill", "-TERM", str(pid)]
                                    await asyncio.create_subprocess_exec(*kill_cmd)
                                    killed_processes.append(pid)
                                else:
                                    logger.debug(f"Preserving launchd memexd process {pid}")
                    
                    if killed_processes:
                        logger.info(f"Terminated {len(killed_processes)} non-launchd memexd processes: {killed_processes}")
                        # Give processes time to clean up
                        await asyncio.sleep(2)
                
            except Exception as e:
                logger.debug(f"Error cleaning up non-launchd processes: {e}")

        except Exception as e:
            logger.error(f"Error in resource cleanup: {e}")

    async def _start_linux_service(self) -> Dict[str, Any]:
        """Start Linux service."""
        service_name = f"{self.service_name}.service"
        systemctl_args = ["systemctl", "--user"]

        # First check if service is installed
        status_check = await self._get_linux_service_status()
        if not status_check.get("success", False) or "not found" in status_check.get("error", "").lower():
            return {
                "success": False,
                "error": "Service is not installed",
                "suggestion": "Run 'wqm service install' first to install the service",
                "help_command": "wqm service install",
            }

        cmd = systemctl_args + ["start", service_name]
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            error_msg = stderr.decode().strip()
            
            # Provide better error messages based on common systemd errors
            if "not found" in error_msg.lower() or "could not find" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Service is not installed or not found",
                    "suggestion": "Run 'wqm service install' first to install the service",
                    "help_command": "wqm service install",
                    "technical_details": error_msg,
                }
            elif "permission denied" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Permission denied when starting service",
                    "suggestion": "Make sure you have permission to start user services",
                    "technical_details": error_msg,
                }
            else:
                return {
                    "success": False,
                    "error": f"Service failed to start (exit code {result.returncode})",
                    "suggestion": "Check service logs with 'wqm service logs' for more details",
                    "technical_details": error_msg,
                }

        return {
            "success": True,
            "service_name": service_name,
            "message": f"Service {service_name} started successfully",
        }

    async def _start_windows_service(self) -> Dict[str, Any]:
        """Start Windows service."""
        service_name = f"memexd-{self.service_name}"
        
        try:
            # First check if service is installed
            status_check = await self._get_windows_service_status()
            if not status_check.get("success", False):
                return {
                    "success": False,
                    "error": "Service is not installed",
                    "suggestion": "Run 'wqm service install' first to install the service",
                    "help_command": "wqm service install",
                }

            # Check if service is already running
            if status_check.get("status") == "running":
                return {
                    "success": True,
                    "service_name": service_name,
                    "message": f"Service {service_name} is already running",
                    "warning": "Service was already started",
                }

            # Start the service
            start_cmd = ["sc", "start", service_name]
            result = await asyncio.create_subprocess_exec(
                *start_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                
                # Provide better error messages based on common Windows errors
                if "1056" in error_msg or "already running" in error_msg.lower():
                    return {
                        "success": True,
                        "service_name": service_name,
                        "message": f"Service {service_name} is already running",
                        "warning": "Service was already started",
                    }
                elif "1053" in error_msg:
                    return {
                        "success": False,
                        "error": "Service failed to respond to start request in timely fashion",
                        "suggestion": "Check service logs and ensure daemon binary is executable",
                        "technical_details": error_msg,
                    }
                elif "2" in error_msg or "not found" in error_msg.lower():
                    return {
                        "success": False,
                        "error": "Service is not installed or not found",
                        "suggestion": "Run 'wqm service install' first to install the service",
                        "help_command": "wqm service install",
                        "technical_details": error_msg,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Service failed to start (exit code {result.returncode})",
                        "suggestion": "Check service logs with 'wqm service logs' for more details",
                        "technical_details": error_msg,
                    }

            # Verify the service actually started by waiting and checking status
            await asyncio.sleep(3)  # Give service time to initialize
            status_check = await self._get_windows_service_status()
            
            if status_check.get("status") == "running":
                return {
                    "success": True,
                    "service_name": service_name,
                    "message": f"Service {service_name} started successfully and is running",
                    "status": status_check
                }
            else:
                return {
                    "success": False,
                    "error": f"Service started but is not running. Status: {status_check.get('status', 'unknown')}",
                    "debug_info": status_check
                }

        except Exception as e:
            logger.error(f"Exception in _start_windows_service: {e}")
            return {
                "success": False,
                "error": f"Exception starting service: {e}",
                "service_name": service_name,
            }

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
        """Stop macOS service with proper launchd lifecycle management."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_path = plist_dir / f"{service_id}.plist"

        try:
            # Step 1: Unload the service to properly stop it (prevents KeepAlive restart)
            if plist_path.exists():
                cmd = ["launchctl", "unload", str(plist_path)]
                result = await asyncio.create_subprocess_exec(
                    *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                logger.debug(f"Unload result - stdout: {stdout.decode()}, stderr: {stderr.decode()}")
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Failed to unload service: {stderr.decode().strip()}",
                        "service_id": service_id,
                    }
                
                # Step 2: Wait for processes to stop gracefully
                await asyncio.sleep(3)
                
                # Step 3: Clean up PID files
                await self._cleanup_pid_files()
                
                # Step 4: Final verification - processes should be gone
                final_check = await self._find_memexd_processes()
                if final_check:
                    logger.warning(f"After unload, found {len(final_check)} memexd processes still running")
                    # This shouldn't happen with proper unload, but we need to clean them up
                    for pid in final_check:
                        logger.warning(f"Cleaning up remaining process: {pid}")
                    
                    # Step 5: Force cleanup of remaining processes
                    await self._cleanup_all_memexd_processes()
                    
                    # Final verification after cleanup
                    post_cleanup_check = await self._find_memexd_processes()
                    if post_cleanup_check:
                        logger.error(f"Failed to cleanup {len(post_cleanup_check)} processes: {post_cleanup_check}")
                    else:
                        logger.info("Successfully cleaned up all remaining processes")
                
                return {
                    "success": True,
                    "service_id": service_id,
                    "message": f"Service {service_id} unloaded successfully",
                    "processes_found_after_unload": len(final_check)
                }
            else:
                return {
                    "success": False,
                    "error": f"Service plist not found at {plist_path}",
                    "service_id": service_id,
                }
            
        except Exception as e:
            logger.error(f"Exception in _stop_macos_service: {e}")
            return {
                "success": False,
                "error": f"Exception stopping service: {e}",
                "service_id": service_id,
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
        service_name = f"memexd-{self.service_name}"
        
        try:
            # Check if service exists and get current status
            status_check = await self._get_windows_service_status()
            if not status_check.get("success", False):
                return {
                    "success": False,
                    "error": f"Service {service_name} not found",
                    "service_name": service_name,
                }

            # Check if service is already stopped
            if status_check.get("status") in ["stopped", "stop_pending"]:
                return {
                    "success": True,
                    "service_name": service_name,
                    "message": f"Service {service_name} is already stopped",
                    "warning": "Service was already stopped",
                }

            # Stop the service
            stop_cmd = ["sc", "stop", service_name]
            result = await asyncio.create_subprocess_exec(
                *stop_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                
                # Provide better error messages based on common Windows errors
                if "1062" in error_msg or "not started" in error_msg.lower():
                    return {
                        "success": True,
                        "service_name": service_name,
                        "message": f"Service {service_name} is already stopped",
                        "warning": "Service was not running",
                    }
                elif "2" in error_msg or "not found" in error_msg.lower():
                    return {
                        "success": False,
                        "error": f"Service {service_name} not found",
                        "service_name": service_name,
                        "technical_details": error_msg,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to stop service (exit code {result.returncode})",
                        "service_name": service_name,
                        "technical_details": error_msg,
                    }

            # Wait for service to stop and verify
            await asyncio.sleep(2)
            final_status = await self._get_windows_service_status()
            
            return {
                "success": True,
                "service_name": service_name,
                "message": f"Service {service_name} stop command sent successfully",
                "final_status": final_status.get("status", "unknown")
            }

        except Exception as e:
            logger.error(f"Exception in _stop_windows_service: {e}")
            return {
                "success": False,
                "error": f"Exception stopping service: {e}",
                "service_name": service_name,
            }

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
        """Get accurate macOS service status with comprehensive process detection."""
        service_id = f"com.workspace-qdrant-mcp.{self.service_name}"

        try:
            # Step 1: Check if service is loaded in launchd using detailed info
            cmd = ["launchctl", "list", service_id]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            service_loaded = result.returncode == 0
            launchd_status = "unknown"
            launchd_pid = None
            launchd_running = False
            launchd_last_exit_code = None
            
            if service_loaded:
                # Parse launchctl list output for detailed info
                output = stdout.decode()
                for line in output.split("\n"):
                    line = line.strip()
                    if '"PID" =' in line:
                        try:
                            pid_str = line.split('=')[1].strip().rstrip(';')
                            if pid_str != "-" and pid_str.isdigit():
                                launchd_pid = int(pid_str)
                                launchd_status = "running"
                                launchd_running = True
                        except (ValueError, IndexError):
                            pass
                    elif '"LastExitStatus" =' in line:
                        try:
                            exit_code = line.split('=')[1].strip().rstrip(';')
                            if exit_code.isdigit():
                                launchd_last_exit_code = int(exit_code)
                        except (ValueError, IndexError):
                            pass
                
                # If no PID but service is loaded, it's stopped
                if launchd_pid is None and service_loaded:
                    launchd_status = "stopped"
            
            # Step 2: Check for ANY memexd processes running on the system
            all_memexd_processes = await self._find_memexd_processes()
            
            # Step 3: Determine actual status based on all available information
            actual_running_pids = []
            launchd_process_found = False
            
            for pid in all_memexd_processes:
                # Check if this is the launchd-managed process
                if launchd_pid and pid == launchd_pid:
                    launchd_process_found = True
                actual_running_pids.append(pid)
            
            # Step 4: Clean up stale PID files if no corresponding processes
            await self._cleanup_stale_pid_files(actual_running_pids)
            
            # Step 5: Determine final status with improved logic
            if service_loaded:
                if launchd_running and launchd_process_found:
                    final_status = "running"
                    primary_pid = launchd_pid
                    service_type = "launchd"
                elif launchd_running and not launchd_process_found:
                    # Launchd thinks it's running but we can't find the process
                    final_status = "error"
                    primary_pid = launchd_pid
                    service_type = "launchd_stale"
                elif actual_running_pids:
                    # Service loaded but not via launchd, processes found
                    final_status = "running_manual"
                    primary_pid = actual_running_pids[0]
                    service_type = "manual"
                else:
                    # Service loaded but no processes running
                    final_status = "stopped"
                    primary_pid = None
                    service_type = "loaded"
            else:
                # Service not loaded
                if actual_running_pids:
                    final_status = "running_manual"
                    primary_pid = actual_running_pids[0]
                    service_type = "manual"
                else:
                    final_status = "not_installed"
                    primary_pid = None
                    service_type = "none"

            return {
                "success": True,
                "service_id": service_id,
                "status": final_status,
                "running": final_status in ["running", "running_manual"],
                "pid": primary_pid,
                "all_pids": actual_running_pids,
                "process_count": len(actual_running_pids),
                "service_type": service_type,
                "platform": "macOS",
                "loaded": service_loaded,
                "launchd_running": launchd_running,
                "launchd_pid": launchd_pid,
                "last_exit_code": launchd_last_exit_code,
                "status_description": self._get_status_description(final_status, service_type, len(actual_running_pids)),
            }
            
        except Exception as e:
            logger.error(f"Exception in _get_macos_service_status: {e}")
            return {
                "success": False,
                "error": f"Failed to get service status: {e}",
                "service_id": service_id,
            }
    
    def _get_status_description(self, status: str, service_type: str, process_count: int) -> str:
        """Get human-readable status description."""
        status_descriptions = {
            "running": f"Service is running normally via {service_type} (PID tracked)",
            "running_manual": f"Service is running manually ({process_count} process{'es' if process_count != 1 else ''})",
            "stopped": "Service is installed but not running",
            "not_installed": "Service is not installed",
            "error": "Service is in an inconsistent state (launchd thinks it's running but process not found)",
        }
        return status_descriptions.get(status, f"Unknown status: {status}")

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
        service_name = f"memexd-{self.service_name}"
        
        try:
            # Use sc queryex for detailed service information
            query_cmd = ["sc", "queryex", service_name]
            result = await asyncio.create_subprocess_exec(
                *query_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                # Service not found or error
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                if "1060" in error_msg or "does not exist" in error_msg.lower():
                    return {
                        "success": False,
                        "error": f"Service {service_name} is not installed",
                        "service_name": service_name,
                        "status": "not_installed",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to query service status: {error_msg}",
                        "service_name": service_name,
                    }

            # Parse the service status output
            output = stdout.decode()
            status = "unknown"
            pid = None
            auto_start = False
            
            # Parse the state from sc queryex output
            for line in output.split('\n'):
                line = line.strip()
                if 'STATE' in line:
                    if 'RUNNING' in line:
                        status = "running"
                    elif 'STOPPED' in line:
                        status = "stopped"
                    elif 'START_PENDING' in line:
                        status = "start_pending"
                    elif 'STOP_PENDING' in line:
                        status = "stop_pending"
                    elif 'PAUSED' in line:
                        status = "paused"
                elif 'PID' in line:
                    # Extract PID if service is running
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'PID' and i + 2 < len(parts):
                            try:
                                pid = int(parts[i + 2])
                            except ValueError:
                                pass

            # Check if service is set to auto-start by querying configuration
            config_cmd = ["sc", "qc", service_name]
            config_result = await asyncio.create_subprocess_exec(
                *config_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            config_stdout, _ = await config_result.communicate()
            
            if config_result.returncode == 0:
                config_output = config_stdout.decode()
                if 'AUTO_START' in config_output:
                    auto_start = True

            return {
                "success": True,
                "service_name": service_name,
                "status": status,
                "running": status == "running",
                "pid": pid,
                "auto_start": auto_start,
                "platform": "Windows",
            }

        except Exception as e:
            logger.error(f"Exception in _get_windows_service_status: {e}")
            return {
                "success": False,
                "error": f"Exception getting service status: {e}",
                "service_name": service_name,
            }

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
        service_name = f"memexd-{self.service_name}"
        
        try:
            logs = []
            
            # First, try to read from our log files in Windows-appropriate location
            log_dir = Path(self._get_log_path("")).parent  # Get log directory
            log_files = [
                log_dir / "memexd.log",
                log_dir / "memexd.error.log"
            ]
            
            for log_file in log_files:
                if log_file.exists():
                    try:
                        with open(log_file, "r", encoding='utf-8', errors='ignore') as f:
                            content = f.readlines()[-lines:]
                        logs.extend([f"[{log_file.name}] {line.rstrip()}" for line in content])
                    except Exception as e:
                        logs.append(f"[{log_file.name}] Error reading log: {e}")

            # If no file logs found, try Windows Event Log
            if not logs:
                try:
                    # Use PowerShell to query Windows Event Log for our service
                    # This is a fallback approach using powershell
                    powershell_cmd = [
                        "powershell", "-Command",
                        f"Get-EventLog -LogName System -Source 'Service Control Manager' -Newest {lines} | "
                        f"Where-Object {{$_.Message -like '*{service_name}*'}} | "
                        "Select-Object -Property TimeGenerated,EntryType,Message | "
                        "Format-Table -AutoSize | Out-String"
                    ]
                    
                    result = await asyncio.create_subprocess_exec(
                        *powershell_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    stdout, stderr = await result.communicate()
                    
                    if result.returncode == 0 and stdout:
                        event_logs = stdout.decode('utf-8', errors='ignore').split('\n')
                        logs.extend([line.strip() for line in event_logs if line.strip()])
                    else:
                        logs.append("No Windows Event Log entries found for this service")
                        
                except Exception as e:
                    logs.append(f"Could not access Windows Event Log: {e}")

            # If still no logs, check for any memexd process logs in temp
            if not logs:
                import os
                temp_dir = Path(os.environ.get('TEMP', 'C:\\temp'))
                temp_log_patterns = ["memexd*.log", "memexd*.pid"]
                
                for pattern in temp_log_patterns:
                    for temp_file in temp_dir.glob(pattern):
                        if temp_file.is_file():
                            try:
                                content = temp_file.read_text(encoding='utf-8', errors='ignore')
                                logs.append(f"[{temp_file.name}] {content}")
                            except Exception:
                                logs.append(f"[{temp_file.name}] Could not read file")

            if not logs:
                logs = [
                    "No logs available.",
                    f"Expected log location: {log_dir}",
                    "Check if the service has been started and is generating logs.",
                ]

            return {
                "success": True,
                "service_name": service_name,
                "logs": logs,
                "lines_requested": lines,
                "platform": "Windows",
            }

        except Exception as e:
            logger.error(f"Exception in _get_windows_service_logs: {e}")
            return {
                "success": False,
                "error": f"Exception getting service logs: {e}",
                "service_name": service_name,
            }

    async def _find_daemon_binary(self) -> Optional[Path]:
        """Find the memexd binary with comprehensive path resolution."""
        binary_name = self.daemon_binary
        if self.system == "windows":
            binary_name += ".exe"
        
        logger.debug(f"Looking for binary: {binary_name}")
        
        # Search locations in order of preference
        search_locations = []
        
        # 1. First check system PATH using 'which' command (highest priority for global installs)
        try:
            which_cmd = "which" if self.system != "windows" else "where"
            result = await asyncio.create_subprocess_exec(
                which_cmd,
                binary_name,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                binary_path = Path(stdout.decode().strip().split("\n")[0])
                if binary_path.exists() and os.access(binary_path, os.X_OK):
                    logger.debug(f"Found globally installed binary via {which_cmd}: {binary_path}")
                    return binary_path
        except Exception as e:
            logger.debug(f"Error using {which_cmd}: {e}")
        
        # 2. Check UV tool installation locations (for globally installed wqm)
        uv_tool_locations = [
            Path.home() / ".local" / "share" / "uv" / "tools" / "wqm-cli" / "bin" / binary_name,
            Path.home() / ".local" / "bin" / binary_name,
        ]
        
        for uv_path in uv_tool_locations:
            if uv_path.exists() and os.access(uv_path, os.X_OK):
                logger.debug(f"Found UV tool binary at: {uv_path}")
                return uv_path
        
        # 3. Project-relative locations
        project_root = Path.cwd()
        
        # Try to find actual project root by walking up
        current_path = project_root
        for _ in range(5):  # Don't go too far up
            if (current_path / "Cargo.toml").exists() or (current_path / "rust-engine").exists():
                project_root = current_path
                break
            parent = current_path.parent
            if parent == current_path:  # Reached filesystem root
                break
            current_path = parent
        
        # Rust build locations
        rust_locations = [
            project_root / "rust-engine" / "target" / "release",
            project_root / "rust-engine" / "target" / "debug",
            project_root / "target" / "release",
            project_root / "target" / "debug",
            # Also check if we're inside rust-engine directory
            project_root / "target" / "release",
            project_root / "target" / "debug",
        ]
        
        for rust_dir in rust_locations:
            search_locations.append(rust_dir / binary_name)
        
        # 4. System PATH locations
        path_env = os.environ.get("PATH", "")
        if path_env:
            for path_dir in path_env.split(os.pathsep):
                if path_dir.strip():
                    search_locations.append(Path(path_dir) / binary_name)
        
        # 5. Common system locations
        common_locations = [
            Path("/usr/local/bin") / binary_name,
            Path("/usr/bin") / binary_name,
            Path.home() / ".local" / "bin" / binary_name,
        ]
        
        if self.system == "darwin":
            common_locations.extend([
                Path("/opt/homebrew/bin") / binary_name,
                Path("/usr/local/homebrew/bin") / binary_name,
            ])
        
        search_locations.extend(common_locations)
        
        # 6. Search all remaining locations
        for binary_path in search_locations:
            try:
                if binary_path.exists() and binary_path.is_file():
                    # Verify it's executable
                    if os.access(binary_path, os.X_OK):
                        logger.debug(f"Found executable binary at: {binary_path}")
                        return binary_path
                    else:
                        logger.debug(f"Found binary but not executable: {binary_path}")
            except (OSError, PermissionError) as e:
                logger.debug(f"Error checking path {binary_path}: {e}")
                continue
        
        # 7. Log all attempted locations for debugging
        logger.error(f"Binary {binary_name} not found in any of these locations:")
        all_attempted = uv_tool_locations + search_locations
        for i, location in enumerate(all_attempted[:15]):  # Show first 15
            logger.error(f"  {i+1}. {location}")
        if len(all_attempted) > 15:
            logger.error(f"  ... and {len(all_attempted) - 15} more locations")
        
        # 8. Provide helpful guidance
        logger.error("")
        logger.error("Possible solutions:")
        logger.error("  1. If using global installation: uv tool install wqm-cli")
        logger.error("  2. If building from source: cd rust-engine && cargo build --release --bin memexd")
        logger.error("  3. Check that memexd is in your PATH: which memexd")
        
        return None

    async def _find_memexd_processes(self) -> list[int]:
        """Find all memexd processes running on the system."""
        try:
            # Use pgrep to find all memexd processes
            pgrep_cmd = ["pgrep", "-f", "memexd"]
            result = await asyncio.create_subprocess_exec(
                *pgrep_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                pids = []
                for line in stdout.decode().strip().split('\n'):
                    if line.strip() and line.strip().isdigit():
                        pid = int(line.strip())
                        # Double-check this is actually a memexd process
                        if await self._verify_process_is_memexd(pid):
                            pids.append(pid)
                return pids
            else:
                # pgrep found no processes
                return []
                
        except Exception as e:
            logger.debug(f"Error finding memexd processes: {e}")
            return []
    
    async def _verify_process_is_memexd(self, pid: int) -> bool:
        """Verify that a PID is actually a memexd process."""
        try:
            # Check both command name and full command line
            ps_cmd = ["ps", "-p", str(pid), "-o", "comm=,args="]
            result = await asyncio.create_subprocess_exec(
                *ps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                output = stdout.decode().strip()
                # Check if either the command name or arguments contain memexd
                return "memexd" in output.lower()
            return False
            
        except Exception:
            return False
    
    async def _cleanup_all_memexd_processes(self) -> None:
        """Terminate all memexd processes gracefully, then forcefully if needed."""
        try:
            # Find all memexd processes
            memexd_pids = await self._find_memexd_processes()
            
            if not memexd_pids:
                logger.debug("No memexd processes found to clean up")
                return
            
            logger.debug(f"Found {len(memexd_pids)} memexd processes to terminate: {memexd_pids}")
            
            # First, try graceful termination with SIGTERM
            for pid in memexd_pids:
                try:
                    term_cmd = ["kill", "-TERM", str(pid)]
                    await asyncio.create_subprocess_exec(*term_cmd)
                    logger.debug(f"Sent SIGTERM to process {pid}")
                except Exception as e:
                    logger.debug(f"Failed to send SIGTERM to {pid}: {e}")
            
            # Wait for graceful shutdown
            await asyncio.sleep(3)
            
            # Check which processes are still running
            remaining_pids = await self._find_memexd_processes()
            
            # Force kill any remaining processes
            for pid in remaining_pids:
                try:
                    kill_cmd = ["kill", "-9", str(pid)]
                    await asyncio.create_subprocess_exec(*kill_cmd)
                    logger.debug(f"Force killed process {pid}")
                except Exception as e:
                    logger.debug(f"Failed to force kill {pid}: {e}")
            
            # Final verification
            final_check = await self._find_memexd_processes()
            if final_check:
                logger.warning(f"Failed to terminate all memexd processes: {final_check}")
            else:
                logger.debug(f"Successfully terminated all {len(memexd_pids)} memexd processes")
                
        except Exception as e:
            logger.error(f"Error in cleanup_all_memexd_processes: {e}")
    
    async def _cleanup_pid_files(self) -> None:
        """Clean up all memexd PID files."""
        pid_files = [
            "/tmp/memexd.pid",
            "/tmp/memexd-launchd.pid", 
            "/tmp/memexd-manual.pid",
            "/tmp/memexd-service.pid",
        ]
        
        for pid_file in pid_files:
            try:
                pid_path = Path(pid_file)
                if pid_path.exists():
                    pid_path.unlink()
                    logger.debug(f"Removed PID file: {pid_file}")
            except OSError as e:
                logger.debug(f"Failed to remove PID file {pid_file}: {e}")
    
    async def _cleanup_stale_pid_files(self, active_pids: list[int]) -> None:
        """Clean up PID files that don't correspond to running processes."""
        pid_files = [
            "/tmp/memexd.pid",
            "/tmp/memexd-launchd.pid", 
            "/tmp/memexd-manual.pid",
            "/tmp/memexd-service.pid",
        ]
        
        for pid_file in pid_files:
            try:
                pid_path = Path(pid_file)
                if pid_path.exists():
                    try:
                        # Read PID from file
                        stored_pid = int(pid_path.read_text().strip())
                        
                        # If this PID is not in our active list, remove the file
                        if stored_pid not in active_pids:
                            # Double-check the process isn't running
                            if not await self._verify_process_is_memexd(stored_pid):
                                pid_path.unlink()
                                logger.debug(f"Removed stale PID file: {pid_file} (PID {stored_pid})")
                            else:
                                logger.debug(f"PID file {pid_file} has running process {stored_pid}, keeping")
                    except (ValueError, FileNotFoundError):
                        # Invalid PID file, remove it
                        pid_path.unlink()
                        logger.debug(f"Removed invalid PID file: {pid_file}")
            except OSError as e:
                logger.debug(f"Error checking PID file {pid_file}: {e}")

    def _get_log_path(self, filename: str) -> str:
        """Get appropriate log path for user service."""
        if self.system == "windows":
            # Windows - use LocalAppData for user logs
            import os
            local_appdata = os.environ.get('LOCALAPPDATA', str(Path.home() / "AppData" / "Local"))
            user_log_dir = Path(local_appdata) / "workspace-qdrant" / "logs"
        elif self.system == "darwin":
            # macOS - use ~/Library/Logs
            user_log_dir = Path.home() / "Library" / "Logs"
        else:
            # Linux - use XDG log directory
            user_log_dir = Path.home() / ".local" / "share" / "workspace-qdrant" / "logs"
        
        user_log_dir.mkdir(parents=True, exist_ok=True)
        return str(user_log_dir / filename)


# Create service manager instance
service_manager = ServiceManager()


@service_app.command("list")
def list_services(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by specific project identifier"
    ),
    show_ports: bool = typer.Option(
        False, "--ports", help="Show allocated ports for each daemon"
    ),
) -> None:
    """List all project daemon instances."""

    async def _list():
        manager = await get_daemon_manager()
        daemons = await manager.list_daemons()
        
        if not daemons:
            console.print("No daemon instances found.")
            return
        
        # Create table
        table = Table(title="Daemon Instances")
        table.add_column("Project ID", style="cyan")
        table.add_column("Project Name", style="white")
        table.add_column("Status", style="white")
        table.add_column("PID", style="dim")
        if show_ports:
            table.add_column("Port", style="yellow")
        table.add_column("Health", style="white")
        
        for daemon_key, daemon_info in daemons.items():
            config = daemon_info.get("config", {})
            status_info = daemon_info.get("status", {})
            process_info = daemon_info.get("process_info", {})
            
            project_id = config.get("project_id", daemon_key)
            project_name = config.get("project_name", "unknown")
            
            # Filter by project if specified
            if project and project not in project_id:
                continue
            
            # Format status with colors
            state = status_info.get("state", "unknown")
            if state == "running":
                status_text = Text("Running", style="green")
            elif state == "stopped":
                status_text = Text("Stopped", style="yellow")
            elif state == "failed":
                status_text = Text("Failed", style="red")
            else:
                status_text = Text(state.title(), style="dim")
            
            # Format health status
            health_status = status_info.get("health_status", "unknown")
            if health_status == "healthy":
                health_text = Text("Healthy", style="green")
            elif health_status == "unhealthy":
                health_text = Text("Unhealthy", style="red")
            else:
                health_text = Text("Unknown", style="dim")
            
            pid = process_info.get("pid", "-")
            pid_str = str(pid) if pid else "-"
            
            row = [project_id, project_name, str(status_text), pid_str]
            if show_ports:
                port = config.get("grpc_port", "-")
                row.append(str(port))
            row.append(str(health_text))
            
            table.add_row(*row)
        
        console.print(table)
        
        # Show port allocation summary if requested
        if show_ports:
            from common.core.daemon_manager import PortManager
            port_manager = PortManager.get_instance()
            allocated_ports = port_manager.get_allocated_ports()
            
            if allocated_ports:
                console.print(f"\nPort allocations: {len(allocated_ports)} ports in use")
                port_ranges = sorted(allocated_ports.keys())
                if port_ranges:
                    console.print(f"Range: {min(port_ranges)}-{max(port_ranges)}")

    handle_async_command(_list())


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
def start_service(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Target specific project daemon instance"
    ),
) -> None:
    """Start the memexd user service or specific project daemon."""

    async def _start():
        if project:
            # Start specific project daemon
            manager = await get_daemon_manager()
            detector = ProjectDetector()
            project_name = detector.get_project_name()
            project_path = os.getcwd()
            
            success = await manager.start_daemon(project_name, project_path)
            
            if success:
                console.print(
                    Panel.fit(
                        f"✅ Project daemon started successfully!\n\n"
                        f"Project: {project_name}\n"
                        f"Project ID: {project}",
                        title="Daemon Start",
                        style="green",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        f"❌ Failed to start project daemon!\n\n"
                        f"Project: {project_name}",
                        title="Start Error",
                        style="red",
                    )
                )
                raise typer.Exit(1)
        else:
            # Start system service
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
                error_text = f"❌ Failed to start service!\n\nError: {result['error']}"
                
                # Add suggestion if available
                if 'suggestion' in result:
                    error_text += f"\n\n💡 Suggestion:\n{result['suggestion']}"
                
                # Add help command if available
                if 'help_command' in result:
                    error_text += f"\n\n🔧 Quick fix:\n{result['help_command']}"
                
                # Add technical details if available
                if 'technical_details' in result:
                    error_text += f"\n\n🔍 Technical details:\n{result['technical_details']}"
                
                console.print(
                    Panel.fit(
                        error_text,
                        title="Start Error",
                        style="red",
                    )
                )
                raise typer.Exit(1)

    handle_async_command(_start())


@service_app.command("stop")
def stop_service(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Target specific project daemon instance"
    ),
) -> None:
    """Stop the memexd user service or specific project daemon."""

    async def _stop():
        if project:
            # Stop specific project daemon
            manager = await get_daemon_manager()
            detector = ProjectDetector()
            project_name = detector.get_project_name()
            project_path = os.getcwd()
            
            success = await manager.stop_daemon(project_name, project_path)
            
            if success:
                console.print(
                    Panel.fit(
                        f"✅ Project daemon stopped successfully!\n\n"
                        f"Project: {project_name}\n"
                        f"Project ID: {project}",
                        title="Daemon Stop",
                        style="green",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        f"❌ Failed to stop project daemon!\n\n"
                        f"Project: {project_name}",
                        title="Stop Error",
                        style="red",
                    )
                )
                raise typer.Exit(1)
        else:
            # Stop system service
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
                if status == "running_manual":
                    status_text = Text("Running (Manual)", style="green")
                else:
                    status_text = Text("Running", style="green")
            elif status == "stopped":
                status_text = Text("Stopped", style="yellow")
            elif status == "failed":
                status_text = Text("Failed", style="red")
            elif status == "not_loaded":
                status_text = Text("Not Loaded", style="dim")
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
