"""Service management commands for workspace-qdrant daemon using memexd.

This provides robust service management using the actual memexd binary:
1. Uses the real memexd binary at /usr/local/bin/memexd
2. Robust error handling for all OS operations
3. Proper service state management with standard OS conventions
4. Cross-platform launchd/systemd support
5. macOS: KeepAlive=false allows proper start/stop control
6. Linux: Restart=on-failure provides crash recovery without interfering with manual stops

Commands:
    wqm service install               # Install daemon as user service
    wqm service uninstall             # Remove user service
    wqm service start                 # Start user service
    wqm service stop                  # Stop user service
    wqm service restart               # Restart user service
    wqm service status                # Show user service status
    wqm service logs                  # Show user service logs
"""

import asyncio
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from common.logging.loguru_config import get_logger
from ..utils import create_command_app, handle_async_command

# Initialize app and logger
service_app = create_command_app(
    name="service",
    help_text="User service management for memexd daemon",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


class MemexdServiceManager:
    """Service manager for the actual memexd binary."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.service_name = "workspace-qdrant-daemon"
        self.service_id = f"com.workspace-qdrant.{self.service_name}"
        
        # Use the actual memexd binary
        self.memexd_binary = Path("/usr/local/bin/memexd")
        self.validate_binary()
        
    def validate_binary(self) -> None:
        """Validate that the memexd binary exists and is executable."""
        if not self.memexd_binary.exists():
            raise FileNotFoundError(
                f"memexd binary not found at {self.memexd_binary}. "
                "Please ensure memexd is installed and available at /usr/local/bin/memexd"
            )
        
        if not os.access(self.memexd_binary, os.X_OK):
            raise PermissionError(
                f"memexd binary at {self.memexd_binary} is not executable. "
                "Please check permissions."
            )
        
        logger.debug(f"Found valid memexd binary at {self.memexd_binary}")
    
    def get_config_path(self) -> Path:
        """Get the default configuration path for memexd."""
        config_dir = Path.home() / ".config" / "workspace-qdrant"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing config file
        workspace_config = config_dir / "workspace_qdrant_config.toml"
        default_config = config_dir / "config.toml"
        
        if workspace_config.exists():
            return workspace_config
        else:
            return default_config
    
    def get_log_path(self) -> Path:
        """Get the log file path for memexd."""
        log_dir = Path.home() / ".local" / "var" / "log" / "workspace-qdrant"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "memexd.log"
    
    def get_pid_path(self) -> Path:
        """Get the PID file path for memexd."""
        pid_dir = Path.home() / ".local" / "var" / "run" / "workspace-qdrant"
        pid_dir.mkdir(parents=True, exist_ok=True)
        return pid_dir / "memexd.pid"
    
    async def install_service(self, auto_start: bool = True) -> Dict[str, Any]:
        """Install the service with proper error handling."""
        try:
            if self.system == "darwin":
                return await self._install_macos_service(auto_start)
            elif self.system == "linux":
                return await self._install_linux_service(auto_start)
            elif self.system == "windows":
                return {"success": False, "error": "Windows support not implemented yet"}
            else:
                return {"success": False, "error": f"Unsupported platform: {self.system}"}
                
        except Exception as e:
            logger.error("Service installation failed", error=str(e), exc_info=True)
            return {"success": False, "error": f"Installation failed: {e}"}
    
    async def _install_macos_service(self, auto_start: bool) -> Dict[str, Any]:
        """Install macOS launchd service with robust error handling using modern bootstrap."""
        
        # Ensure memexd binary exists
        try:
            self.validate_binary()
        except (FileNotFoundError, PermissionError) as e:
            return {
                "success": False,
                "error": str(e)
            }
        
        # Service file location
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_path = plist_dir / f"{self.service_id}.plist"
        
        try:
            # Create directory with proper error handling
            plist_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = plist_dir / ".wqm_test"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                return {
                    "success": False,
                    "error": f"No write permission to {plist_dir}",
                    "suggestion": f"Check permissions: ls -la {plist_dir}"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot create service directory: {e}"
            }
        
        # Get paths for memexd configuration
        config_path = self.get_config_path()
        log_path = self.get_log_path()
        pid_path = self.get_pid_path()
        
        # Create simple, robust plist for memexd
        # KeepAlive=false allows proper start/stop control
        # Use systemctl-style restart behavior (restart only on crash)
        plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{self.service_id}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{self.memexd_binary}</string>
        <string>--config</string>
        <string>{config_path}</string>
        <string>--log-level</string>
        <string>info</string>
        <string>--pid-file</string>
        <string>{pid_path}</string>
    </array>
    
    <key>RunAtLoad</key>
    <{"true" if auto_start else "false"}/>
    
    <key>KeepAlive</key>
    <false/>
    
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    
    <key>StandardErrorPath</key>
    <string>{log_path}.error</string>
    
    <key>WorkingDirectory</key>
    <string>{Path.home()}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>{Path.home()}</string>
    </dict>
</dict>
</plist>'''

        # Write plist with error handling
        try:
            plist_path.write_text(plist_content)
        except Exception as e:
            return {
                "success": False, 
                "error": f"Failed to write plist: {e}",
                "plist_path": str(plist_path)
            }
        
        # Bootstrap service with proper error handling using modern launchctl
        try:
            # First remove any existing service to avoid conflicts
            await self._bootout_service()
            await asyncio.sleep(1)
            
            # Bootstrap the service using modern gui domain syntax for LaunchAgents
            gui_domain = f"gui/{os.getuid()}"
            cmd = ["launchctl", "bootstrap", gui_domain, str(plist_path)]
            result = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                # Try to provide better error messages
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                
                if "service already loaded" in error_msg.lower():
                    # Service already exists - this is actually OK
                    pass
                else:
                    return {
                        "success": False,
                        "error": f"Failed to bootstrap service: {error_msg}",
                        "plist_path": str(plist_path),
                        "returncode": result.returncode
                    }
            
            return {
                "success": True,
                "service_id": self.service_id,
                "plist_path": str(plist_path),
                "binary_path": str(self.memexd_binary),
                "config_path": str(config_path),
                "log_path": str(log_path),
                "pid_path": str(pid_path),
                "auto_start": auto_start,
                "message": f"Service {self.service_id} installed successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception bootstrapping service: {e}",
                "plist_path": str(plist_path)
            }
    
    async def _install_linux_service(self, auto_start: bool) -> Dict[str, Any]:
        """Install Linux systemd service."""
        service_name = f"{self.service_name}.service"
        service_dir = Path.home() / ".config" / "systemd" / "user"
        service_path = service_dir / service_name
        
        try:
            service_dir.mkdir(parents=True, exist_ok=True)
            
            # Get paths for memexd configuration
            config_path = self.get_config_path()
            log_path = self.get_log_path()
            pid_path = self.get_pid_path()
            
            service_content = f'''[Unit]
Description=Workspace Qdrant Daemon (memexd)
After=network.target

[Service]
Type=simple
ExecStart={self.memexd_binary} --config {config_path} --log-level info --pid-file {pid_path}
Restart=on-failure
RestartSec=5
Environment=HOME={Path.home()}
WorkingDirectory={Path.home()}
StandardOutput=append:{log_path}
StandardError=append:{log_path}.error

[Install]
WantedBy=default.target
'''
            
            service_path.write_text(service_content)
            
            # Reload systemd
            cmd = ["systemctl", "--user", "daemon-reload"]
            result = await asyncio.create_subprocess_exec(*cmd)
            await result.wait()
            
            if auto_start:
                cmd = ["systemctl", "--user", "enable", service_name]
                result = await asyncio.create_subprocess_exec(*cmd)
                await result.wait()
            
            return {
                "success": True,
                "service_name": service_name,
                "service_path": str(service_path),
                "binary_path": str(self.memexd_binary),
                "config_path": str(config_path),
                "log_path": str(log_path),
                "pid_path": str(pid_path),
                "message": f"Service {service_name} installed successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to install Linux service: {e}"
            }
    
    async def uninstall_service(self) -> Dict[str, Any]:
        """Uninstall service with proper cleanup."""
        try:
            if self.system == "darwin":
                return await self._uninstall_macos_service()
            elif self.system == "linux":
                return await self._uninstall_linux_service()
            else:
                return {"success": False, "error": f"Unsupported platform: {self.system}"}
                
        except Exception as e:
            logger.error("Service uninstall failed", error=str(e), exc_info=True)
            return {"success": False, "error": f"Uninstall failed: {e}"}
    
    async def _uninstall_macos_service(self) -> Dict[str, Any]:
        """Uninstall macOS service with proper cleanup using modern bootout."""
        # Find the actual plist file - it might have a different service ID
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        actual_plist = None
        actual_service_id = None
        
        # Look for workspace-qdrant related plist files
        for plist_file in plist_dir.glob("*workspace-qdrant*.plist"):
            if plist_file.is_file():
                actual_plist = plist_file
                actual_service_id = plist_file.stem
                break
        
        if not actual_plist:
            return {
                "success": False, 
                "error": "Service not installed",
                "suggestion": "No workspace-qdrant service found"
            }
        
        try:
            # Stop service first and handle memexd shutdown bug
            await self._force_stop_service(actual_service_id)
            await asyncio.sleep(1)
            
            # Bootout service using modern launchctl
            await self._bootout_service(actual_service_id)
            await asyncio.sleep(1)
            
            # Remove plist file
            actual_plist.unlink()
            
            # Cleanup PID file
            pid_file = self.get_pid_path()
            if pid_file.exists():
                pid_file.unlink()
            
            return {
                "success": True,
                "service_id": actual_service_id,
                "plist_path": str(actual_plist),
                "message": f"Service {actual_service_id} uninstalled successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to uninstall service: {e}"
            }
    
    async def _bootout_service(self, service_id: Optional[str] = None) -> None:
        """Bootout service using modern launchctl, ignoring errors if not loaded."""
        try:
            gui_domain = f"gui/{os.getuid()}"
            
            if service_id:
                # Bootout specific service
                cmd = ["launchctl", "bootout", gui_domain, service_id]
            else:
                # Find and bootout any workspace-qdrant service
                plist_dir = Path.home() / "Library" / "LaunchAgents"
                for plist_file in plist_dir.glob("*workspace-qdrant*.plist"):
                    if plist_file.is_file():
                        cmd = ["launchctl", "bootout", gui_domain, str(plist_file)]
                        result = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        await result.communicate()
                return
                
                # No service found to bootout
                return
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await result.communicate()
            # Don't check return code - bootout can fail if service not loaded
        except:
            pass  # Ignore bootout errors
            
    async def _force_stop_service(self, service_id: str) -> None:
        """Force stop service, handling memexd shutdown bug with SIGKILL if necessary."""
        try:
            # First try graceful stop - use gui domain for LaunchAgents
            gui_domain = f"gui/{os.getuid()}"
            cmd = ["launchctl", "kill", "TERM", gui_domain + "/" + service_id]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await result.communicate()
            
            # Wait for graceful shutdown
            await asyncio.sleep(3)
            
            # Check if any memexd processes are still running
            ps_cmd = ["pgrep", "-f", "memexd"]
            ps_result = await asyncio.create_subprocess_exec(
                *ps_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            ps_stdout, ps_stderr = await ps_result.communicate()
            
            if ps_result.returncode == 0 and ps_stdout.decode().strip():
                # memexd processes still running despite SIGTERM - force kill them
                pids = [p.strip() for p in ps_stdout.decode().strip().split('\n') if p.strip().isdigit()]
                
                for pid in pids:
                    try:
                        # Force kill the stubborn memexd process
                        kill_cmd = ["kill", "-KILL", pid]
                        kill_result = await asyncio.create_subprocess_exec(
                            *kill_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        await kill_result.communicate()
                    except:
                        pass  # Ignore kill errors
                        
        except:
            pass  # Ignore force stop errors
    
    async def _uninstall_linux_service(self) -> Dict[str, Any]:
        """Uninstall Linux service."""
        service_name = f"{self.service_name}.service"
        service_path = Path.home() / ".config" / "systemd" / "user" / service_name
        
        if not service_path.exists():
            return {"success": False, "error": "Service not installed"}
        
        try:
            # Stop and disable
            for action in ["stop", "disable"]:
                cmd = ["systemctl", "--user", action, service_name]
                result = await asyncio.create_subprocess_exec(*cmd)
                await result.wait()
            
            service_path.unlink()
            
            # Reload systemd
            cmd = ["systemctl", "--user", "daemon-reload"]
            result = await asyncio.create_subprocess_exec(*cmd)
            await result.wait()
            
            return {
                "success": True,
                "service_name": service_name,
                "message": f"Service {service_name} uninstalled successfully"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to uninstall: {e}"}
    
    async def start_service(self) -> Dict[str, Any]:
        """Start service with proper error handling."""
        try:
            if self.system == "darwin":
                return await self._start_macos_service()
            elif self.system == "linux":
                return await self._start_linux_service()
            else:
                return {"success": False, "error": f"Unsupported platform: {self.system}"}
        except Exception as e:
            logger.error("Service start failed", error=str(e), exc_info=True)
            return {"success": False, "error": f"Start failed: {e}"}
    
    async def _start_macos_service(self) -> Dict[str, Any]:
        """Start macOS service with proper service ID detection using modern launchctl."""
        # Find the actual plist file - it might have a different service ID
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        actual_plist = None
        actual_service_id = None
        
        # Look for workspace-qdrant related plist files
        for plist_file in plist_dir.glob("*workspace-qdrant*.plist"):
            if plist_file.is_file():
                actual_plist = plist_file
                actual_service_id = plist_file.stem
                break
        
        if not actual_plist:
            return {
                "success": False,
                "error": "Service not installed",
                "suggestion": "Run 'wqm service install' first"
            }
        
        try:
            gui_domain = f"gui/{os.getuid()}"
            
            # Check if service is bootstrapped, if not bootstrap it first
            list_cmd = ["launchctl", "list", actual_service_id]
            list_result = await asyncio.create_subprocess_exec(
                *list_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await list_result.communicate()
            
            if list_result.returncode != 0:
                # Service not bootstrapped, bootstrap it first
                bootstrap_cmd = ["launchctl", "bootstrap", gui_domain, str(actual_plist)]
                bootstrap_result = await asyncio.create_subprocess_exec(
                    *bootstrap_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                bootstrap_stdout, bootstrap_stderr = await bootstrap_result.communicate()
                
                if bootstrap_result.returncode != 0:
                    error_msg = bootstrap_stderr.decode().strip() or bootstrap_stdout.decode().strip()
                    if "service already loaded" not in error_msg.lower():
                        return {
                            "success": False,
                            "error": f"Failed to bootstrap service: {error_msg}",
                            "returncode": bootstrap_result.returncode
                        }
            
            # Now use modern launchctl kickstart to start the service
            cmd = ["launchctl", "kickstart", "-k", gui_domain + "/" + actual_service_id]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                return {
                    "success": False,
                    "error": f"Failed to start service: {error_msg}",
                    "returncode": result.returncode
                }
            
            # Verify service started
            await asyncio.sleep(2)
            status = await self.get_service_status()
            
            if status.get("success") and status.get("running"):
                return {
                    "success": True,
                    "service_id": actual_service_id,
                    "plist_path": str(actual_plist),
                    "message": "Service started successfully",
                    "status": status
                }
            else:
                return {
                    "success": False,
                    "error": "Service command succeeded but service is not running",
                    "service_id": actual_service_id,
                    "plist_path": str(actual_plist),
                    "debug_info": status
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception starting service: {e}"
            }
    
    async def _start_linux_service(self) -> Dict[str, Any]:
        """Start Linux service."""
        service_name = f"{self.service_name}.service"
        
        try:
            cmd = ["systemctl", "--user", "start", service_name]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode().strip()
                return {
                    "success": False,
                    "error": f"Failed to start service: {error_msg}"
                }
            
            return {
                "success": True,
                "service_name": service_name,
                "message": "Service started successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception starting service: {e}"
            }
    
    async def stop_service(self) -> Dict[str, Any]:
        """Stop service with proper error handling."""
        try:
            if self.system == "darwin":
                return await self._stop_macos_service()
            elif self.system == "linux":
                return await self._stop_linux_service()
            else:
                return {"success": False, "error": f"Unsupported platform: {self.system}"}
        except Exception as e:
            logger.error("Service stop failed", error=str(e), exc_info=True)
            return {"success": False, "error": f"Stop failed: {e}"}
    
    async def _stop_macos_service(self) -> Dict[str, Any]:
        """Stop macOS service using modern launchctl and handle memexd shutdown bug."""
        try:
            # Find the actual plist file - it might have a different service ID
            plist_dir = Path.home() / "Library" / "LaunchAgents"
            actual_plist = None
            actual_service_id = None
            
            # Look for workspace-qdrant related plist files
            for plist_file in plist_dir.glob("*workspace-qdrant*.plist"):
                if plist_file.is_file():
                    actual_plist = plist_file
                    actual_service_id = plist_file.stem
                    break
            
            if not actual_plist:
                return {
                    "success": False,
                    "error": "No workspace-qdrant service found",
                    "suggestion": "Service may not be installed"
                }
            
            gui_domain = f"gui/{os.getuid()}"
            
            # First try graceful stop using modern launchctl kill with SIGTERM
            cmd = ["launchctl", "kill", "TERM", gui_domain + "/" + actual_service_id]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            # Wait for graceful shutdown
            await asyncio.sleep(3)
            
            # Check if any memexd processes are still running (memexd shutdown bug)
            ps_cmd = ["pgrep", "-f", "memexd"]
            ps_result = await asyncio.create_subprocess_exec(
                *ps_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            ps_stdout, ps_stderr = await ps_result.communicate()
            
            if ps_result.returncode == 0 and ps_stdout.decode().strip():
                # memexd is stuck - force kill with SIGKILL due to shutdown bug
                force_kill_cmd = ["launchctl", "kill", "KILL", gui_domain + "/" + actual_service_id]
                force_result = await asyncio.create_subprocess_exec(
                    *force_kill_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await force_result.communicate()
                
                # Wait a moment for force kill to take effect
                await asyncio.sleep(2)
                
                # Check again if processes are gone
                final_ps_result = await asyncio.create_subprocess_exec(
                    *ps_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                final_ps_stdout, final_ps_stderr = await final_ps_result.communicate()
                
                if final_ps_result.returncode == 0 and final_ps_stdout.decode().strip():
                    # Still running despite SIGKILL - manually kill remaining processes
                    pids = [p.strip() for p in final_ps_stdout.decode().strip().split('\n') if p.strip().isdigit()]
                    
                    for pid in pids:
                        try:
                            manual_kill_cmd = ["kill", "-KILL", pid]
                            manual_kill_result = await asyncio.create_subprocess_exec(
                                *manual_kill_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            await manual_kill_result.communicate()
                        except:
                            pass  # Ignore manual kill errors
                    
                    return {
                        "success": True,
                        "service_id": actual_service_id,
                        "plist_path": str(actual_plist),
                        "message": "Service stopped successfully (forced due to memexd shutdown bug)",
                        "method": "force_kill",
                        "status": "force_stopped"
                    }
            
            # Verify final stop status
            status_cmd = ["launchctl", "list", actual_service_id]
            status_result = await asyncio.create_subprocess_exec(
                *status_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            status_stdout, status_stderr = await status_result.communicate()
            
            if status_result.returncode == 0:
                # Service is still loaded, check if it's actually stopped
                output = status_stdout.decode().strip()
                lines = output.split('\n')
                
                pid = None
                for line in lines:
                    if '"PID" =' in line:
                        try:
                            pid_str = line.split('=')[1].strip().rstrip(';')
                            if pid_str != "-" and pid_str.isdigit():
                                pid = int(pid_str)
                        except (ValueError, IndexError):
                            pass
                
                if pid is None:
                    # Service is loaded but not running (stopped successfully)
                    return {
                        "success": True,
                        "service_id": actual_service_id,
                        "plist_path": str(actual_plist),
                        "message": "Service stopped successfully",
                        "method": "graceful_stop",
                        "status": "loaded_but_stopped"
                    }
                else:
                    # Service is still running after all attempts
                    return {
                        "success": False,
                        "error": f"Service still running with PID {pid} after stop attempts",
                        "service_id": actual_service_id,
                        "plist_path": str(actual_plist),
                        "suggestion": "Manual intervention may be required"
                    }
            else:
                # Service not in launchctl list - successfully stopped
                return {
                    "success": True,
                    "service_id": actual_service_id,
                    "plist_path": str(actual_plist),
                    "message": "Service stopped successfully",
                    "method": "graceful_stop",
                    "status": "not_loaded"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception stopping service: {e}"
            }
    
    async def _stop_linux_service(self) -> Dict[str, Any]:
        """Stop Linux service."""
        service_name = f"{self.service_name}.service"
        
        try:
            cmd = ["systemctl", "--user", "stop", service_name]
            result = await asyncio.create_subprocess_exec(*cmd)
            await result.wait()
            
            return {
                "success": True,
                "service_name": service_name,
                "message": "Service stopped successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception stopping service: {e}"
            }
    
    async def restart_service(self) -> Dict[str, Any]:
        """Restart service by stopping then starting."""
        stop_result = await self.stop_service()
        if not stop_result["success"]:
            return stop_result
            
        await asyncio.sleep(2)
        return await self.start_service()
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status with proper error handling."""
        try:
            if self.system == "darwin":
                return await self._get_macos_service_status()
            elif self.system == "linux":
                return await self._get_linux_service_status()
            else:
                return {"success": False, "error": f"Unsupported platform: {self.system}"}
        except Exception as e:
            logger.error("Service status check failed", error=str(e), exc_info=True)
            return {"success": False, "error": f"Status check failed: {e}"}
    
    async def _get_macos_service_status(self) -> Dict[str, Any]:
        """Get macOS service status using launchctl with proper service ID detection."""
        # Find the actual plist file - it might have a different service ID
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        actual_plist = None
        actual_service_id = None
        
        # Look for workspace-qdrant related plist files
        for plist_file in plist_dir.glob("*workspace-qdrant*.plist"):
            if plist_file.is_file():
                actual_plist = plist_file
                actual_service_id = plist_file.stem
                break
        
        if not actual_plist:
            return {
                "success": True,
                "status": "not_installed",
                "running": False,
                "message": "Service is not installed"
            }
        
        try:
            # Use launchctl list to check service status
            cmd = ["launchctl", "list", actual_service_id]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                # Service is loaded, parse output for PID
                output = stdout.decode().strip()
                lines = output.split('\n')
                
                pid = None
                status = "loaded"
                
                for line in lines:
                    if '"PID" =' in line:
                        try:
                            pid_str = line.split('=')[1].strip().rstrip(';')
                            if pid_str != "-" and pid_str.isdigit():
                                pid = int(pid_str)
                                status = "running"
                        except (ValueError, IndexError):
                            pass
                
                # Double-check by looking for actual memexd processes
                if not pid:
                    ps_cmd = ["pgrep", "-f", "memexd"]
                    ps_result = await asyncio.create_subprocess_exec(
                        *ps_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    ps_stdout, ps_stderr = await ps_result.communicate()
                    
                    if ps_result.returncode == 0 and ps_stdout.decode().strip():
                        # Found memexd processes
                        pids = [p.strip() for p in ps_stdout.decode().strip().split('\n') if p.strip()]
                        if pids:
                            pid = int(pids[0])  # Use first PID
                            status = "running"
                
                return {
                    "success": True,
                    "status": status,
                    "running": pid is not None,
                    "pid": pid,
                    "service_id": actual_service_id,
                    "plist_path": str(actual_plist),
                    "platform": "macOS"
                }
            else:
                # Service not loaded, check if processes are running anyway
                ps_cmd = ["pgrep", "-f", "memexd"]
                ps_result = await asyncio.create_subprocess_exec(
                    *ps_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                ps_stdout, ps_stderr = await ps_result.communicate()
                
                if ps_result.returncode == 0 and ps_stdout.decode().strip():
                    # Found memexd processes even though service not loaded
                    pids = [p.strip() for p in ps_stdout.decode().strip().split('\n') if p.strip()]
                    return {
                        "success": True,
                        "status": "running_unmanaged",
                        "running": True,
                        "pid": int(pids[0]) if pids else None,
                        "service_id": actual_service_id,
                        "plist_path": str(actual_plist),
                        "platform": "macOS",
                        "message": "Process running but not managed by launchd"
                    }
                else:
                    return {
                        "success": True,
                        "status": "not_loaded",
                        "running": False,
                        "service_id": actual_service_id,
                        "plist_path": str(actual_plist),
                        "platform": "macOS"
                    }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception checking service status: {e}",
                "service_id": actual_service_id or self.service_id
            }
    
    async def _get_linux_service_status(self) -> Dict[str, Any]:
        """Get Linux service status using systemctl."""
        service_name = f"{self.service_name}.service"
        
        try:
            cmd = ["systemctl", "--user", "is-active", service_name]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            status = stdout.decode().strip()
            running = status == "active"
            
            return {
                "success": True,
                "status": status,
                "running": running,
                "service_name": service_name,
                "platform": "Linux"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception checking service status: {e}",
                "service_name": service_name
            }

    async def get_service_logs(self, lines: int = 50) -> Dict[str, Any]:
        """Get service logs with proper error handling."""
        try:
            if self.system == "darwin":
                return await self._get_macos_service_logs(lines)
            elif self.system == "linux":
                return await self._get_linux_service_logs(lines)
            else:
                return {"success": False, "error": f"Unsupported platform: {self.system}"}
        except Exception as e:
            logger.error("Service logs retrieval failed", error=str(e), exc_info=True)
            return {"success": False, "error": f"Logs retrieval failed: {e}"}

    async def _get_macos_service_logs(self, lines: int) -> Dict[str, Any]:
        """Get macOS service logs."""
        log_path = self.get_log_path()
        log_files = [
            log_path,
            Path(str(log_path) + ".error")
        ]
        
        logs = []
        for log_file in log_files:
            if log_file.exists():
                try:
                    content = log_file.read_text()
                    log_lines = content.split('\n')[-lines:]
                    logs.extend([f"[{log_file.name}] {line}" for line in log_lines if line.strip()])
                except Exception as e:
                    logs.append(f"[{log_file.name}] Error reading log: {e}")
        
        if not logs:
            logs = ["No logs found", "Check if service has been started"]
        
        return {
            "success": True,
            "service_id": self.service_id,
            "logs": logs,
            "lines_requested": lines
        }

    async def _get_linux_service_logs(self, lines: int) -> Dict[str, Any]:
        """Get Linux service logs."""
        service_name = f"{self.service_name}.service"
        
        try:
            cmd = ["journalctl", "--user", "-u", service_name, "-n", str(lines), "--no-pager"]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logs = stdout.decode().strip().split('\n')
            else:
                logs = [f"Error getting logs: {stderr.decode()}"]
            
            return {
                "success": True,
                "service_name": service_name,
                "logs": logs,
                "lines_requested": lines
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception getting service logs: {e}"
            }


# Create service manager instance
service_manager = MemexdServiceManager()


# CLI Commands
@service_app.command("install")
def install_service(
    auto_start: bool = typer.Option(
        True, "--auto-start/--no-auto-start", help="Start service automatically on boot"
    ),
) -> None:
    """Install the workspace daemon as a user service."""

    async def _install():
        result = await service_manager.install_service(auto_start=auto_start)
        
        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service installed successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}\n"
                    f"Binary: {result.get('binary_path', 'N/A')}\n"
                    f"Auto-start: {'Enabled' if auto_start else 'Disabled'}\n"
                    f"Platform: {service_manager.system.title()}",
                    title="Installation Success",
                    style="green",
                )
            )
        else:
            error_text = f"❌ Installation failed!\n\nError: {result['error']}"
            
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
def uninstall_service() -> None:
    """Uninstall the workspace daemon service."""

    async def _uninstall():
        result = await service_manager.uninstall_service()
        
        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service uninstalled successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}\n"
                    f"All service files removed",
                    title="Uninstall Success",
                    style="green",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"❌ Uninstall failed!\n\nError: {result['error']}",
                    title="Uninstall Error",
                    style="red",
                )
            )
            raise typer.Exit(1)

    handle_async_command(_uninstall())


@service_app.command("start")
def start_service() -> None:
    """Start the workspace daemon service."""

    async def _start():
        result = await service_manager.start_service()
        
        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service started successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}\n"
                    f"Status: Running",
                    title="Service Start",
                    style="green",
                )
            )
        else:
            error_text = f"❌ Failed to start service!\n\nError: {result['error']}"
            
            if 'suggestion' in result:
                error_text += f"\n\n💡 Suggestion:\n{result['suggestion']}"
                
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
def stop_service() -> None:
    """Stop the workspace daemon service."""

    async def _stop():
        result = await service_manager.stop_service()
        
        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service stopped successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}\n"
                    f"Status: Stopped",
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
    """Restart the workspace daemon service."""

    async def _restart():
        result = await service_manager.restart_service()
        
        if result["success"]:
            console.print(
                Panel.fit(
                    f"✅ Service restarted successfully!\n\n"
                    f"Service: {result.get('service_id', result.get('service_name'))}\n"
                    f"Status: Running",
                    title="Service Restart",
                    style="green",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"❌ Failed to restart service!\n\nError: {result['error']}",
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
    """Show workspace daemon service status."""

    async def _status():
        result = await service_manager.get_service_status()

        if result["success"]:
            service_name = result.get(
                "service_id", result.get("service_name", "workspace-daemon")
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
            elif status == "loaded":
                status_text = Text("Loaded (Not Running)", style="yellow")
            elif status == "not_installed":
                status_text = Text("Not Installed", style="dim")
            elif status == "not_loaded":
                status_text = Text("Installed (Not Loaded)", style="yellow")
            else:
                status_text = Text(status.title(), style="dim")

            table.add_row("Status", str(status_text))
            table.add_row("Platform", platform)
            table.add_row("Binary", str(service_manager.memexd_binary))

            if pid:
                table.add_row("PID", str(pid))

            console.print(table)

            if verbose and running:
                console.print("\n💡 Service is running and processing workspace events")
                
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
) -> None:
    """Show workspace daemon service logs."""

    async def _logs():
        result = await service_manager.get_service_logs(lines)

        if result["success"]:
            service_name = result.get(
                "service_id", result.get("service_name", "workspace-daemon")
            )
            logs = result.get("logs", [])

            console.print(f"\n📋 Recent logs for {service_name} (last {lines} lines):\n")

            if logs:
                for log_line in logs:
                    console.print(log_line)
            else:
                console.print("No logs available")

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