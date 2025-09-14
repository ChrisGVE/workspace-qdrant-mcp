"""
LSP Notification System

This module provides user notification functionality for missing LSP servers
when unsupported files are detected, with installation guidance and throttling.
"""

import json
from common.logging.loguru_config import get_logger
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Any
from enum import Enum

logger = get_logger(__name__)


class NotificationLevel(Enum):
    """Notification importance levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"


class NotificationHandler(Enum):
    """Available notification delivery methods."""
    CONSOLE = "console"
    FILE = "file"
    CALLBACK = "callback"


@dataclass
class LSPInstallationInfo:
    """Information about how to install a specific LSP server."""
    
    lsp_name: str
    description: str
    package_managers: Dict[str, str] = field(default_factory=dict)  # manager -> command
    manual_instructions: Dict[str, str] = field(default_factory=dict)  # platform -> instructions
    official_url: str = ""
    note: str = ""


@dataclass
class NotificationEntry:
    """A single notification entry with metadata."""
    
    file_extension: str
    lsp_name: str
    message: str
    level: NotificationLevel
    timestamp: float
    count: int = 1
    dismissed: bool = False
    installation_info: Optional[LSPInstallationInfo] = None


class LSPNotificationManager:
    """
    Manages user notifications for missing LSP servers with installation guidance.
    Provides throttling, platform-specific instructions, and dismiss functionality.
    """
    
    # Platform-specific installation instructions for common LSPs
    LSP_INSTALLATION_DB = {
        'rust-analyzer': LSPInstallationInfo(
            lsp_name='rust-analyzer',
            description='Rust language server providing IDE features for Rust',
            package_managers={
                'rustup': 'rustup component add rust-analyzer',
                'cargo': 'cargo install rust-analyzer',
                'brew': 'brew install rust-analyzer',
                'apt': 'sudo apt install rust-analyzer',
                'pacman': 'sudo pacman -S rust-analyzer',
                'dnf': 'sudo dnf install rust-analyzer',
                'scoop': 'scoop install rust-analyzer',
                'chocolatey': 'choco install rust-analyzer'
            },
            manual_instructions={
                'Darwin': 'Download from GitHub releases or install via Homebrew',
                'Linux': 'Install via package manager or download from GitHub releases', 
                'Windows': 'Download from GitHub releases or use package manager'
            },
            official_url='https://rust-analyzer.github.io/manual.html#installation',
            note='Recommended: Use rustup for automatic management'
        ),
        'ruff': LSPInstallationInfo(
            lsp_name='ruff',
            description='Extremely fast Python linter and code formatter',
            package_managers={
                'pip': 'pip install ruff',
                'pipx': 'pipx install ruff',
                'conda': 'conda install -c conda-forge ruff',
                'brew': 'brew install ruff',
                'cargo': 'cargo install ruff',
                'apt': 'sudo apt install ruff',
                'pacman': 'sudo pacman -S ruff',
                'scoop': 'scoop install ruff',
                'chocolatey': 'choco install ruff'
            },
            official_url='https://docs.astral.sh/ruff/installation/',
            note='Works with both Python linting and LSP functionality'
        ),
        'typescript-language-server': LSPInstallationInfo(
            lsp_name='typescript-language-server',
            description='TypeScript and JavaScript language server',
            package_managers={
                'npm': 'npm install -g typescript-language-server typescript',
                'yarn': 'yarn global add typescript-language-server typescript',
                'pnpm': 'pnpm add -g typescript-language-server typescript',
                'brew': 'brew install typescript-language-server',
                'pacman': 'sudo pacman -S typescript-language-server',
                'chocolatey': 'choco install typescript'
            },
            official_url='https://github.com/typescript-language-server/typescript-language-server',
            note='Requires TypeScript to be installed alongside'
        ),
        'pyright': LSPInstallationInfo(
            lsp_name='pyright',
            description='Static type checker and language server for Python',
            package_managers={
                'npm': 'npm install -g pyright',
                'yarn': 'yarn global add pyright',
                'pnpm': 'pnpm add -g pyright',
                'pip': 'pip install pyright',
                'pipx': 'pipx install pyright',
                'conda': 'conda install -c conda-forge pyright',
                'brew': 'brew install pyright'
            },
            official_url='https://github.com/microsoft/pyright',
            note='Microsoft\'s Python static type checker'
        ),
        'pylsp': LSPInstallationInfo(
            lsp_name='pylsp',
            description='Python LSP Server (community fork of python-language-server)',
            package_managers={
                'pip': 'pip install python-lsp-server[all]',
                'pipx': 'pipx install python-lsp-server[all]',
                'conda': 'conda install -c conda-forge python-lsp-server',
                'apt': 'sudo apt install python3-pylsp',
                'pacman': 'sudo pacman -S python-lsp-server'
            },
            official_url='https://github.com/python-lsp/python-lsp-server',
            note='Install with [all] for full feature support'
        ),
        'gopls': LSPInstallationInfo(
            lsp_name='gopls',
            description='Official Go language server',
            package_managers={
                'go': 'go install golang.org/x/tools/gopls@latest',
                'brew': 'brew install gopls',
                'apt': 'sudo apt install gopls',
                'pacman': 'sudo pacman -S gopls',
                'dnf': 'sudo dnf install gopls',
                'chocolatey': 'choco install gopls'
            },
            official_url='https://pkg.go.dev/golang.org/x/tools/gopls',
            note='Requires Go to be installed first'
        ),
        'clangd': LSPInstallationInfo(
            lsp_name='clangd',
            description='C/C++ language server from the LLVM project',
            package_managers={
                'apt': 'sudo apt install clangd',
                'brew': 'brew install llvm',
                'pacman': 'sudo pacman -S clang',
                'dnf': 'sudo dnf install clang-tools-extra',
                'chocolatey': 'choco install llvm',
                'scoop': 'scoop install llvm'
            },
            manual_instructions={
                'Linux': 'Usually available in distribution repositories as clangd or clang-tools',
                'Darwin': 'Install via Homebrew or Xcode command line tools',
                'Windows': 'Install LLVM package or Visual Studio with Clang support'
            },
            official_url='https://clangd.llvm.org/installation',
            note='Part of LLVM project, often bundled with development tools'
        ),
        'java-language-server': LSPInstallationInfo(
            lsp_name='java-language-server',
            description='Java language server (Eclipse JDT Language Server)',
            package_managers={
                'brew': 'brew install jdtls',
                'apt': 'sudo apt install eclipse-jdt-ls',
                'pacman': 'sudo pacman -S jdtls'
            },
            manual_instructions={
                'general': 'Download from Eclipse JDT Language Server releases on GitHub'
            },
            official_url='https://github.com/eclipse/eclipse.jdt.ls',
            note='Requires Java JDK to be installed'
        ),
        'lua-language-server': LSPInstallationInfo(
            lsp_name='lua-language-server',
            description='Lua language server with rich IDE features',
            package_managers={
                'brew': 'brew install lua-language-server',
                'pacman': 'sudo pacman -S lua-language-server',
                'apt': 'sudo apt install lua-language-server',
                'cargo': 'cargo install --git https://github.com/LuaLS/lua-language-server'
            },
            official_url='https://luals.github.io/',
            note='Supports Lua 5.1-5.4 and LuaJIT'
        ),
        'zls': LSPInstallationInfo(
            lsp_name='zls',
            description='Zig language server',
            package_managers={
                'zig': 'Install from Zig installation (may be included)',
                'brew': 'brew install zls',
                'cargo': 'cargo install zls'
            },
            manual_instructions={
                'general': 'Download from ZLS releases on GitHub or build from source'
            },
            official_url='https://github.com/zigtools/zls',
            note='Ensure Zig compiler is installed and up to date'
        ),
        'haskell-language-server': LSPInstallationInfo(
            lsp_name='haskell-language-server',
            description='Official Haskell language server',
            package_managers={
                'ghcup': 'ghcup install hls',
                'cabal': 'cabal install haskell-language-server',
                'stack': 'stack install haskell-language-server',
                'brew': 'brew install haskell-language-server',
                'pacman': 'sudo pacman -S haskell-language-server',
                'apt': 'sudo apt install haskell-language-server'
            },
            official_url='https://haskell-language-server.readthedocs.io/',
            note='Recommended: Install via GHCup for version management'
        )
    }
    
    def __init__(self, 
                 max_notifications_per_type: int = 3,
                 notification_cooldown: int = 300,  # 5 minutes
                 persist_file: Optional[str] = None,
                 default_handlers: Optional[List[NotificationHandler]] = None):
        """
        Initialize LSP notification manager.
        
        Args:
            max_notifications_per_type: Max notifications per file type per session
            notification_cooldown: Cooldown period in seconds between repeat notifications
            persist_file: Optional file to persist notification history
            default_handlers: Default notification delivery methods
        """
        self.max_notifications_per_type = max_notifications_per_type
        self.notification_cooldown = notification_cooldown
        self.persist_file = persist_file
        
        # Notification storage
        self.notifications: Dict[str, NotificationEntry] = {}
        self.dismissed_types: Set[str] = set()
        self.session_counts: Dict[str, int] = {}
        
        # Handlers
        self.default_handlers = default_handlers or [NotificationHandler.CONSOLE]
        self.custom_callbacks: Dict[str, Callable] = {}
        
        # Load persisted state if available
        self._load_persisted_state()
        
        logger.debug(f"LSP notification manager initialized with {len(self.default_handlers)} handlers")
    
    def _load_persisted_state(self) -> None:
        """Load notification history from persistent storage."""
        if not self.persist_file:
            return
        
        try:
            persist_path = Path(self.persist_file)
            if persist_path.exists():
                with open(persist_path, 'r') as f:
                    data = json.load(f)
                
                # Load dismissed types
                self.dismissed_types = set(data.get('dismissed_types', []))
                
                # Load notification history (recent only)
                current_time = time.time()
                for entry_data in data.get('notifications', []):
                    # Only load recent notifications (within cooldown period)
                    if current_time - entry_data['timestamp'] < self.notification_cooldown * 2:
                        entry = NotificationEntry(
                            file_extension=entry_data['file_extension'],
                            lsp_name=entry_data['lsp_name'],
                            message=entry_data['message'],
                            level=NotificationLevel(entry_data['level']),
                            timestamp=entry_data['timestamp'],
                            count=entry_data['count'],
                            dismissed=entry_data['dismissed']
                        )
                        self.notifications[entry_data['file_extension']] = entry
                
                logger.debug(f"Loaded {len(self.notifications)} notifications from persistence")
        except Exception as e:
            logger.warning(f"Failed to load persisted notification state: {e}")
    
    def _save_persisted_state(self) -> None:
        """Save notification history to persistent storage."""
        if not self.persist_file:
            return
        
        try:
            persist_path = Path(self.persist_file)
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'dismissed_types': list(self.dismissed_types),
                'notifications': [
                    {
                        'file_extension': entry.file_extension,
                        'lsp_name': entry.lsp_name,
                        'message': entry.message,
                        'level': entry.level.value,
                        'timestamp': entry.timestamp,
                        'count': entry.count,
                        'dismissed': entry.dismissed
                    }
                    for entry in self.notifications.values()
                ]
            }
            
            with open(persist_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved notification state to {persist_path}")
        except Exception as e:
            logger.warning(f"Failed to save notification state: {e}")
    
    def _should_notify(self, file_extension: str) -> bool:
        """Check if we should send a notification for this file type."""
        # Check if type is dismissed
        if file_extension in self.dismissed_types:
            return False
        
        # Check session count limit
        session_count = self.session_counts.get(file_extension, 0)
        if session_count >= self.max_notifications_per_type:
            return False
        
        # Check cooldown period
        if file_extension in self.notifications:
            last_notification = self.notifications[file_extension]
            if time.time() - last_notification.timestamp < self.notification_cooldown:
                return False
        
        return True
    
    def _get_platform_specific_instructions(self, lsp_name: str) -> Dict[str, str]:
        """Get platform-specific installation instructions."""
        if lsp_name not in self.LSP_INSTALLATION_DB:
            return {}
        
        install_info = self.LSP_INSTALLATION_DB[lsp_name]
        current_platform = platform.system()
        
        instructions = {}
        
        # Add package manager instructions
        for manager, command in install_info.package_managers.items():
            instructions[f"via_{manager}"] = command
        
        # Add platform-specific manual instructions
        if current_platform in install_info.manual_instructions:
            instructions["manual"] = install_info.manual_instructions[current_platform]
        elif "general" in install_info.manual_instructions:
            instructions["manual"] = install_info.manual_instructions["general"]
        
        # Add official URL
        if install_info.official_url:
            instructions["official_docs"] = f"Documentation: {install_info.official_url}"
        
        # Add note if available
        if install_info.note:
            instructions["note"] = install_info.note
        
        return instructions
    
    def _format_installation_message(self, file_extension: str, lsp_name: str) -> str:
        """Format a user-friendly installation message."""
        if lsp_name not in self.LSP_INSTALLATION_DB:
            return f"No LSP found for {file_extension} files. Consider installing an LSP server for better IDE support."
        
        install_info = self.LSP_INSTALLATION_DB[lsp_name]
        current_platform = platform.system()
        
        message_parts = [
            f"No LSP found for {file_extension} files.",
            f"Consider installing {lsp_name}: {install_info.description}",
            "",
            "Installation options:"
        ]
        
        # Add the most relevant installation methods for current platform
        instructions = self._get_platform_specific_instructions(lsp_name)
        
        # Prioritize common package managers by platform
        platform_priority = {
            'Darwin': ['brew', 'via_npm', 'via_pip', 'via_cargo'],
            'Linux': ['via_apt', 'via_pacman', 'via_dnf', 'via_npm', 'via_pip'],
            'Windows': ['via_chocolatey', 'via_scoop', 'via_npm', 'via_pip']
        }
        
        priority_list = platform_priority.get(current_platform, [])
        
        # Add prioritized instructions first
        for key in priority_list:
            if key in instructions:
                message_parts.append(f"  • {instructions[key]}")
        
        # Add other instructions
        for key, instruction in instructions.items():
            if key not in priority_list and not key.startswith('official') and key != 'note':
                message_parts.append(f"  • {instruction}")
        
        # Add official documentation link
        if 'official_docs' in instructions:
            message_parts.extend(["", instructions['official_docs']])
        
        # Add note if available
        if 'note' in instructions:
            message_parts.extend(["", f"Note: {instructions['note']}"])
        
        return "\n".join(message_parts)
    
    def notify_missing_lsp(self, 
                          file_extension: str, 
                          lsp_name: str,
                          level: NotificationLevel = NotificationLevel.INFO,
                          custom_handlers: Optional[List[NotificationHandler]] = None) -> bool:
        """
        Notify about a missing LSP server for a file type.
        
        Args:
            file_extension: File extension that triggered the notification
            lsp_name: Name of the recommended LSP server
            level: Notification importance level
            custom_handlers: Override default notification handlers
            
        Returns:
            True if notification was sent, False if throttled
        """
        if not self._should_notify(file_extension):
            logger.debug(f"Notification throttled for {file_extension}")
            return False
        
        # Create notification message
        message = self._format_installation_message(file_extension, lsp_name)
        
        # Create notification entry
        if file_extension in self.notifications:
            # Update existing entry
            entry = self.notifications[file_extension]
            entry.count += 1
            entry.timestamp = time.time()
            entry.message = message
        else:
            # Create new entry
            installation_info = self.LSP_INSTALLATION_DB.get(lsp_name)
            entry = NotificationEntry(
                file_extension=file_extension,
                lsp_name=lsp_name,
                message=message,
                level=level,
                timestamp=time.time(),
                installation_info=installation_info
            )
            self.notifications[file_extension] = entry
        
        # Update session count
        self.session_counts[file_extension] = self.session_counts.get(file_extension, 0) + 1
        
        # Send notification through handlers
        handlers = custom_handlers or self.default_handlers
        for handler in handlers:
            self._send_notification(entry, handler)
        
        # Persist state
        self._save_persisted_state()
        
        logger.info(f"Sent LSP notification for {file_extension} (LSP: {lsp_name})")
        return True
    
    def _send_notification(self, entry: NotificationEntry, handler: NotificationHandler) -> None:
        """Send notification through specified handler."""
        try:
            if handler == NotificationHandler.CONSOLE:
                self._send_console_notification(entry)
            elif handler == NotificationHandler.FILE:
                self._send_file_notification(entry)
            elif handler == NotificationHandler.CALLBACK:
                self._send_callback_notification(entry)
        except Exception as e:
            logger.error(f"Failed to send notification via {handler}: {e}")
    
    def _send_console_notification(self, entry: NotificationEntry) -> None:
        """Send notification to console/logger."""
        level_map = {
            NotificationLevel.INFO: logger.info,
            NotificationLevel.WARNING: logger.warning,
            NotificationLevel.ERROR: logger.error
        }
        
        log_func = level_map.get(entry.level, logger.info)
        log_func(f"LSP Notification:\n{entry.message}")
    
    def _send_file_notification(self, entry: NotificationEntry) -> None:
        """Send notification to file."""
        if not self.persist_file:
            logger.warning("File notification requested but no persist_file configured")
            return
        
        log_file = Path(self.persist_file).parent / "notifications.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp))
        log_entry = f"[{timestamp}] {entry.level.value.upper()}: {entry.file_extension} -> {entry.lsp_name}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
    
    def _send_callback_notification(self, entry: NotificationEntry) -> None:
        """Send notification through registered callbacks."""
        for callback_name, callback_func in self.custom_callbacks.items():
            try:
                callback_func(entry)
            except Exception as e:
                logger.error(f"Callback {callback_name} failed: {e}")
    
    def dismiss_file_type(self, file_extension: str) -> None:
        """Dismiss future notifications for a specific file type."""
        self.dismissed_types.add(file_extension)
        
        # Mark existing notification as dismissed
        if file_extension in self.notifications:
            self.notifications[file_extension].dismissed = True
        
        self._save_persisted_state()
        logger.info(f"Dismissed notifications for {file_extension}")
    
    def undismiss_file_type(self, file_extension: str) -> None:
        """Re-enable notifications for a specific file type."""
        self.dismissed_types.discard(file_extension)
        
        # Reset session count
        self.session_counts.pop(file_extension, None)
        
        # Mark existing notification as not dismissed
        if file_extension in self.notifications:
            self.notifications[file_extension].dismissed = False
        
        self._save_persisted_state()
        logger.info(f"Re-enabled notifications for {file_extension}")
    
    def register_callback(self, name: str, callback: Callable[[NotificationEntry], None]) -> None:
        """Register a custom notification callback."""
        self.custom_callbacks[name] = callback
        logger.debug(f"Registered notification callback: {name}")
    
    def unregister_callback(self, name: str) -> None:
        """Unregister a custom notification callback."""
        self.custom_callbacks.pop(name, None)
        logger.debug(f"Unregistered notification callback: {name}")
    
    def get_notification_history(self) -> List[NotificationEntry]:
        """Get list of all notifications sent."""
        return list(self.notifications.values())
    
    def get_dismissed_types(self) -> Set[str]:
        """Get set of dismissed file types."""
        return self.dismissed_types.copy()
    
    def clear_history(self) -> None:
        """Clear notification history and reset session counts."""
        self.notifications.clear()
        self.session_counts.clear()
        self._save_persisted_state()
        logger.info("Cleared notification history")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics and summary."""
        total_notifications = len(self.notifications)
        dismissed_count = len(self.dismissed_types)
        active_count = sum(1 for entry in self.notifications.values() if not entry.dismissed)
        
        extension_counts = {}
        for entry in self.notifications.values():
            ext = entry.file_extension
            extension_counts[ext] = extension_counts.get(ext, 0) + entry.count
        
        return {
            'total_notifications': total_notifications,
            'active_notifications': active_count,
            'dismissed_types': dismissed_count,
            'session_counts': dict(self.session_counts),
            'extension_counts': extension_counts,
            'available_lsps': list(self.LSP_INSTALLATION_DB.keys()),
            'handlers_configured': [handler.value for handler in self.default_handlers],
            'custom_callbacks': list(self.custom_callbacks.keys())
        }


# Global instance for convenient access
_default_notification_manager: Optional[LSPNotificationManager] = None


def get_default_notification_manager() -> LSPNotificationManager:
    """Get the default global LSP notification manager instance."""
    global _default_notification_manager
    if _default_notification_manager is None:
        _default_notification_manager = LSPNotificationManager()
    return _default_notification_manager


def notify_missing_lsp(file_extension: str, lsp_name: str, level: NotificationLevel = NotificationLevel.INFO) -> bool:
    """Convenience function to notify about missing LSP using default manager."""
    return get_default_notification_manager().notify_missing_lsp(file_extension, lsp_name, level)


def dismiss_file_type(file_extension: str) -> None:
    """Convenience function to dismiss file type using default manager."""
    get_default_notification_manager().dismiss_file_type(file_extension)