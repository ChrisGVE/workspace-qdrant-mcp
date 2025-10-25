"""
Copilot/Codex session detection for context injection.

This module provides utilities to detect when code is running within a GitHub Copilot
or Codex session, enabling automatic context injection via code comments and directives.

Detection targets:
- GitHub Copilot (VSCode extension)
- Codex API sessions
- Other AI coding assistants (JetBrains AI, Cursor, etc.)
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loguru import logger


class CopilotSessionType(Enum):
    """Types of AI coding assistant sessions."""

    GITHUB_COPILOT = "github_copilot"  # GitHub Copilot in VSCode/IDE
    CODEX_API = "codex_api"  # Direct Codex API usage
    CURSOR = "cursor"  # Cursor editor
    JETBRAINS_AI = "jetbrains_ai"  # JetBrains AI Assistant
    TABNINE = "tabnine"  # Tabnine
    UNKNOWN = "unknown"  # Unknown AI assistant


@dataclass
class CopilotSession:
    """
    Information about a detected Copilot/Codex session.

    Attributes:
        is_active: Whether an AI coding assistant session is detected
        session_type: Type of AI coding assistant if detected
        ide_name: Name of the IDE (e.g., "vscode", "idea", "cursor")
        ide_version: Version of the IDE if available
        detection_method: How the session was detected
        workspace_path: Workspace/project path if available
    """

    is_active: bool
    session_type: CopilotSessionType | None = None
    ide_name: str | None = None
    ide_version: str | None = None
    detection_method: str | None = None
    workspace_path: Path | None = None


class CopilotDetector:
    """
    Detects Copilot/Codex and other AI coding assistant sessions.

    This detector checks for AI coding assistant sessions by examining:
    1. Environment variables (VSCODE_PID, TERM_PROGRAM, IDE-specific vars)
    2. Process information (parent process name, IDE processes)
    3. LSP server detection (Copilot LSP, language servers)
    4. IDE-specific markers (socket files, configuration files)
    5. API usage patterns (Codex API headers, authentication)

    The detector uses multiple methods for robustness and can identify
    various AI coding assistants beyond just GitHub Copilot.
    """

    # Environment variable names for IDE detection
    ENV_VSCODE_PID = "VSCODE_PID"
    ENV_VSCODE_IPC = "VSCODE_IPC_HOOK"
    ENV_TERM_PROGRAM = "TERM_PROGRAM"
    ENV_JETBRAINS_IDE = "JETBRAINS_IDE"
    ENV_CURSOR_SESSION = "CURSOR_SESSION"

    # IDE process names to detect
    IDE_PROCESS_NAMES = {
        "code": "vscode",
        "code-insiders": "vscode-insiders",
        "cursor": "cursor",
        "idea": "intellij",
        "pycharm": "pycharm",
        "webstorm": "webstorm",
        "rider": "rider",
        "goland": "goland",
        "clion": "clion",
    }

    # Copilot-specific markers
    COPILOT_LSP_SOCKET_PATTERNS = [
        "/tmp/copilot-lsp-*.sock",
        "~/.config/github-copilot/",
    ]

    @classmethod
    def detect(cls) -> CopilotSession:
        """
        Detect if currently running in an AI coding assistant session.

        Uses multiple detection methods in order of reliability:
        1. Environment variable detection (IDE-specific)
        2. Parent process inspection
        3. LSP server detection
        4. IDE configuration file detection

        Returns:
            CopilotSession with detection results and metadata
        """
        # Method 1: Check VSCode environment variables
        vscode_session = cls._detect_vscode()
        if vscode_session.is_active:
            return vscode_session

        # Method 2: Check Cursor editor
        cursor_session = cls._detect_cursor()
        if cursor_session.is_active:
            return cursor_session

        # Method 3: Check JetBrains IDEs
        jetbrains_session = cls._detect_jetbrains()
        if jetbrains_session.is_active:
            return jetbrains_session

        # Method 4: Check parent process for IDE
        process_session = cls._detect_from_process()
        if process_session.is_active:
            return process_session

        # Method 5: Check for Copilot LSP server
        lsp_session = cls._detect_copilot_lsp()
        if lsp_session.is_active:
            return lsp_session

        # No AI coding assistant detected
        logger.debug("No Copilot/AI coding assistant session detected")
        return CopilotSession(
            is_active=False,
            session_type=None,
            detection_method=None,
        )

    @classmethod
    def is_active(cls) -> bool:
        """
        Check if currently running in a Copilot/AI coding assistant session.

        Convenience method that returns just the boolean active status.

        Returns:
            True if AI coding assistant session is detected, False otherwise
        """
        return cls.detect().is_active

    @classmethod
    def _detect_vscode(cls) -> CopilotSession:
        """
        Detect VSCode session via environment variables.

        Returns:
            CopilotSession if VSCode detected, inactive session otherwise
        """
        # Check VSCODE_PID environment variable
        vscode_pid = os.environ.get(cls.ENV_VSCODE_PID)
        vscode_ipc = os.environ.get(cls.ENV_VSCODE_IPC)
        term_program = os.environ.get(cls.ENV_TERM_PROGRAM)

        if vscode_pid or vscode_ipc:
            logger.debug("VSCode session detected via environment variables")

            # Try to determine if it's VSCode or VSCode Insiders
            ide_name = "vscode"
            if vscode_ipc and "insiders" in vscode_ipc.lower():
                ide_name = "vscode-insiders"

            return CopilotSession(
                is_active=True,
                session_type=CopilotSessionType.GITHUB_COPILOT,
                ide_name=ide_name,
                detection_method="environment_variable_vscode",
                workspace_path=cls._get_workspace_from_env(),
            )

        if term_program and "vscode" in term_program.lower():
            logger.debug("VSCode session detected via TERM_PROGRAM")
            return CopilotSession(
                is_active=True,
                session_type=CopilotSessionType.GITHUB_COPILOT,
                ide_name="vscode",
                detection_method="environment_variable_term",
                workspace_path=cls._get_workspace_from_env(),
            )

        return CopilotSession(is_active=False)

    @classmethod
    def _detect_cursor(cls) -> CopilotSession:
        """
        Detect Cursor editor session.

        Returns:
            CopilotSession if Cursor detected, inactive session otherwise
        """
        cursor_session = os.environ.get(cls.ENV_CURSOR_SESSION)

        if cursor_session:
            logger.debug("Cursor editor session detected")
            return CopilotSession(
                is_active=True,
                session_type=CopilotSessionType.CURSOR,
                ide_name="cursor",
                detection_method="environment_variable_cursor",
                workspace_path=cls._get_workspace_from_env(),
            )

        return CopilotSession(is_active=False)

    @classmethod
    def _detect_jetbrains(cls) -> CopilotSession:
        """
        Detect JetBrains IDE session.

        Returns:
            CopilotSession if JetBrains IDE detected, inactive session otherwise
        """
        jetbrains_ide = os.environ.get(cls.ENV_JETBRAINS_IDE)

        if jetbrains_ide:
            logger.debug(f"JetBrains IDE detected: {jetbrains_ide}")
            return CopilotSession(
                is_active=True,
                session_type=CopilotSessionType.JETBRAINS_AI,
                ide_name=jetbrains_ide.lower(),
                detection_method="environment_variable_jetbrains",
                workspace_path=cls._get_workspace_from_env(),
            )

        return CopilotSession(is_active=False)

    @classmethod
    def _detect_from_process(cls) -> CopilotSession:
        """
        Detect IDE session by inspecting parent processes.

        Returns:
            CopilotSession if IDE process found, inactive session otherwise
        """
        try:
            import psutil

            current_process = psutil.Process(os.getpid())

            # Walk up the process tree to find IDE
            process = current_process
            max_depth = 10  # Limit depth to avoid infinite loops

            for _ in range(max_depth):
                parent = process.parent()
                if parent is None:
                    break

                parent_name = parent.name().lower()

                # Check against known IDE process names
                for proc_name, ide_name in cls.IDE_PROCESS_NAMES.items():
                    if proc_name in parent_name:
                        logger.debug(f"IDE detected from process: {parent_name} -> {ide_name}")

                        # Determine session type based on IDE
                        session_type = CopilotSessionType.GITHUB_COPILOT
                        if "cursor" in ide_name:
                            session_type = CopilotSessionType.CURSOR
                        elif any(jb in ide_name for jb in ["intellij", "pycharm", "webstorm", "rider", "goland", "clion"]):
                            session_type = CopilotSessionType.JETBRAINS_AI

                        return CopilotSession(
                            is_active=True,
                            session_type=session_type,
                            ide_name=ide_name,
                            detection_method="parent_process",
                        )

                process = parent

        except ImportError:
            logger.debug("psutil not available for process detection")
        except Exception as e:
            logger.debug(f"Error inspecting processes: {e}")

        return CopilotSession(is_active=False)

    @classmethod
    def _detect_copilot_lsp(cls) -> CopilotSession:
        """
        Detect Copilot LSP server via socket files or running processes.

        Returns:
            CopilotSession if Copilot LSP detected, inactive session otherwise
        """
        # Check for Copilot LSP socket files
        import glob

        for pattern in cls.COPILOT_LSP_SOCKET_PATTERNS:
            expanded_pattern = os.path.expanduser(pattern)
            matches = glob.glob(expanded_pattern)

            if matches:
                logger.debug(f"Copilot LSP detected via socket: {matches[0]}")
                return CopilotSession(
                    is_active=True,
                    session_type=CopilotSessionType.GITHUB_COPILOT,
                    ide_name="unknown",
                    detection_method="lsp_socket",
                )

        # Check for Copilot LSP process
        try:
            import psutil

            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    proc_name = proc.info['name'].lower() if proc.info['name'] else ""
                    cmdline = " ".join(proc.info['cmdline']).lower() if proc.info['cmdline'] else ""

                    if "copilot" in proc_name or "copilot" in cmdline:
                        logger.debug(f"Copilot LSP process detected: {proc_name}")
                        return CopilotSession(
                            is_active=True,
                            session_type=CopilotSessionType.GITHUB_COPILOT,
                            ide_name="unknown",
                            detection_method="lsp_process",
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error detecting Copilot LSP: {e}")

        return CopilotSession(is_active=False)

    @classmethod
    def _get_workspace_from_env(cls) -> Path | None:
        """
        Get workspace/project path from environment variables.

        Returns:
            Path to workspace if found, None otherwise
        """
        # Check common workspace environment variables
        workspace_vars = [
            "VSCODE_WORKSPACE",
            "PWD",
            "OLDPWD",
        ]

        for var in workspace_vars:
            workspace = os.environ.get(var)
            if workspace and os.path.isdir(workspace):
                return Path(workspace)

        return None

    @classmethod
    def get_comment_style(cls, language: str) -> tuple[str, str]:
        """
        Get comment style for injecting rules into code.

        Args:
            language: Programming language (e.g., "python", "javascript", "rust")

        Returns:
            Tuple of (line_comment_prefix, block_comment_style)
            Block comment style is tuple of (start, end) or None

        Examples:
            >>> CopilotDetector.get_comment_style("python")
            ('#', None)
            >>> CopilotDetector.get_comment_style("javascript")
            ('//', ('/*', '*/'))
        """
        comment_styles = {
            # C-style languages
            "c": ("//", ("/*", "*/")),
            "cpp": ("//", ("/*", "*/")),
            "cxx": ("//", ("/*", "*/")),
            "cc": ("//", ("/*", "*/")),
            "java": ("//", ("/*", "*/")),
            "javascript": ("//", ("/*", "*/")),
            "typescript": ("//", ("/*", "*/")),
            "jsx": ("//", ("/*", "*/")),
            "tsx": ("//", ("/*", "*/")),
            "go": ("//", ("/*", "*/")),
            "rust": ("//", ("/*", "*/")),
            "swift": ("//", ("/*", "*/")),
            "kotlin": ("//", ("/*", "*/")),
            "scala": ("//", ("/*", "*/")),
            "dart": ("//", ("/*", "*/")),
            "php": ("//", ("/*", "*/")),
            "csharp": ("//", ("/*", "*/")),
            "objective-c": ("//", ("/*", "*/")),

            # Hash-style languages
            "python": ("#", ('"""', '"""')),
            "ruby": ("#", ("=begin", "=end")),
            "perl": ("#", ("=pod", "=cut")),
            "shell": ("#", None),
            "bash": ("#", None),
            "zsh": ("#", None),
            "fish": ("#", None),
            "powershell": ("#", ("<#", "#>")),
            "yaml": ("#", None),
            "toml": ("#", None),
            "makefile": ("#", None),
            "dockerfile": ("#", None),
            "r": ("#", None),
            "julia": ("#", ("\"\"\"", "\"\"\"")),

            # Lisp-style languages
            "lisp": (";", None),
            "scheme": (";", None),
            "clojure": (";", None),
            "elisp": (";", None),

            # ML-style languages
            "ocaml": ("(*", ("(*", "*)")),
            "fsharp": ("//", ("(*", "*)")),
            "haskell": ("--", ("{-", "-}")),
            "elm": ("--", ("{-", "-}")),

            # Other languages
            "lua": ("--", ("--[[", "]]")),
            "sql": ("--", ("/*", "*/")),
            "html": ("<!--", ("<!--", "-->")),
            "xml": ("<!--", ("<!--", "-->")),
            "css": ("/*", ("/*", "*/")),
            "scss": ("//", ("/*", "*/")),
            "less": ("//", ("/*", "*/")),
            "vim": ('"', None),
            "latex": ("%", None),
            "matlab": ("%", ("%{", "%}")),
            "fortran": ("!", None),
            "ada": ("--", None),
            "pascal": ("//", ("{", "}")),
            "erlang": ("%", None),
            "elixir": ("#", None),
            "zig": ("//", None),
            "nim": ("#", ("##[", "]##")),
            "crystal": ("#", None),
            "groovy": ("//", ("/*", "*/")),
        }

        lang_lower = language.lower()
        return comment_styles.get(lang_lower, ("#", None))


# Convenience functions


def is_copilot_session() -> bool:
    """
    Check if currently running in a Copilot/AI coding assistant session.

    Convenience function that wraps CopilotDetector.is_active().

    Returns:
        True if AI coding assistant session is detected, False otherwise

    Example:
        >>> from context_injection import is_copilot_session
        >>> if is_copilot_session():
        ...     # Inject rules via code comments
        ...     pass
    """
    return CopilotDetector.is_active()


def get_copilot_session() -> CopilotSession:
    """
    Get detailed information about the current Copilot/AI session.

    Convenience function that wraps CopilotDetector.detect().

    Returns:
        CopilotSession with detection results and metadata

    Example:
        >>> from context_injection import get_copilot_session
        >>> session = get_copilot_session()
        >>> if session.is_active:
        ...     print(f"IDE: {session.ide_name}")
        ...     print(f"Type: {session.session_type.value}")
        ...     print(f"Detected via: {session.detection_method}")
    """
    return CopilotDetector.detect()


def get_code_comment_prefix(language: str) -> str:
    """
    Get the line comment prefix for a programming language.

    Args:
        language: Programming language name

    Returns:
        Comment prefix string (e.g., "#", "//", "--")

    Example:
        >>> get_code_comment_prefix("python")
        '#'
        >>> get_code_comment_prefix("javascript")
        '//'
    """
    prefix, _ = CopilotDetector.get_comment_style(language)
    return prefix
