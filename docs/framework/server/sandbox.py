"""Secure code sandbox for executing documentation examples."""

import ast
import sys
import io
import time
import asyncio
import subprocess
import tempfile
import resource
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr


class SecurityError(Exception):
    """Raised when code violates security constraints."""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class CodeSandbox:
    """Secure sandbox for executing code examples."""

    def __init__(self, timeout: int = 30, memory_limit: int = 128):
        """Initialize the code sandbox.

        Args:
            timeout: Maximum execution time in seconds
            memory_limit: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.memory_limit = memory_limit * 1024 * 1024  # Convert to bytes

        # Define allowed modules and functions
        self.allowed_modules = {
            'builtins',
            'math',
            'random',
            'string',
            'json',
            'datetime',
            'collections',
            'itertools',
            'functools',
            're',
            'pathlib',
            'typing',
        }

        # Dangerous functions to block
        self.blocked_names = {
            'eval',
            'exec',
            'compile',
            '__import__',
            'open',
            'file',
            'input',
            'raw_input',
            'execfile',
            'reload',
            'vars',
            'globals',
            'locals',
            'dir',
            'hasattr',
            'getattr',
            'setattr',
            'delattr',
            'callable',
            'exit',
            'quit',
            'help',
            'license',
            'credits',
        }

        # AST node types to block
        self.blocked_ast_nodes = {
            ast.Import,      # Block all imports for simplicity
            ast.ImportFrom,
            ast.Global,      # Block global declarations
            ast.Nonlocal,    # Block nonlocal declarations
        }

    def is_available(self) -> bool:
        """Check if the sandbox is available."""
        try:
            # Test basic Python execution
            result = subprocess.run([sys.executable, '-c', 'print("test")'],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def validate_code(self, code: str) -> None:
        """Validate code for security issues.

        Args:
            code: Code to validate

        Raises:
            SecurityError: If code violates security constraints
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SecurityError(f"Syntax error: {e}")

        # Check for blocked AST node types
        for node in ast.walk(tree):
            if type(node) in self.blocked_ast_nodes:
                raise SecurityError(f"Blocked operation: {type(node).__name__}")

            # Check for blocked function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.blocked_names:
                        raise SecurityError(f"Blocked function: {node.func.id}")

            # Check for attribute access on dangerous modules
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in ['os', 'sys', 'subprocess', 'socket']:
                        raise SecurityError(f"Blocked module access: {node.value.id}.{node.attr}")

            # Check for dangerous string operations
            if isinstance(node, ast.Str):
                dangerous_patterns = ['__', 'eval', 'exec', 'import']
                for pattern in dangerous_patterns:
                    if pattern in node.s:
                        raise SecurityError(f"Potentially dangerous string: {node.s[:50]}...")

    async def execute_code(self, code: str, language: str = "python",
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute code in the sandbox.

        Args:
            code: Code to execute
            language: Programming language (only 'python' supported)
            context: Additional context variables

        Returns:
            Dictionary with execution results

        Raises:
            SecurityError: If code violates security constraints
            ExecutionTimeoutError: If execution times out
        """
        if language != "python":
            raise ValueError(f"Unsupported language: {language}")

        # Validate code security
        self.validate_code(code)

        # Prepare execution context
        exec_context = self._prepare_context(context or {})

        # Execute code
        start_time = time.time()

        try:
            result = await self._execute_python_code(code, exec_context)
            execution_time = time.time() - start_time

            return {
                'result': result.get('result'),
                'output': result.get('output'),
                'error': result.get('error'),
                'execution_time': execution_time,
                'memory_used': result.get('memory_used'),
            }

        except asyncio.TimeoutError:
            raise ExecutionTimeoutError(f"Code execution timed out after {self.timeout}s")

    async def _execute_python_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code with restrictions."""
        # Create isolated execution environment
        exec_globals = {
            '__builtins__': self._create_restricted_builtins(),
            **context
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result_value = None
        error_message = None

        try:
            # Set resource limits
            if hasattr(resource, 'RLIMIT_AS'):
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))

            # Redirect output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to evaluate as expression first
                try:
                    # Compile and execute as expression
                    compiled = compile(code, '<sandbox>', 'eval')
                    result_value = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, eval, compiled, exec_globals
                        ),
                        timeout=self.timeout
                    )
                except SyntaxError:
                    # If it's not a valid expression, execute as statements
                    compiled = compile(code, '<sandbox>', 'exec')
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, exec, compiled, exec_globals
                        ),
                        timeout=self.timeout
                    )
                    # Look for result in locals
                    if 'result' in exec_globals:
                        result_value = exec_globals['result']

        except Exception as e:
            error_message = str(e)

        # Get memory usage (approximate)
        memory_used = None
        if hasattr(resource, 'getrusage'):
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                memory_used = usage.ru_maxrss  # Peak memory usage
            except Exception:
                pass

        return {
            'result': str(result_value) if result_value is not None else None,
            'output': stdout_capture.getvalue(),
            'error': error_message or stderr_capture.getvalue() or None,
            'memory_used': memory_used
        }

    def _create_restricted_builtins(self) -> Dict[str, Any]:
        """Create a restricted builtins dictionary."""
        # Start with safe builtins
        safe_builtins = {
            # Basic types
            'bool', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'set', 'frozenset',
            # Type checking
            'type', 'isinstance', 'issubclass',
            # Iteration
            'iter', 'next', 'enumerate', 'zip', 'range', 'reversed',
            # Math
            'abs', 'divmod', 'max', 'min', 'pow', 'round', 'sum',
            # Containers
            'len', 'sorted', 'any', 'all',
            # String/formatting
            'repr', 'str', 'format', 'chr', 'ord',
            # Functional
            'map', 'filter',
            # Constants
            'True', 'False', 'None',
            # Exceptions
            'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
            'AttributeError', 'RuntimeError', 'NotImplementedError',
        }

        # Build restricted builtins dict
        original_builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        restricted = {}

        for name in safe_builtins:
            if name in original_builtins:
                restricted[name] = original_builtins[name]

        # Add some safe modules
        import math
        restricted['math'] = math

        import random
        # Create restricted random module
        class RestrictedRandom:
            def __init__(self):
                self._random = random.Random()

            def random(self):
                return self._random.random()

            def randint(self, a, b):
                return self._random.randint(a, b)

            def choice(self, seq):
                return self._random.choice(seq)

            def shuffle(self, seq):
                return self._random.shuffle(seq)

        restricted['random'] = RestrictedRandom()

        return restricted

    def _prepare_context(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare execution context with user variables."""
        # Validate user context
        safe_context = {}

        for key, value in user_context.items():
            # Only allow safe types
            if isinstance(value, (int, float, str, bool, list, dict, tuple, set, type(None))):
                safe_context[key] = value
            else:
                # Convert to string representation for safety
                safe_context[key] = str(value)

        return safe_context

    async def execute_file(self, file_path: str,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a Python file in the sandbox.

        Args:
            file_path: Path to Python file
            context: Additional context variables

        Returns:
            Dictionary with execution results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                code = f.read()

        return await self.execute_code(code, context=context)

    def validate_examples(self, examples: List[str]) -> List[Dict[str, Any]]:
        """Validate a list of code examples.

        Args:
            examples: List of code examples

        Returns:
            List of validation results
        """
        results = []

        for i, example in enumerate(examples):
            result = {
                'index': i,
                'example': example[:100] + "..." if len(example) > 100 else example,
                'valid': False,
                'error': None,
                'suggestions': []
            }

            try:
                self.validate_code(example)
                # Also try to compile it
                compile(example, f'<example_{i}>', 'exec')
                result['valid'] = True
            except SecurityError as e:
                result['error'] = f"Security issue: {e}"
                result['suggestions'].append("Remove dangerous operations")
            except SyntaxError as e:
                result['error'] = f"Syntax error: {e}"
                result['suggestions'].append("Fix syntax errors")
            except Exception as e:
                result['error'] = f"Validation error: {e}"

            results.append(result)

        return results

    async def test_code_examples(self, examples: List[str],
                                context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Test code examples by executing them.

        Args:
            examples: List of code examples to test
            context: Additional context for execution

        Returns:
            List of test results
        """
        results = []

        for i, example in enumerate(examples):
            result = {
                'index': i,
                'example': example[:100] + "..." if len(example) > 100 else example,
                'success': False,
                'output': None,
                'error': None,
                'execution_time': None
            }

            try:
                exec_result = await self.execute_code(example, context=context)
                result.update({
                    'success': exec_result.get('error') is None,
                    'output': exec_result.get('output') or exec_result.get('result'),
                    'error': exec_result.get('error'),
                    'execution_time': exec_result.get('execution_time')
                })
            except Exception as e:
                result['error'] = str(e)

            results.append(result)

        return results

    def get_safe_modules(self) -> List[str]:
        """Get list of modules that are safe to use.

        Returns:
            List of safe module names
        """
        return list(self.allowed_modules)

    def get_blocked_operations(self) -> List[str]:
        """Get list of blocked operations.

        Returns:
            List of blocked operation names
        """
        return list(self.blocked_names)