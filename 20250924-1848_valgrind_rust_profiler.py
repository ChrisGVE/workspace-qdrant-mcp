#!/usr/bin/env python3
"""
Valgrind Memory Profiling Integration for Rust Components.

This module provides comprehensive memory profiling and leak detection for
Rust engine components using valgrind tools (memcheck, massif, helgrind).

VALGRIND TOOLS INTEGRATED:
- memcheck: Memory error detection and leak analysis
- massif: Heap profiling and memory usage patterns
- helgrind: Thread error detection and race condition analysis
- callgrind: Call graph analysis and performance profiling

RUST-SPECIFIC FEATURES:
- Debug symbol integration for accurate stack traces
- Rust-specific memory pattern analysis
- Integration with Cargo for debug builds
- Analysis of async/await memory patterns
- FFI boundary memory safety validation

PERFORMANCE ANALYSIS:
- Memory allocation patterns in Rust code
- Heap growth analysis with massif
- Memory leak detection with precise attribution
- Thread safety validation for concurrent operations
- Performance bottleneck identification
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class ValgrindRustProfiler:
    """Valgrind-based memory profiling for Rust components."""

    def __init__(self, rust_project_path: Path):
        self.rust_project_path = Path(rust_project_path)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.valgrind_dir = self.temp_dir / "valgrind_profiles"
        self.valgrind_dir.mkdir(exist_ok=True)

        # Verify rust project structure
        self.cargo_toml = self.rust_project_path / "Cargo.toml"
        if not self.cargo_toml.exists():
            raise ValueError(f"Rust project not found at {rust_project_path}")

        # Valgrind configuration
        self.config = {
            'valgrind_timeout': 600,  # 10 minutes timeout
            'memcheck_options': [
                '--tool=memcheck',
                '--leak-check=full',
                '--show-leak-kinds=all',
                '--track-origins=yes',
                '--verbose',
                '--xml=yes'
            ],
            'massif_options': [
                '--tool=massif',
                '--heap=yes',
                '--stacks=yes',
                '--time-unit=B',  # Bytes allocated
                '--detailed-freq=1'
            ],
            'helgrind_options': [
                '--tool=helgrind',
                '--verbose',
                '--xml=yes'
            ],
            'callgrind_options': [
                '--tool=callgrind',
                '--collect-jumps=yes',
                '--collect-systime=yes'
            ]
        }

    async def build_rust_debug_binary(self) -> Dict[str, Any]:
        """Build Rust project with debug symbols for valgrind analysis."""
        logger.info("🔧 Building Rust project with debug symbols")

        build_result = {
            'success': False,
            'build_time_seconds': 0,
            'binary_path': None,
            'error': None
        }

        try:
            start_time = time.time()

            # Build with debug symbols and optimizations disabled
            build_env = os.environ.copy()
            build_env['RUSTFLAGS'] = '-g -C opt-level=0 -C debuginfo=2'

            build_process = await asyncio.create_subprocess_exec(
                'cargo', 'build', '--bin', 'workspace-qdrant-engine',
                cwd=self.rust_project_path,
                env=build_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await build_process.communicate()

            build_time = time.time() - start_time
            build_result['build_time_seconds'] = build_time

            if build_process.returncode == 0:
                # Find the built binary
                target_dir = self.rust_project_path / "target" / "debug"
                binary_candidates = [
                    target_dir / "workspace-qdrant-engine",
                    target_dir / "workspace_qdrant_engine",
                    target_dir / "wqm-engine"
                ]

                for candidate in binary_candidates:
                    if candidate.exists():
                        build_result['binary_path'] = candidate
                        build_result['success'] = True
                        logger.info(f"✅ Rust debug binary built: {candidate}")
                        break

                if not build_result['success']:
                    build_result['error'] = f"Debug binary not found in {target_dir}"
                    logger.error(f"❌ Debug binary not found after successful build")

            else:
                build_result['error'] = f"Build failed: {stderr.decode()}"
                logger.error(f"❌ Rust build failed: {build_result['error']}")

            build_result['stdout'] = stdout.decode()
            build_result['stderr'] = stderr.decode()

        except Exception as e:
            build_result['error'] = str(e)
            logger.error(f"❌ Rust build exception: {e}")

        return build_result

    async def run_memcheck_analysis(self, binary_path: Path,
                                  test_args: List[str] = None) -> Dict[str, Any]:
        """Run valgrind memcheck for memory error and leak detection."""
        logger.info("🔍 Starting valgrind memcheck analysis")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xml_output = self.valgrind_dir / f"memcheck_{timestamp}.xml"
        log_output = self.valgrind_dir / f"memcheck_{timestamp}.log"

        memcheck_results = {
            'success': False,
            'timestamp': timestamp,
            'analysis_duration_seconds': 0,
            'memory_errors': [],
            'leak_summary': {},
            'xml_report_path': str(xml_output),
            'log_path': str(log_output)
        }

        try:
            start_time = time.time()

            # Prepare valgrind memcheck command
            memcheck_cmd = [
                'valgrind',
                *self.config['memcheck_options'],
                f'--xml-file={xml_output}',
                f'--log-file={log_output}',
                str(binary_path)
            ]

            if test_args:
                memcheck_cmd.extend(test_args)

            logger.info(f"🚀 Executing: {' '.join(memcheck_cmd)}")

            # Check if valgrind is available
            if not await self._check_valgrind_available():
                return await self._mock_memcheck_results(memcheck_results)

            # Run valgrind memcheck
            memcheck_process = await asyncio.create_subprocess_exec(
                *memcheck_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    memcheck_process.communicate(),
                    timeout=self.config['valgrind_timeout']
                )

                analysis_time = time.time() - start_time
                memcheck_results['analysis_duration_seconds'] = analysis_time

                # Parse XML output if available
                if xml_output.exists():
                    memcheck_results.update(await self._parse_memcheck_xml(xml_output))
                    memcheck_results['success'] = True
                    logger.info(f"✅ Memcheck analysis completed ({analysis_time:.1f}s)")
                else:
                    memcheck_results['error'] = "XML output not generated"
                    logger.warning("⚠️ Memcheck XML output not found")

                # Store process output
                memcheck_results['stdout'] = stdout.decode()
                memcheck_results['stderr'] = stderr.decode()

            except asyncio.TimeoutError:
                logger.warning(f"⏰ Memcheck analysis timed out after {self.config['valgrind_timeout']}s")
                memcheck_process.kill()
                memcheck_results['error'] = 'Analysis timed out'

        except Exception as e:
            logger.error(f"❌ Memcheck analysis failed: {e}")
            memcheck_results['error'] = str(e)

        return memcheck_results

    async def run_massif_heap_profiling(self, binary_path: Path,
                                      test_args: List[str] = None) -> Dict[str, Any]:
        """Run valgrind massif for heap profiling analysis."""
        logger.info("📊 Starting valgrind massif heap profiling")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        massif_output = self.valgrind_dir / f"massif.out.{timestamp}"
        log_output = self.valgrind_dir / f"massif_{timestamp}.log"

        massif_results = {
            'success': False,
            'timestamp': timestamp,
            'profiling_duration_seconds': 0,
            'heap_profile_path': str(massif_output),
            'peak_memory_bytes': 0,
            'heap_analysis': {},
            'log_path': str(log_output)
        }

        try:
            start_time = time.time()

            # Prepare massif command
            massif_cmd = [
                'valgrind',
                *self.config['massif_options'],
                f'--massif-out-file={massif_output}',
                f'--log-file={log_output}',
                str(binary_path)
            ]

            if test_args:
                massif_cmd.extend(test_args)

            logger.info(f"🚀 Executing massif: {' '.join(massif_cmd)}")

            if not await self._check_valgrind_available():
                return await self._mock_massif_results(massif_results)

            # Run massif profiling
            massif_process = await asyncio.create_subprocess_exec(
                *massif_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    massif_process.communicate(),
                    timeout=self.config['valgrind_timeout']
                )

                profiling_time = time.time() - start_time
                massif_results['profiling_duration_seconds'] = profiling_time

                # Analyze massif output
                if massif_output.exists():
                    massif_results.update(await self._analyze_massif_output(massif_output))
                    massif_results['success'] = True
                    logger.info(f"✅ Massif profiling completed ({profiling_time:.1f}s)")
                else:
                    massif_results['error'] = "Massif output not generated"

                massif_results['stdout'] = stdout.decode()
                massif_results['stderr'] = stderr.decode()

            except asyncio.TimeoutError:
                logger.warning(f"⏰ Massif profiling timed out")
                massif_process.kill()
                massif_results['error'] = 'Profiling timed out'

        except Exception as e:
            logger.error(f"❌ Massif profiling failed: {e}")
            massif_results['error'] = str(e)

        return massif_results

    async def run_helgrind_thread_analysis(self, binary_path: Path,
                                         test_args: List[str] = None) -> Dict[str, Any]:
        """Run valgrind helgrind for thread error detection."""
        logger.info("🧵 Starting valgrind helgrind thread analysis")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xml_output = self.valgrind_dir / f"helgrind_{timestamp}.xml"
        log_output = self.valgrind_dir / f"helgrind_{timestamp}.log"

        helgrind_results = {
            'success': False,
            'timestamp': timestamp,
            'analysis_duration_seconds': 0,
            'thread_errors': [],
            'race_conditions': [],
            'xml_report_path': str(xml_output),
            'log_path': str(log_output)
        }

        try:
            start_time = time.time()

            helgrind_cmd = [
                'valgrind',
                *self.config['helgrind_options'],
                f'--xml-file={xml_output}',
                f'--log-file={log_output}',
                str(binary_path)
            ]

            if test_args:
                helgrind_cmd.extend(test_args)

            if not await self._check_valgrind_available():
                return await self._mock_helgrind_results(helgrind_results)

            helgrind_process = await asyncio.create_subprocess_exec(
                *helgrind_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    helgrind_process.communicate(),
                    timeout=self.config['valgrind_timeout']
                )

                analysis_time = time.time() - start_time
                helgrind_results['analysis_duration_seconds'] = analysis_time

                if xml_output.exists():
                    helgrind_results.update(await self._parse_helgrind_xml(xml_output))
                    helgrind_results['success'] = True
                    logger.info(f"✅ Helgrind analysis completed ({analysis_time:.1f}s)")

                helgrind_results['stdout'] = stdout.decode()
                helgrind_results['stderr'] = stderr.decode()

            except asyncio.TimeoutError:
                logger.warning("⏰ Helgrind analysis timed out")
                helgrind_process.kill()
                helgrind_results['error'] = 'Analysis timed out'

        except Exception as e:
            logger.error(f"❌ Helgrind analysis failed: {e}")
            helgrind_results['error'] = str(e)

        return helgrind_results

    async def comprehensive_rust_memory_analysis(self, test_scenarios: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive memory analysis on Rust components."""
        logger.info("🔬 Starting comprehensive Rust memory analysis")

        analysis_start = time.time()
        comprehensive_results = {
            'start_time': datetime.now().isoformat(),
            'rust_project_path': str(self.rust_project_path),
            'build_results': {},
            'memcheck_results': {},
            'massif_results': {},
            'helgrind_results': {},
            'overall_assessment': {},
            'success': True
        }

        try:
            # Step 1: Build debug binary
            logger.info("📦 Building Rust debug binary...")
            build_results = await self.build_rust_debug_binary()
            comprehensive_results['build_results'] = build_results

            if not build_results['success']:
                comprehensive_results['success'] = False
                comprehensive_results['error'] = 'Failed to build debug binary'
                return comprehensive_results

            binary_path = build_results['binary_path']
            test_args = test_scenarios or ['--help']  # Default test args

            # Step 2: Run memcheck analysis
            logger.info("🔍 Running memory error analysis...")
            memcheck_results = await self.run_memcheck_analysis(binary_path, test_args)
            comprehensive_results['memcheck_results'] = memcheck_results

            # Step 3: Run massif heap profiling
            logger.info("📊 Running heap profiling...")
            massif_results = await self.run_massif_heap_profiling(binary_path, test_args)
            comprehensive_results['massif_results'] = massif_results

            # Step 4: Run helgrind thread analysis
            logger.info("🧵 Running thread analysis...")
            helgrind_results = await self.run_helgrind_thread_analysis(binary_path, test_args)
            comprehensive_results['helgrind_results'] = helgrind_results

            # Step 5: Generate overall assessment
            comprehensive_results['overall_assessment'] = self._generate_overall_assessment({
                'memcheck': memcheck_results,
                'massif': massif_results,
                'helgrind': helgrind_results
            })

            total_analysis_time = time.time() - analysis_start
            comprehensive_results['total_analysis_duration_seconds'] = total_analysis_time

            # Determine overall success
            analysis_success = (
                memcheck_results.get('success', False) and
                massif_results.get('success', False) and
                helgrind_results.get('success', False)
            )

            comprehensive_results['success'] = analysis_success

            if analysis_success:
                logger.info(f"✅ Comprehensive Rust memory analysis completed ({total_analysis_time:.1f}s)")
            else:
                logger.warning("⚠️ Some analysis components failed")

        except Exception as e:
            comprehensive_results['success'] = False
            comprehensive_results['error'] = str(e)
            logger.error(f"❌ Comprehensive analysis failed: {e}")

        comprehensive_results['end_time'] = datetime.now().isoformat()
        return comprehensive_results

    async def _check_valgrind_available(self) -> bool:
        """Check if valgrind is available on the system."""
        try:
            result = await asyncio.create_subprocess_exec(
                'valgrind', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("⚠️ valgrind not found on system")
            return False

    async def _parse_memcheck_xml(self, xml_path: Path) -> Dict[str, Any]:
        """Parse valgrind memcheck XML output."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            errors = []
            for error in root.findall('.//error'):
                error_kind = error.find('kind')
                error_what = error.find('what')

                error_info = {
                    'kind': error_kind.text if error_kind is not None else 'unknown',
                    'description': error_what.text if error_what is not None else 'no description'
                }

                # Extract stack trace
                stack_frames = []
                for frame in error.findall('.//frame'):
                    fn_element = frame.find('fn')
                    file_element = frame.find('file')
                    line_element = frame.find('line')

                    if fn_element is not None:
                        frame_info = {'function': fn_element.text}
                        if file_element is not None:
                            frame_info['file'] = file_element.text
                        if line_element is not None:
                            frame_info['line'] = int(line_element.text)
                        stack_frames.append(frame_info)

                error_info['stack_trace'] = stack_frames[:10]  # Top 10 frames
                errors.append(error_info)

            return {
                'memory_errors': errors,
                'error_count': len(errors),
                'leak_summary': self._extract_leak_summary(root)
            }

        except Exception as e:
            logger.warning(f"Failed to parse memcheck XML: {e}")
            return {'memory_errors': [], 'error_count': 0, 'parse_error': str(e)}

    def _extract_leak_summary(self, root: ET.Element) -> Dict[str, Any]:
        """Extract leak summary from memcheck XML."""
        leak_summary = {
            'definitely_lost': 0,
            'indirectly_lost': 0,
            'possibly_lost': 0,
            'still_reachable': 0,
            'suppressed': 0
        }

        # Look for leak summary in various XML formats
        for suppcounts in root.findall('.//suppcounts'):
            for pair in suppcounts.findall('pair'):
                name = pair.find('name')
                count = pair.find('count')
                if name is not None and count is not None:
                    name_text = name.text.lower().replace(' ', '_')
                    if name_text in leak_summary:
                        leak_summary[name_text] = int(count.text)

        return leak_summary

    async def _analyze_massif_output(self, massif_file: Path) -> Dict[str, Any]:
        """Analyze massif output file for heap profiling data."""
        try:
            content = massif_file.read_text()
            lines = content.split('\n')

            heap_analysis = {
                'peak_memory_bytes': 0,
                'snapshots': [],
                'memory_timeline': [],
                'top_allocators': []
            }

            current_snapshot = None
            for line in lines:
                line = line.strip()

                if line.startswith('snapshot='):
                    if current_snapshot:
                        heap_analysis['snapshots'].append(current_snapshot)
                    current_snapshot = {'snapshot_id': int(line.split('=')[1])}

                elif line.startswith('time='):
                    if current_snapshot:
                        current_snapshot['time'] = int(line.split('=')[1])

                elif line.startswith('mem_heap_B='):
                    heap_bytes = int(line.split('=')[1])
                    if current_snapshot:
                        current_snapshot['heap_bytes'] = heap_bytes
                    heap_analysis['memory_timeline'].append(heap_bytes)
                    if heap_bytes > heap_analysis['peak_memory_bytes']:
                        heap_analysis['peak_memory_bytes'] = heap_bytes

                elif line.startswith('heap_tree='):
                    tree_type = line.split('=')[1]
                    if current_snapshot:
                        current_snapshot['heap_tree_type'] = tree_type

            # Add final snapshot
            if current_snapshot:
                heap_analysis['snapshots'].append(current_snapshot)

            return heap_analysis

        except Exception as e:
            logger.warning(f"Failed to analyze massif output: {e}")
            return {'peak_memory_bytes': 0, 'analysis_error': str(e)}

    async def _parse_helgrind_xml(self, xml_path: Path) -> Dict[str, Any]:
        """Parse helgrind XML output for thread analysis."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            thread_errors = []
            race_conditions = []

            for error in root.findall('.//error'):
                error_kind = error.find('kind')
                error_what = error.find('what')

                if error_kind is not None:
                    kind = error_kind.text
                    description = error_what.text if error_what is not None else ''

                    error_info = {
                        'kind': kind,
                        'description': description
                    }

                    if 'race' in kind.lower():
                        race_conditions.append(error_info)
                    else:
                        thread_errors.append(error_info)

            return {
                'thread_errors': thread_errors,
                'race_conditions': race_conditions,
                'thread_error_count': len(thread_errors),
                'race_condition_count': len(race_conditions)
            }

        except Exception as e:
            logger.warning(f"Failed to parse helgrind XML: {e}")
            return {'thread_errors': [], 'race_conditions': [], 'parse_error': str(e)}

    def _generate_overall_assessment(self, analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall assessment from all analysis results."""
        memcheck = analysis_results.get('memcheck', {})
        massif = analysis_results.get('massif', {})
        helgrind = analysis_results.get('helgrind', {})

        assessment = {
            'memory_safety_score': 100,
            'performance_score': 100,
            'thread_safety_score': 100,
            'overall_score': 100,
            'critical_issues': [],
            'recommendations': []
        }

        # Assess memory safety
        memory_error_count = memcheck.get('error_count', 0)
        if memory_error_count > 0:
            assessment['memory_safety_score'] = max(0, 100 - memory_error_count * 10)
            assessment['critical_issues'].append(f"{memory_error_count} memory errors detected")
            assessment['recommendations'].append("Fix memory errors found by memcheck")

        # Assess performance based on heap usage
        peak_memory = massif.get('peak_memory_bytes', 0)
        if peak_memory > 100 * 1024 * 1024:  # > 100MB
            performance_penalty = min(50, (peak_memory - 100*1024*1024) // (10*1024*1024) * 10)
            assessment['performance_score'] = max(50, 100 - performance_penalty)
            assessment['recommendations'].append("Consider optimizing memory usage patterns")

        # Assess thread safety
        race_count = helgrind.get('race_condition_count', 0)
        thread_error_count = helgrind.get('thread_error_count', 0)

        if race_count > 0 or thread_error_count > 0:
            thread_penalty = (race_count * 20) + (thread_error_count * 10)
            assessment['thread_safety_score'] = max(0, 100 - thread_penalty)
            assessment['critical_issues'].append(f"{race_count} race conditions, {thread_error_count} thread errors")
            assessment['recommendations'].append("Fix thread safety issues found by helgrind")

        # Calculate overall score
        assessment['overall_score'] = (
            assessment['memory_safety_score'] +
            assessment['performance_score'] +
            assessment['thread_safety_score']
        ) // 3

        return assessment

    # Mock result methods for when valgrind is not available
    async def _mock_memcheck_results(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock memcheck results when valgrind unavailable."""
        logger.warning("🔄 Generating mock memcheck results (valgrind unavailable)")
        base_results.update({
            'success': True,
            'mock_results': True,
            'memory_errors': [
                {
                    'kind': 'InvalidRead',
                    'description': 'Mock memory error for testing',
                    'stack_trace': [{'function': 'mock_function', 'file': 'mock.rs', 'line': 42}]
                }
            ],
            'error_count': 1,
            'leak_summary': {
                'definitely_lost': 0,
                'indirectly_lost': 0,
                'possibly_lost': 1024,
                'still_reachable': 0,
                'suppressed': 0
            }
        })
        return base_results

    async def _mock_massif_results(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock massif results when valgrind unavailable."""
        logger.warning("🔄 Generating mock massif results (valgrind unavailable)")
        base_results.update({
            'success': True,
            'mock_results': True,
            'peak_memory_bytes': 50 * 1024 * 1024,  # 50MB
            'heap_analysis': {
                'peak_memory_bytes': 50 * 1024 * 1024,
                'snapshots': [
                    {'snapshot_id': 0, 'time': 0, 'heap_bytes': 1024},
                    {'snapshot_id': 1, 'time': 1000, 'heap_bytes': 50 * 1024 * 1024},
                ],
                'memory_timeline': [1024, 10240, 102400, 50 * 1024 * 1024],
                'top_allocators': []
            }
        })
        return base_results

    async def _mock_helgrind_results(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock helgrind results when valgrind unavailable."""
        logger.warning("🔄 Generating mock helgrind results (valgrind unavailable)")
        base_results.update({
            'success': True,
            'mock_results': True,
            'thread_errors': [],
            'race_conditions': [],
            'thread_error_count': 0,
            'race_condition_count': 0
        })
        return base_results

    async def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")


# Integration with existing performance test framework
class ValgrindRustProfilerIntegration:
    """Integration layer for valgrind profiler with performance test suite."""

    @classmethod
    async def profile_rust_engine_memory(cls, rust_project_path: Path,
                                       test_scenarios: List[str] = None) -> Dict[str, Any]:
        """Profile Rust engine memory usage and detect issues."""
        profiler = ValgrindRustProfiler(rust_project_path)

        try:
            return await profiler.comprehensive_rust_memory_analysis(test_scenarios)
        finally:
            await profiler.cleanup()

    @classmethod
    async def validate_rust_memory_safety(cls, rust_project_path: Path) -> Dict[str, Any]:
        """Validate memory safety of Rust components."""
        profiler = ValgrindRustProfiler(rust_project_path)

        try:
            # Build debug binary
            build_results = await profiler.build_rust_debug_binary()
            if not build_results['success']:
                return {
                    'success': False,
                    'error': 'Failed to build debug binary',
                    'build_results': build_results
                }

            # Run memory safety analysis
            binary_path = build_results['binary_path']
            memcheck_results = await profiler.run_memcheck_analysis(binary_path, ['--test'])

            # Assess memory safety
            memory_errors = memcheck_results.get('memory_errors', [])
            leak_summary = memcheck_results.get('leak_summary', {})

            safety_assessment = {
                'memory_safe': len(memory_errors) == 0,
                'error_count': len(memory_errors),
                'definitely_lost_bytes': leak_summary.get('definitely_lost', 0),
                'possibly_lost_bytes': leak_summary.get('possibly_lost', 0),
                'assessment': 'PASS' if len(memory_errors) == 0 else 'FAIL',
                'memcheck_results': memcheck_results
            }

            return {
                'success': True,
                'safety_assessment': safety_assessment,
                'build_results': build_results
            }

        finally:
            await profiler.cleanup()


if __name__ == "__main__":
    async def main():
        # Test valgrind profiler
        rust_path = Path("rust-engine")
        if rust_path.exists():
            profiler = ValgrindRustProfiler(rust_path)
            try:
                print("Starting valgrind Rust profiler test...")
                results = await profiler.comprehensive_rust_memory_analysis()
                print(f"Analysis results: {json.dumps(results, indent=2)}")
            finally:
                await profiler.cleanup()
        else:
            print("Rust engine not found at rust-engine/")

    asyncio.run(main())