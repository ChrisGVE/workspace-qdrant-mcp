"""
File ingestion throughput benchmarks.

Measures performance of file parsing and ingestion across various file sizes
and types. Tests both single-file and batch ingestion scenarios.

Run with: uv run pytest tests/benchmarks/benchmark_file_ingestion.py --benchmark-only
"""

import asyncio
import tempfile
from pathlib import Path

import pytest
from wqm_cli.cli.parsers import CodeParser, MarkdownParser, TextParser


class TestDataGenerator:
    """Generates test files for benchmarking."""

    @staticmethod
    def generate_text_content(size_kb: int) -> str:
        """Generate text content of specified size in KB."""
        # Generate realistic text with variety
        lines = []
        target_bytes = size_kb * 1024
        current_bytes = 0

        line_templates = [
            "This is a sample line of text for benchmarking purposes. ",
            "Performance testing requires realistic data of varying lengths. ",
            "File ingestion throughput depends on file size and type. ",
            "The workspace-qdrant-mcp system handles multiple formats. ",
            "Benchmark results help identify performance bottlenecks. ",
        ]

        line_idx = 0
        while current_bytes < target_bytes:
            line = line_templates[line_idx % len(line_templates)]
            lines.append(line + "\n")
            current_bytes += len(line.encode("utf-8"))
            line_idx += 1

        return "".join(lines)

    @staticmethod
    def generate_python_content(size_kb: int) -> str:
        """Generate Python code content of specified size in KB."""
        lines = []
        target_bytes = size_kb * 1024
        current_bytes = 0

        code_templates = [
            "def function_{idx}(param1: str, param2: int) -> bool:\n",
            '    """Docstring for function_{idx}."""\n',
            "    result = param1 + str(param2)\n",
            "    return len(result) > 0\n",
            "\n",
            "class Class_{idx}:\n",
            '    """Class docstring."""\n',
            "    def __init__(self, value: int):\n",
            "        self.value = value\n",
            "\n",
        ]

        idx = 0
        while current_bytes < target_bytes:
            for template in code_templates:
                line = template.format(idx=idx)
                lines.append(line)
                current_bytes += len(line.encode("utf-8"))
                if current_bytes >= target_bytes:
                    break
            idx += 1

        return "".join(lines)

    @staticmethod
    def generate_markdown_content(size_kb: int) -> str:
        """Generate Markdown content of specified size in KB."""
        lines = []
        target_bytes = size_kb * 1024
        current_bytes = 0

        md_templates = [
            "# Heading Level 1\n\n",
            "## Heading Level 2\n\n",
            "This is a paragraph with **bold text** and *italic text*. ",
            "It contains [links](https://example.com) and `inline code`. \n\n",
            "- List item 1\n",
            "- List item 2\n",
            "- List item 3\n\n",
            "```python\n",
            "def example():\n",
            "    return True\n",
            "```\n\n",
        ]

        idx = 0
        while current_bytes < target_bytes:
            line = md_templates[idx % len(md_templates)]
            lines.append(line)
            current_bytes += len(line.encode("utf-8"))
            idx += 1

        return "".join(lines)

    @staticmethod
    def generate_json_content(size_kb: int) -> str:
        """Generate JSON content of specified size in KB."""
        import json

        target_bytes = size_kb * 1024
        data = {"items": []}

        idx = 0
        while len(json.dumps(data).encode("utf-8")) < target_bytes:
            data["items"].append(
                {
                    "id": idx,
                    "name": f"Item {idx}",
                    "description": f"This is item number {idx} for benchmarking",
                    "value": idx * 1.5,
                    "active": idx % 2 == 0,
                }
            )
            idx += 1

        return json.dumps(data, indent=2)

    @staticmethod
    def create_test_file(content: str, extension: str, tmp_dir: Path) -> Path:
        """Create a test file with the given content and extension."""
        file_path = tmp_dir / f"test_file{extension}"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    @staticmethod
    def create_test_files_batch(
        num_files: int, size_kb: int, extension: str, tmp_dir: Path
    ) -> list[Path]:
        """Create a batch of test files."""
        generator_map = {
            ".txt": TestDataGenerator.generate_text_content,
            ".py": TestDataGenerator.generate_python_content,
            ".md": TestDataGenerator.generate_markdown_content,
            ".json": TestDataGenerator.generate_json_content,
        }

        generator = generator_map.get(extension, TestDataGenerator.generate_text_content)
        content = generator(size_kb)

        files = []
        for i in range(num_files):
            file_path = tmp_dir / f"test_file_{i}{extension}"
            file_path.write_text(content, encoding="utf-8")
            files.append(file_path)

        return files


# Fixtures for test data
@pytest.fixture(scope="module")
def tmp_benchmark_dir():
    """Create a temporary directory for benchmark test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="module")
def test_files_small(tmp_benchmark_dir):
    """Generate small test files (1KB)."""
    return {
        "txt": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_text_content(1), ".txt", tmp_benchmark_dir
        ),
        "py": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_python_content(1), ".py", tmp_benchmark_dir
        ),
        "md": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_markdown_content(1), ".md", tmp_benchmark_dir
        ),
        "json": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_json_content(1), ".json", tmp_benchmark_dir
        ),
    }


@pytest.fixture(scope="module")
def test_files_medium(tmp_benchmark_dir):
    """Generate medium test files (100KB)."""
    return {
        "txt": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_text_content(100), ".txt", tmp_benchmark_dir
        ),
        "py": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_python_content(100), ".py", tmp_benchmark_dir
        ),
        "md": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_markdown_content(100), ".md", tmp_benchmark_dir
        ),
        "json": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_json_content(100), ".json", tmp_benchmark_dir
        ),
    }


@pytest.fixture(scope="module")
def test_files_large(tmp_benchmark_dir):
    """Generate large test files (1MB)."""
    return {
        "txt": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_text_content(1024), ".txt", tmp_benchmark_dir
        ),
        "py": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_python_content(1024), ".py", tmp_benchmark_dir
        ),
        "md": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_markdown_content(1024), ".md", tmp_benchmark_dir
        ),
        "json": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_json_content(1024), ".json", tmp_benchmark_dir
        ),
    }


@pytest.fixture(scope="module")
def test_files_very_large(tmp_benchmark_dir):
    """Generate very large test files (10MB)."""
    return {
        "txt": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_text_content(10 * 1024), ".txt", tmp_benchmark_dir
        ),
        "py": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_python_content(10 * 1024), ".py", tmp_benchmark_dir
        ),
        "md": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_markdown_content(10 * 1024),
            ".md",
            tmp_benchmark_dir,
        ),
        "json": TestDataGenerator.create_test_file(
            TestDataGenerator.generate_json_content(10 * 1024), ".json", tmp_benchmark_dir
        ),
    }


# Single file parsing benchmarks
@pytest.mark.benchmark
def test_parse_small_text_file(benchmark, test_files_small):
    """Benchmark parsing small (1KB) text file."""
    parser = TextParser()
    file_path = test_files_small["txt"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_medium_text_file(benchmark, test_files_medium):
    """Benchmark parsing medium (100KB) text file."""
    parser = TextParser()
    file_path = test_files_medium["txt"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_large_text_file(benchmark, test_files_large):
    """Benchmark parsing large (1MB) text file."""
    parser = TextParser()
    file_path = test_files_large["txt"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_very_large_text_file(benchmark, test_files_very_large):
    """Benchmark parsing very large (10MB) text file."""
    parser = TextParser()
    file_path = test_files_very_large["txt"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_small_python_file(benchmark, test_files_small):
    """Benchmark parsing small (1KB) Python file."""
    parser = CodeParser()
    file_path = test_files_small["py"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_medium_python_file(benchmark, test_files_medium):
    """Benchmark parsing medium (100KB) Python file."""
    parser = CodeParser()
    file_path = test_files_medium["py"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_large_python_file(benchmark, test_files_large):
    """Benchmark parsing large (1MB) Python file."""
    parser = CodeParser()
    file_path = test_files_large["py"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_small_markdown_file(benchmark, test_files_small):
    """Benchmark parsing small (1KB) Markdown file."""
    parser = MarkdownParser()
    file_path = test_files_small["md"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_medium_markdown_file(benchmark, test_files_medium):
    """Benchmark parsing medium (100KB) Markdown file."""
    parser = MarkdownParser()
    file_path = test_files_medium["md"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_large_markdown_file(benchmark, test_files_large):
    """Benchmark parsing large (1MB) Markdown file."""
    parser = MarkdownParser()
    file_path = test_files_large["md"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_small_json_file(benchmark, test_files_small):
    """Benchmark parsing small (1KB) JSON file."""
    parser = TextParser()
    file_path = test_files_small["json"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_medium_json_file(benchmark, test_files_medium):
    """Benchmark parsing medium (100KB) JSON file."""
    parser = TextParser()
    file_path = test_files_medium["json"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


@pytest.mark.benchmark
def test_parse_large_json_file(benchmark, test_files_large):
    """Benchmark parsing large (1MB) JSON file."""
    parser = TextParser()
    file_path = test_files_large["json"]

    def run_parse():
        return asyncio.run(parser.parse(file_path))

    result = benchmark(run_parse)
    assert result.content_hash


# Batch parsing benchmarks
@pytest.mark.benchmark
def test_parse_batch_10_small_files(benchmark, tmp_benchmark_dir):
    """Benchmark parsing batch of 10 small (1KB) files."""
    files = TestDataGenerator.create_test_files_batch(10, 1, ".txt", tmp_benchmark_dir)
    parser = TextParser()

    def run_parse_batch():
        async def parse_batch():
            results = []
            for file_path in files:
                result = await parser.parse(file_path)
                results.append(result)
            return results

        return asyncio.run(parse_batch())

    results = benchmark(run_parse_batch)
    assert len(results) == 10


@pytest.mark.benchmark
def test_parse_batch_10_medium_files(benchmark, tmp_benchmark_dir):
    """Benchmark parsing batch of 10 medium (100KB) files."""
    files = TestDataGenerator.create_test_files_batch(10, 100, ".txt", tmp_benchmark_dir)
    parser = TextParser()

    def run_parse_batch():
        async def parse_batch():
            results = []
            for file_path in files:
                result = await parser.parse(file_path)
                results.append(result)
            return results

        return asyncio.run(parse_batch())

    results = benchmark(run_parse_batch)
    assert len(results) == 10


@pytest.mark.benchmark
def test_parse_batch_50_small_files(benchmark, tmp_benchmark_dir):
    """Benchmark parsing batch of 50 small (1KB) files."""
    files = TestDataGenerator.create_test_files_batch(50, 1, ".txt", tmp_benchmark_dir)
    parser = TextParser()

    def run_parse_batch():
        async def parse_batch():
            results = []
            for file_path in files:
                result = await parser.parse(file_path)
                results.append(result)
            return results

        return asyncio.run(parse_batch())

    results = benchmark(run_parse_batch)
    assert len(results) == 50


@pytest.mark.benchmark
def test_parse_batch_mixed_types(benchmark, tmp_benchmark_dir):
    """Benchmark parsing batch of mixed file types."""
    files = []
    files.extend(TestDataGenerator.create_test_files_batch(5, 10, ".txt", tmp_benchmark_dir))
    files.extend(TestDataGenerator.create_test_files_batch(5, 10, ".py", tmp_benchmark_dir))
    files.extend(TestDataGenerator.create_test_files_batch(5, 10, ".md", tmp_benchmark_dir))
    files.extend(TestDataGenerator.create_test_files_batch(5, 10, ".json", tmp_benchmark_dir))

    def run_parse_batch():
        async def parse_batch():
            results = []
            for file_path in files:
                # Select appropriate parser based on extension
                if file_path.suffix == ".py":
                    parser = CodeParser()
                elif file_path.suffix == ".md":
                    parser = MarkdownParser()
                else:
                    parser = TextParser()

                result = await parser.parse(file_path)
                results.append(result)
            return results

        return asyncio.run(parse_batch())

    results = benchmark(run_parse_batch)
    assert len(results) == 20


# Throughput measurement benchmarks
@pytest.mark.benchmark
def test_throughput_small_files(benchmark, tmp_benchmark_dir):
    """Measure throughput for small files (files/second)."""
    files = TestDataGenerator.create_test_files_batch(100, 1, ".txt", tmp_benchmark_dir)
    parser = TextParser()

    def run_parse_all():
        async def parse_all():
            results = []
            for file_path in files:
                result = await parser.parse(file_path)
                results.append(result)
            return results

        return asyncio.run(parse_all())

    results = benchmark(run_parse_all)
    assert len(results) == 100
    # Benchmark will automatically calculate ops/sec (files/sec)


@pytest.mark.benchmark
def test_throughput_medium_files(benchmark, tmp_benchmark_dir):
    """Measure throughput for medium files (files/second and MB/second)."""
    files = TestDataGenerator.create_test_files_batch(20, 100, ".txt", tmp_benchmark_dir)
    parser = TextParser()

    def run_parse_all():
        async def parse_all():
            results = []
            for file_path in files:
                result = await parser.parse(file_path)
                results.append(result)
            return results

        return asyncio.run(parse_all())

    results = benchmark(run_parse_all)
    assert len(results) == 20
    # Total data processed: 20 files * 100KB = 2MB
