"""Minimal stub implementation of the testcontainers Qdrant container.

The real project depends on the third-party ``testcontainers`` package to spin up
Qdrant instances for integration testing. For fast unit test runs we only need a
very small subset of that API so the unit tests can import the helpers in
``tests.utils.testcontainers_qdrant`` without requiring the external dependency.
"""

from __future__ import annotations

from typing import Iterable


class QdrantContainer:
    """Stub container with the minimal API used by the tests."""

    def __init__(self, image: str | None = None) -> None:
        self.image = image
        self._ports: list[int] = []
        self._started = False

    # The real implementation returns ``self`` for fluent chaining – keep that behaviour.
    def with_exposed_ports(self, *ports: Iterable[int]) -> "QdrantContainer":
        self._ports.extend(int(port) for port in ports)
        return self

    def start(self) -> "QdrantContainer":
        self._started = True
        return self

    def stop(self) -> None:
        self._started = False

    # Helper accessors used to compose URLs in the fixtures -----------------
    def get_container_host_ip(self) -> str:
        return "127.0.0.1"

    def get_exposed_port(self, port: int | str) -> str:
        return str(port)

    # Context manager support (used implicitly by some helpers) -------------
    def __enter__(self) -> "QdrantContainer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

