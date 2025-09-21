"""
Playwright configuration for workspace-qdrant-mcp functional testing.

This configuration sets up web UI testing for MCP server interfaces,
admin dashboards, and any web-based components.
"""

from pathlib import Path
from typing import Dict, Any
import os

# Base configuration
CONFIG: Dict[str, Any] = {
    # Test directory configuration
    "testDir": Path(__file__).parent,
    "outputDir": Path(__file__).parent / "test-results",
    "testIgnore": ["**/node_modules/**"],

    # Global test settings
    "timeout": 30000,  # 30 seconds per test
    "expect": {
        "timeout": 5000,  # 5 seconds for assertions
    },

    # Test execution settings
    "fullyParallel": True,
    "forbidOnly": bool(os.getenv("CI")),  # Fail if test.only in CI
    "retries": 2 if os.getenv("CI") else 0,
    "workers": 4 if os.getenv("CI") else 2,

    # Reporter configuration
    "reporter": [
        ["list"],
        ["json", {"outputFile": "test-results/results.json"}],
        ["html", {"outputFolder": "test-results/html-report", "open": "never"}],
        ["junit", {"outputFile": "test-results/junit.xml"}],
    ],

    # Global test setup
    "use": {
        "baseURL": "http://localhost:8000",  # Default MCP server URL
        "trace": "on-first-retry",
        "screenshot": "only-on-failure",
        "video": "retain-on-failure",
        "actionTimeout": 10000,
        "navigationTimeout": 15000,
    },

    # Browser projects configuration
    "projects": [
        {
            "name": "chromium",
            "use": {
                "...devices": "Desktop Chrome",
                "viewport": {"width": 1280, "height": 720},
                "ignoreHTTPSErrors": True,
                "acceptDownloads": True,
            },
        },
        {
            "name": "firefox",
            "use": {
                "...devices": "Desktop Firefox",
                "viewport": {"width": 1280, "height": 720},
                "ignoreHTTPSErrors": True,
            },
        },
        {
            "name": "webkit",
            "use": {
                "...devices": "Desktop Safari",
                "viewport": {"width": 1280, "height": 720},
                "ignoreHTTPSErrors": True,
            },
        },
        # Mobile testing
        {
            "name": "Mobile Chrome",
            "use": {
                "...devices": "Pixel 5",
            },
        },
        {
            "name": "Mobile Safari",
            "use": {
                "...devices": "iPhone 12",
            },
        },
    ],

    # Web server configuration for testing
    "webServer": {
        "command": "uv run workspace-qdrant-mcp --transport http --host 127.0.0.1 --port 8000",
        "port": 8000,
        "reuseExistingServer": not bool(os.getenv("CI")),
        "timeout": 120000,  # 2 minutes to start server
    },
}

# Environment-specific configuration
if os.getenv("CI"):
    # CI optimizations
    CONFIG["workers"] = 1  # Single worker in CI for stability
    CONFIG["retries"] = 3
    CONFIG["use"]["video"] = "off"  # Disable video in CI
    CONFIG["projects"] = [
        # Only test Chromium in CI for speed
        {
            "name": "chromium-ci",
            "use": {
                "...devices": "Desktop Chrome",
                "viewport": {"width": 1280, "height": 720},
                "ignoreHTTPSErrors": True,
            },
        }
    ]

# Development configuration
if os.getenv("PLAYWRIGHT_DEBUG"):
    CONFIG["use"]["headless"] = False
    CONFIG["use"]["slowMo"] = 1000
    CONFIG["workers"] = 1
    CONFIG["timeout"] = 0  # Disable timeout in debug mode