"""Playwright configuration for workspace web UI testing."""

from playwright.sync_api import Playwright


def pytest_configure(config):
    """Configure pytest for Playwright."""
    config.addinivalue_line("markers", "playwright: mark test as a playwright test")


# Playwright configuration
def configure_playwright(p: Playwright):
    """Configure Playwright browsers and settings."""
    return {
        "browsers": ["chromium"],  # Use Chromium for consistent testing
        "headless": True,
        "base_url": "http://localhost:3000",
        "timeout": 30000,  # 30 second timeout
        "video": "retain-on-failure",
        "screenshot": "only-on-failure",
    }


# Browser launch arguments for CI/CD environments
BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage", 
    "--disable-extensions",
    "--disable-plugins",
    "--disable-images",
    "--disable-javascript-harmony-shipping",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]


# Test configuration constants
TEST_CONFIG = {
    "base_url": "http://localhost:3000",
    "dev_server_port": 3000,
    "mock_daemon_port": 8899,
    "timeout": 30000,
    "slow_timeout": 60000,
    "expect_timeout": 10000,
}