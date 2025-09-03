# Web UI Functional Testing Guide

This document describes the comprehensive web UI functional testing implementation for Task 78, which validates the complete workspace web interface functionality using browser automation and integration testing.

## Overview

The web UI testing infrastructure includes:

1. **CLI Command Tests** - Validate `wqm web` commands
2. **Playwright Browser Tests** - Automate web interface testing
3. **Integration Tests** - Test CLI + UI workflows together
4. **Mock Services** - Isolated testing with mock daemon APIs

## Test Structure

```
tests/
├── integration/
│   ├── test_web_commands_integration.py    # CLI commands testing
│   └── test_web_ui_integration.py          # CLI + UI integration
├── playwright/
│   └── test_web_ui_functionality.py        # Browser automation tests
└── ...

scripts/
└── run_web_ui_tests.py                     # Comprehensive test runner

playwright.config.py                        # Playwright configuration
```

## Prerequisites

### System Requirements

- **Python 3.10+** with virtual environment
- **Node.js 16+** and npm
- **Git** with submodules initialized
- **Chrome/Chromium** (installed automatically by Playwright)

### Dependencies Installation

```bash
# Install Python dependencies (includes Playwright)
pip install -e .[dev]

# Install Playwright browsers
python -m playwright install chromium

# Initialize web-ui submodule if not already done  
git submodule update --init --recursive

# Install Node.js dependencies
cd web-ui && npm install
```

## Running Tests

### Using the Test Runner (Recommended)

```bash
# Run all tests with automatic setup
python scripts/run_web_ui_tests.py --all

# Run only CLI command tests
python scripts/run_web_ui_tests.py --cli-only

# Run only UI browser tests  
python scripts/run_web_ui_tests.py --ui-only

# Run integration tests
python scripts/run_web_ui_tests.py --integration

# Start dev server and run full UI tests
python scripts/run_web_ui_tests.py --dev-server

# Skip dependency installation
python scripts/run_web_ui_tests.py --no-install --all
```

### Using pytest Directly

```bash
# CLI command tests
pytest tests/integration/test_web_commands_integration.py -v

# Playwright browser tests (requires dev server)
pytest tests/playwright/test_web_ui_functionality.py -v

# Integration tests
pytest tests/integration/test_web_ui_integration.py -v

# Run tests with specific markers
pytest -m "web_integration and not requires_dev_server" -v

# Run tests excluding dev server requirements
pytest -m "not requires_dev_server" tests/playwright/ -v
```

## Test Categories

### 1. CLI Command Tests (`test_web_commands_integration.py`)

Tests the `wqm web` CLI commands:

- **`wqm web install`** - Dependency management
- **`wqm web build`** - Production build process
- **`wqm web dev`** - Development server startup
- **`wqm web start`** - Production server startup  
- **`wqm web status`** - Status reporting

**Key Test Scenarios:**
- Successful command execution
- Error handling (missing npm, build failures)
- Custom options (ports, output directories)
- Environment variable handling
- Mock web-ui directory structures

### 2. Playwright Browser Tests (`test_web_ui_functionality.py`)

Tests web interface functionality with browser automation:

#### Navigation Tests
- Homepage loading
- Navigation menu functionality
- Page routing

#### Workspace Status Page
- Daemon connection status display
- Safety mode toggle functionality
- Read-only mode enforcement
- Real-time status updates
- Configuration display

#### Processing Queue Page  
- Queue items display
- Progress indicators
- Queue management (refresh, clear)
- Item details viewing
- Real-time queue updates

#### Memory Rules Page (CRUD Operations)
- Rules list display
- Create new memory rules
- Edit existing rules
- Delete rules with confirmation
- Form validation
- Rule priority management

#### Error Handling & Safety
- Daemon connection error handling
- Form validation errors
- Dangerous operation confirmations
- Read-only mode enforcement
- Graceful degradation

**Mock Daemon Server:**
- Provides test API endpoints
- Simulates realistic data responses
- Allows testing without real daemon

### 3. Integration Tests (`test_web_ui_integration.py`)

Tests combined CLI and UI workflows:

#### CLI to Web Workflows
- `install` → `build` → `status` → `start` pipeline
- Build output verification
- Configuration consistency
- Error propagation

#### Full Stack Integration
- CLI commands + daemon + web UI interaction
- Configuration sharing between components
- Safety mode consistency
- Error handling coordination

#### Development Workflows
- Development server management
- Production build and serve
- Live reloading testing

## Mock Services

### Mock Daemon Server

Located in `test_web_ui_functionality.py`, provides:

```python
# API endpoints
GET  /api/status                 # Daemon status
GET  /api/memory-rules          # List memory rules  
POST /api/memory-rules          # Create rule
PUT  /api/memory-rules/{id}     # Update rule
DELETE /api/memory-rules/{id}   # Delete rule
GET  /api/processing-queue      # Queue status
POST /api/processing-queue/clear # Clear queue
POST /api/safety-mode           # Toggle safety
POST /api/read-only-mode        # Toggle read-only
```

**Test Data:**
- Sample memory rules with different patterns
- Processing queue with various statuses
- Realistic daemon status information

## Configuration

### Playwright Configuration (`playwright.config.py`)

```python
TEST_CONFIG = {
    "base_url": "http://localhost:3000",
    "dev_server_port": 3000,
    "mock_daemon_port": 8899,
    "timeout": 30000,
    "slow_timeout": 60000,
    "expect_timeout": 10000,
}
```

### Browser Settings
- **Browser:** Chromium (consistent across environments)
- **Headless:** True (for CI/CD)
- **Viewport:** 1280x720 
- **Screenshots:** On failure only
- **Videos:** Retain on failure

### Test Markers

```ini
[tool:pytest]
markers =
    playwright: Playwright browser automation tests
    web_integration: Web integration tests  
    requires_dev_server: Tests requiring development server
    requires_daemon: Tests requiring daemon server
    ui_functional: UI functional tests
```

## Environment Considerations

### Development Environment
- Tests can run with or without development server
- Mock services provide isolation
- Graceful degradation when services unavailable

### CI/CD Environment  
- Headless browser mode
- No external dependencies required
- Mock all external services
- Comprehensive error reporting

### Local Testing
- Optional real development server
- Visual debugging available
- Interactive test development

## Troubleshooting

### Common Issues

**1. Development Server Not Starting**
```bash
# Check Node.js and npm
node --version && npm --version

# Install dependencies
cd web-ui && npm install

# Check package.json scripts
npm run build
npm start
```

**2. Playwright Browser Issues**
```bash  
# Reinstall browsers
python -m playwright install chromium

# Check browser installation
python -m playwright install --help
```

**3. Test Timeouts**
```bash
# Increase timeout for slow systems
pytest --timeout=600 tests/playwright/

# Run without dev server requirements  
pytest -m "not requires_dev_server" tests/playwright/
```

**4. Port Conflicts**
```bash
# Check port usage
lsof -i :3000
lsof -i :8899

# Use different ports
python scripts/run_web_ui_tests.py --dev-server --port 3001
```

### Debug Mode

```bash
# Run with verbose output
python scripts/run_web_ui_tests.py --verbose --all

# Run single test with debug
pytest tests/playwright/test_web_ui_functionality.py::TestWorkspaceStatusPage::test_status_page_displays_info -v -s

# Enable Playwright debug mode
PWDEBUG=1 pytest tests/playwright/test_web_ui_functionality.py -v
```

## Test Coverage

The web UI testing covers:

### ✅ Implemented Features
- CLI command functionality and error handling
- Basic web interface navigation
- Status dashboard display  
- Processing queue management
- Memory rules CRUD operations
- Safety mode toggles
- Read-only mode enforcement
- Form validation and error handling
- Mock API integration
- Development server management

### 🔄 Future Enhancements
- Real daemon integration testing
- Performance and load testing
- Cross-browser compatibility
- Mobile responsive testing
- Accessibility testing
- Advanced UI interactions
- WebSocket real-time updates
- File upload/download testing

## Integration with Task 78

This implementation fulfills Task 78 requirements:

1. **✅ Web Command Testing** - Complete CLI command validation
2. **✅ Browser Automation** - Playwright-based UI testing
3. **✅ Workspace Features** - Status, queue, memory rules testing
4. **✅ Safety Systems** - Safety mode and confirmation dialogs
5. **✅ Real-time Updates** - Mock live data refresh testing
6. **✅ Error Handling** - Comprehensive error scenario coverage
7. **✅ Integration Testing** - CLI + UI workflow validation

The testing infrastructure provides a solid foundation for validating web UI functionality while maintaining isolation and reliability for automated testing environments.

## Example Usage

```bash
# Quick validation run
python scripts/run_web_ui_tests.py --cli-only --no-install

# Full development testing  
python scripts/run_web_ui_tests.py --all --dev-server

# CI/CD pipeline testing
python scripts/run_web_ui_tests.py --all --no-install > test_results.log 2>&1
```

This comprehensive testing approach ensures the web UI functionality is thoroughly validated across all interaction patterns and error scenarios.