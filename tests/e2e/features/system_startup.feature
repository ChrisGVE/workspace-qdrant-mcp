Feature: System Startup and Initialization
  As a system operator
  I want the system to start up correctly
  So that all components are ready for operation

  Background:
    Given the system is not running
    And all previous test data is cleaned up

  Scenario: Sequential component startup
    When I start Qdrant service
    Then Qdrant should be healthy within 30 seconds
    When I start daemon service
    Then daemon should be healthy within 30 seconds
    When I start MCP server
    Then MCP server should be healthy within 30 seconds
    And all components should be running

  Scenario: Component dependency validation
    When I try to start MCP server without daemon
    Then MCP server should enter degraded mode
    And MCP server should log daemon unavailability warning

  Scenario: Parallel component startup
    When I start all components simultaneously
    Then all components should reach healthy state
    And component startup should complete within 60 seconds
    And no startup conflicts should occur

  Scenario: Startup with missing configuration
    Given Qdrant configuration file is missing
    When I start Qdrant service
    Then Qdrant should create default configuration
    And Qdrant should start successfully

  Scenario: Recovery from partial startup
    Given Qdrant is running
    And daemon is running
    And MCP server failed to start
    When I restart MCP server
    Then MCP server should connect to daemon
    And system should be fully operational
