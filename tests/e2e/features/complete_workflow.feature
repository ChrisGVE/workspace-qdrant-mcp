Feature: Complete System Workflow
  As a user
  I want to perform complete workflows from ingestion to search
  So that I can validate end-to-end functionality

  Background:
    Given the system is fully operational
    And I have a test project workspace
    And the project is initialized with Git

  Scenario: Complete document ingestion workflow
    When I configure a watch folder for the project
    And I create a new Python file "src/main.py"
    Then the daemon should detect the file within 5 seconds
    And the file should be ingested to Qdrant within 10 seconds
    And metadata should include project context
    And I should be able to search for the file content

  Scenario: Multi-file ingestion and search
    When I create 10 Python files in the project
    Then all files should be ingested within 30 seconds
    When I search for "function definition"
    Then search results should include relevant files
    And results should be ranked by relevance

  Scenario: Real-time file modification tracking
    Given a file "docs/api.md" exists in the project
    When I modify the file content
    Then the daemon should detect the change within 5 seconds
    And the updated content should be re-ingested
    And search should return the updated content

  Scenario: Project switching workflow
    Given I have two separate projects
    And both projects have files ingested
    When I switch to project A
    And I search for content
    Then search results should only include project A files
    When I switch to project B
    And I search for the same content
    Then search results should only include project B files

  Scenario: Collection management workflow
    When I create a new collection "test-collection"
    And I add documents to "test-collection"
    Then I should be able to list all collections
    And "test-collection" should appear in the list
    When I delete "test-collection"
    Then "test-collection" should not appear in collection list
    And documents should be removed from Qdrant
