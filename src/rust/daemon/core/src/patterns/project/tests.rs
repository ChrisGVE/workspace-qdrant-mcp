use super::helpers::file_matches_pattern;
use super::*;

#[test]
fn test_detector_initialization() {
    let detector = ProjectDetector::new();
    assert!(detector.is_ok(), "Should initialize project detector");
}

#[test]
fn test_rust_project_detection() {
    let detector = ProjectDetector::new().unwrap();
    let files = vec![
        "Cargo.toml".to_string(),
        "Cargo.lock".to_string(),
        "src/main.rs".to_string(),
        "src/lib.rs".to_string(),
        "tests/integration_test.rs".to_string(),
    ];

    let project_info = detector.analyze_project(&files);

    assert_eq!(project_info.primary_language, Some("rust".to_string()));
    assert!(project_info.languages.contains(&"rust".to_string()));
    assert!(!project_info.build_systems.is_empty());
    assert_eq!(project_info.build_systems[0].name, "cargo");
    assert!(project_info.confidence >= ProjectConfidence::High);
}

#[test]
fn test_javascript_project_detection() {
    let detector = ProjectDetector::new().unwrap();
    let files = vec![
        "package.json".to_string(),
        "package-lock.json".to_string(),
        "src/index.js".to_string(),
        "src/App.jsx".to_string(),
        "public/index.html".to_string(),
    ];

    let project_info = detector.analyze_project(&files);

    assert!(
        project_info.primary_language == Some("javascript".to_string())
            || project_info.primary_language == Some("typescript".to_string())
    );
    assert!(!project_info.build_systems.is_empty());
    assert!(project_info.frameworks.contains(&"react".to_string()));
}

#[test]
fn test_python_project_detection() {
    let detector = ProjectDetector::new().unwrap();
    let files = vec![
        "pyproject.toml".to_string(),
        "setup.py".to_string(),
        "src/main.py".to_string(),
        "tests/test_main.py".to_string(),
        "requirements.txt".to_string(),
    ];

    let project_info = detector.analyze_project(&files);

    assert_eq!(project_info.primary_language, Some("python".to_string()));
    assert!(project_info.languages.contains(&"python".to_string()));
    assert!(project_info.confidence >= ProjectConfidence::Medium);
}

#[test]
fn test_monorepo_detection() {
    let detector = ProjectDetector::new().unwrap();
    let files = vec![
        "Cargo.toml".to_string(),
        "package.json".to_string(),
        "go.mod".to_string(),
        "rust-service/src/main.rs".to_string(),
        "js-frontend/src/index.js".to_string(),
        "go-api/main.go".to_string(),
    ];

    let project_info = detector.analyze_project(&files);

    assert_eq!(project_info.project_type, ProjectType::Monorepo);
    assert!(project_info.build_systems.len() > 1);
    assert!(project_info.languages.len() > 1);
}

#[test]
fn test_documentation_project_detection() {
    let detector = ProjectDetector::new().unwrap();
    let files = vec![
        "README.md".to_string(),
        "docs/guide.md".to_string(),
        "docs/api.md".to_string(),
        "docs/tutorial.md".to_string(),
        "mkdocs.yml".to_string(),
    ];

    let project_info = detector.analyze_project(&files);

    assert_eq!(project_info.project_type, ProjectType::Documentation);
}

#[test]
fn test_file_pattern_matching() {
    assert!(file_matches_pattern("package.json", "package.json"));
    assert!(file_matches_pattern("src/main.rs", "*.rs"));
    assert!(file_matches_pattern("test.txt", "test.txt"));
    assert!(file_matches_pattern("src/App.jsx", "*App.jsx"));
    assert!(!file_matches_pattern("main.py", "*.rs"));
}

#[test]
fn test_global_detector() {
    let detector = ProjectDetector::global();
    assert!(detector.is_ok(), "Global detector should be available");
}

#[test]
fn test_convenience_function() {
    let files = vec!["Cargo.toml".to_string(), "src/main.rs".to_string()];

    let project_info = analyze_project_from_files(&files);
    assert_eq!(project_info.primary_language, Some("rust".to_string()));
}
