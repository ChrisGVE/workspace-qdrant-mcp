//! N8 environment-key tests (PRD F-01). One positive assertion per key plus the
//! enumeration contract.

use wqm_common::names::EnvVar;

#[test]
fn qdrant_url_key() {
    assert_eq!(EnvVar::QdrantUrl.key(), "QDRANT_URL");
}

#[test]
fn qdrant_api_key_key() {
    assert_eq!(EnvVar::QdrantApiKey.key(), "QDRANT_API_KEY");
}

#[test]
fn fastembed_model_key() {
    assert_eq!(EnvVar::FastembedModel.key(), "FASTEMBED_MODEL");
}

#[test]
fn database_path_key() {
    assert_eq!(EnvVar::DatabasePath.key(), "WQM_DATABASE_PATH");
}

#[test]
fn log_level_key() {
    assert_eq!(EnvVar::LogLevel.key(), "WQM_LOG_LEVEL");
}

#[test]
fn all_enumerates_the_five_env_vars_in_order() {
    assert_eq!(
        EnvVar::ALL,
        [
            EnvVar::QdrantUrl,
            EnvVar::QdrantApiKey,
            EnvVar::FastembedModel,
            EnvVar::DatabasePath,
            EnvVar::LogLevel,
        ]
    );
}
