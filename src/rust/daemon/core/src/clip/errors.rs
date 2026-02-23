/// CLIP encoder error types.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClipError {
    #[error("CLIP model initialization failed: {0}")]
    InitFailed(String),

    #[error("Image encoding failed: {0}")]
    ImageEncodingFailed(String),

    #[error("Text encoding failed: {0}")]
    TextEncodingFailed(String),

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),
}
