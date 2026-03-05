mod encoder;
/// CLIP multimodal embedding module.
///
/// Provides CLIP ViT-B-32 image and text encoding via fastembed's ONNX
/// models. Both encoders produce 512-dimensional L2-normalized vectors
/// in a shared embedding space, enabling cross-modal similarity search.
///
/// # Usage
///
/// ```rust,no_run
/// use workspace_qdrant_core::clip::{ClipEncoder, ClipConfig};
///
/// let encoder = ClipEncoder::new(&ClipConfig::default()).unwrap();
///
/// // Encode an image
/// let image_bytes = std::fs::read("photo.jpg").unwrap();
/// let image_vec = encoder.encode_image(&image_bytes).unwrap();
///
/// // Encode text (in the same vector space)
/// let text_vec = encoder.encode_text("a photo of a cat").unwrap();
///
/// // Cross-modal similarity
/// let similarity = ClipEncoder::cosine_similarity(&image_vec, &text_vec);
/// ```
pub mod errors;

pub use encoder::{ClipConfig, ClipEncoder, CLIP_EMBEDDING_DIM};
pub use errors::ClipError;
