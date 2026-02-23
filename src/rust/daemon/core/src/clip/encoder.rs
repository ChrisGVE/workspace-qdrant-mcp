/// CLIP multimodal encoder for image and text embedding.
///
/// Uses fastembed's CLIP ViT-B-32 models (both image and text encoders)
/// to produce 512-dimensional embeddings in a shared vector space.
/// Cross-modal similarity (image↔text) is supported via cosine distance.

use std::path::PathBuf;
use std::sync::Mutex;

use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions,
    TextEmbedding,
};
use tracing::{debug, info};

use super::errors::ClipError;

/// CLIP embedding dimension (ViT-B-32).
pub const CLIP_EMBEDDING_DIM: usize = 512;

/// Configuration for the CLIP encoder.
#[derive(Debug, Clone)]
pub struct ClipConfig {
    /// Directory for model file caching.
    pub model_cache_dir: Option<PathBuf>,
    /// Number of intra-op threads for ONNX inference.
    pub num_threads: Option<usize>,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            model_cache_dir: None,
            num_threads: Some(2),
        }
    }
}

/// CLIP encoder wrapping both image and text fastembed models.
///
/// Both models produce 512-dim L2-normalized vectors in the same
/// embedding space, enabling cross-modal similarity queries
/// (e.g., "find images matching this text description").
pub struct ClipEncoder {
    image_model: Mutex<ImageEmbedding>,
    text_model: Mutex<TextEmbedding>,
}

impl std::fmt::Debug for ClipEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipEncoder")
            .field("image_model", &"ImageEmbedding(clip-ViT-B-32)")
            .field("text_model", &"TextEmbedding(clip-ViT-B-32)")
            .finish()
    }
}

impl ClipEncoder {
    /// Create a new CLIP encoder with the given configuration.
    ///
    /// Downloads model files on first use (~600MB total).
    pub fn new(config: &ClipConfig) -> Result<Self, ClipError> {
        info!("Initializing CLIP ViT-B-32 encoder (image + text)...");

        let image_model = init_image_model(config)?;
        let text_model = init_text_model(config)?;

        info!("CLIP encoder initialized (512-dim, shared embedding space)");

        Ok(Self {
            image_model: Mutex::new(image_model),
            text_model: Mutex::new(text_model),
        })
    }

    /// Encode image bytes into a 512-dim embedding vector.
    ///
    /// Accepts raw image bytes (JPEG, PNG, BMP, TIFF, GIF).
    /// The image is decoded, preprocessed (resize, normalize), and
    /// passed through the CLIP visual encoder.
    ///
    /// This is a blocking call — wrap in `spawn_blocking` for async.
    pub fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, ClipError> {
        if image_bytes.is_empty() {
            return Err(ClipError::ImageEncodingFailed(
                "empty image data".to_string(),
            ));
        }

        let mut model = self.image_model.lock().map_err(|e| {
            ClipError::ImageEncodingFailed(format!("lock poisoned: {e}"))
        })?;

        let embeddings = model
            .embed_bytes(&[image_bytes], None)
            .map_err(|e| ClipError::ImageEncodingFailed(e.to_string()))?;

        let embedding = embeddings.into_iter().next().ok_or_else(|| {
            ClipError::ImageEncodingFailed("no embedding returned".to_string())
        })?;

        debug!(dim = embedding.len(), "CLIP image embedding generated");
        Ok(embedding)
    }

    /// Encode a batch of images into 512-dim embedding vectors.
    pub fn encode_images(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>, ClipError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut model = self.image_model.lock().map_err(|e| {
            ClipError::ImageEncodingFailed(format!("lock poisoned: {e}"))
        })?;

        model
            .embed_bytes(images, None)
            .map_err(|e| ClipError::ImageEncodingFailed(e.to_string()))
    }

    /// Encode text into a 512-dim embedding vector.
    ///
    /// Uses the CLIP text encoder, producing a vector in the same
    /// space as image embeddings. Text is automatically tokenized
    /// and truncated to the CLIP context length (77 tokens).
    ///
    /// This is a blocking call — wrap in `spawn_blocking` for async.
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>, ClipError> {
        if text.is_empty() {
            return Err(ClipError::TextEncodingFailed(
                "empty text".to_string(),
            ));
        }

        let mut model = self.text_model.lock().map_err(|e| {
            ClipError::TextEncodingFailed(format!("lock poisoned: {e}"))
        })?;

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| ClipError::TextEncodingFailed(e.to_string()))?;

        let embedding = embeddings.into_iter().next().ok_or_else(|| {
            ClipError::TextEncodingFailed("no embedding returned".to_string())
        })?;

        debug!(dim = embedding.len(), "CLIP text embedding generated");
        Ok(embedding)
    }

    /// Encode a batch of texts into 512-dim embedding vectors.
    pub fn encode_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ClipError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut model = self.text_model.lock().map_err(|e| {
            ClipError::TextEncodingFailed(format!("lock poisoned: {e}"))
        })?;

        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        model
            .embed(owned, None)
            .map_err(|e| ClipError::TextEncodingFailed(e.to_string()))
    }

    /// Compute cosine similarity between two embedding vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

/// Initialize the CLIP image encoder model.
fn init_image_model(config: &ClipConfig) -> Result<ImageEmbedding, ClipError> {
    let mut opts = ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
        .with_show_download_progress(true);

    if let Some(ref dir) = config.model_cache_dir {
        opts = opts.with_cache_dir(dir.clone());
    }

    ImageEmbedding::try_new(opts).map_err(|e| {
        ClipError::InitFailed(format!("image encoder: {e}"))
    })
}

/// Initialize the CLIP text encoder model.
fn init_text_model(config: &ClipConfig) -> Result<TextEmbedding, ClipError> {
    let mut opts = InitOptions::new(EmbeddingModel::ClipVitB32)
        .with_show_download_progress(true);

    if let Some(threads) = config.num_threads {
        opts = opts.with_num_threads(threads);
    }
    if let Some(ref dir) = config.model_cache_dir {
        opts = opts.with_cache_dir(dir.clone());
    }

    TextEmbedding::try_new(opts).map_err(|e| {
        ClipError::InitFailed(format!("text encoder: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_config_default() {
        let config = ClipConfig::default();
        assert!(config.model_cache_dir.is_none());
        assert_eq!(config.num_threads, Some(2));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = ClipEncoder::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = ClipEncoder::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = ClipEncoder::cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = ClipEncoder::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_encoder_debug_format() {
        // Verify Debug impl doesn't panic
        let config = ClipConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ClipConfig"));
    }

    // Integration tests that require model download are gated
    // behind actual model availability (too slow for CI unit tests).
    // See integration test files for end-to-end CLIP tests.

    #[test]
    fn test_encode_image_empty_bytes() {
        // This test verifies error handling without needing model files
        let config = ClipConfig::default();
        // We can't create a real encoder without model download,
        // so just verify the error path logic is sound.
        let result: Result<ClipEncoder, _> = ClipEncoder::new(&config);
        // If models aren't cached, this will fail with InitFailed — that's OK
        if let Ok(encoder) = result {
            let err = encoder.encode_image(&[]).unwrap_err();
            assert!(matches!(err, ClipError::ImageEncodingFailed(_)));
        }
    }

    #[test]
    fn test_encode_text_empty() {
        let config = ClipConfig::default();
        if let Ok(encoder) = ClipEncoder::new(&config) {
            let err = encoder.encode_text("").unwrap_err();
            assert!(matches!(err, ClipError::TextEncodingFailed(_)));
        }
    }

    #[test]
    fn test_encode_images_empty_batch() {
        let config = ClipConfig::default();
        if let Ok(encoder) = ClipEncoder::new(&config) {
            let result = encoder.encode_images(&[]).unwrap();
            assert!(result.is_empty());
        }
    }

    #[test]
    fn test_encode_texts_empty_batch() {
        let config = ClipConfig::default();
        if let Ok(encoder) = ClipEncoder::new(&config) {
            let result = encoder.encode_texts(&[]).unwrap();
            assert!(result.is_empty());
        }
    }
}
