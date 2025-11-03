use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Options {
    pub model: fastembed::EmbeddingModel,
    pub cache_dir: PathBuf,
}
