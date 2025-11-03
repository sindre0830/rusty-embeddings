use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use fastembed::TextEmbedding;

mod io;

pub struct EmbeddingService {
    pub model: fastembed::EmbeddingModel,
    pub cache_dir: PathBuf,
}

impl Default for EmbeddingService {
    fn default() -> Self {
        Self {
            model: fastembed::EmbeddingModel::default(),
            cache_dir: PathBuf::from(".cache"),
        }
    }
}

impl EmbeddingService {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, model: fastembed::EmbeddingModel) -> Self {
        self.model = model;
        self
    }

    pub fn cache_dir(mut self, path: PathBuf) -> Self {
        self.cache_dir = path;
        self
    }

    pub fn hash_options(&self) -> String {
        let mut h = blake3::Hasher::new();

        h.update(self.model.to_string().as_bytes());

        h.finalize().to_hex().to_string()
    }

    /// Builds embeddings for the given texts.
    pub fn build(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];

        let cache = io::Cache::new(&self.cache_dir, &self.hash_options());
        cache.ensure_dirs()?;

        let mut key_positions: HashMap<String, Vec<usize>> = HashMap::new();
        let mut misses: Vec<(String, String)> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let key = io::Cache::key_for_text(text);
            if let Some(vec) = cache.load(&key) {
                results[i] = Some(vec);
            } else {
                let entry = key_positions.entry(key.clone()).or_default();
                if entry.is_empty() {
                    misses.push((key.clone(), text.clone()));
                }
                entry.push(i);
            }
        }

        if !misses.is_empty() {
            let mut model = load_model(self);

            let batch: Vec<String> = misses.iter().map(|(_, t)| t.clone()).collect();
            let vecs = model
                .embed(batch, None)
                .context("embedding texts with fastembed")?;

            for ((key, _), v) in misses.into_iter().zip(vecs.into_iter()) {
                if let Err(e) = cache.save(&key, &v) {
                    eprintln!("warning: cache write failed for {key}: {e}");
                }
                if let Some(pos) = key_positions.remove(&key) {
                    for idx in pos {
                        results[idx] = Some(v.clone());
                    }
                }
            }
        }

        Ok(results
            .into_iter()
            .map(|x| x.expect("embedding should be present"))
            .collect())
    }

    /// returns indices of candidates sorted by cosine similarity (descending)
    pub fn rank_candidates(
        &self,
        query: &str,
        candidates: &[Vec<f32>],
    ) -> Result<Vec<(usize, f32)>> {
        let query_vec = self
            .build(vec![query.to_string()])?
            .pop()
            .expect("query embedding missing");

        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(&query_vec, v)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        Ok(scored)
    }
}

fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f32 = vec1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let magnitude2: f32 = vec2.iter().map(|b| b * b).sum::<f32>().sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1 * magnitude2)
    }
}

fn load_model(service: &EmbeddingService) -> TextEmbedding {
    let mut model_options = fastembed::InitOptions::new(service.model.clone());
    model_options.cache_dir = service.cache_dir.join(".fastembed_cache").clone();
    model_options.show_download_progress = false;

    TextEmbedding::try_new(model_options).expect("failed to initialize FastEmbed model")
}
