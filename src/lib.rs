use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

mod io;

#[derive(Clone)]
pub struct EmbeddingService {
    pub model: EmbeddingModel,
    pub cache_dir: PathBuf,
    model_inner: Arc<OnceLock<Mutex<TextEmbedding>>>,
}

impl Default for EmbeddingService {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::default(),
            cache_dir: PathBuf::from(".cache"),
            model_inner: Arc::new(OnceLock::new()),
        }
    }
}

impl EmbeddingService {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, model: EmbeddingModel) -> Self {
        self.model = model;
        self
    }

    pub fn cache_dir(mut self, path: PathBuf) -> Self {
        self.cache_dir = path;
        self
    }

    fn hash_options(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(self.model.to_string().as_bytes());
        h.finalize().to_hex().to_string()
    }

    fn cache(&self) -> io::Cache {
        io::Cache::new(&self.cache_dir, &self.hash_options())
    }

    /// lazily initialize the fastembed model.
    ///
    /// uses `OnceLock` + `Mutex` so we can safely call `embed` (which needs `&mut self`)
    /// from multiple threads.
    fn get_model(&self) -> Result<&Mutex<TextEmbedding>> {
        if let Some(m) = self.model_inner.get() {
            return Ok(m);
        }

        let mut options = InitOptions::new(self.model.clone());
        options.cache_dir = self.cache_dir.join(".fastembed_cache");
        options.show_download_progress = false;

        let text_model = TextEmbedding::try_new(options)
            .context("failed to initialize fastembed TextEmbedding")?;

        // if another thread beat us to initialization, just use that one
        match self.model_inner.set(Mutex::new(text_model)) {
            Ok(()) => {}
            Err(_already) => {}
        }

        self.model_inner
            .get()
            .ok_or_else(|| anyhow::anyhow!("embedding model not initialized"))
    }

    /// build embeddings for the given texts, using cache + dedup.
    ///
    /// note: empty input returns `Ok(Vec::new())`.
    pub fn build<S>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>>
    where
        S: AsRef<str>,
    {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let cache = self.cache();
        cache
            .ensure_dirs()
            .context("failed to initialize embedding cache dirs")?;

        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut key_positions: HashMap<String, Vec<usize>> = HashMap::new();
        let mut misses: Vec<(String, String)> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let text_ref = text.as_ref();
            let key = io::Cache::key_for_text(text_ref);

            if let Some(vec) = cache.load(&key) {
                results[i] = Some(vec);
            } else {
                let entry = key_positions.entry(key.clone()).or_default();
                if entry.is_empty() {
                    // only embed each unique text once
                    misses.push((key.clone(), text_ref.to_owned()));
                }
                entry.push(i);
            }
        }

        if !misses.is_empty() {
            let model_lock = self.get_model()?;
            let mut model = model_lock
                .lock()
                .map_err(|_| anyhow::anyhow!("embedding model mutex poisoned"))?;

            let batch: Vec<String> = misses.iter().map(|(_, t)| t.clone()).collect();
            let vecs = model
                .embed(batch, None)
                .context("embedding texts with fastembed")?;

            if vecs.len() != misses.len() {
                bail!(
                    "fastembed returned {} vectors for {} inputs",
                    vecs.len(),
                    misses.len()
                );
            }

            for ((key, _), v) in misses.into_iter().zip(vecs.into_iter()) {
                if let Err(e) = cache.save(&key, &v) {
                    eprintln!("warning: cache write failed for {key}: {e}");
                }

                if let Some(positions) = key_positions.remove(&key) {
                    for idx in positions {
                        results[idx] = Some(v.clone());
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(results.len());
        for (idx, maybe_vec) in results.into_iter().enumerate() {
            let v = maybe_vec.with_context(|| format!("missing embedding for index {idx}"))?;
            out.push(v);
        }

        Ok(out)
    }

    /// embed a single query string.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let mut res = self.build(&[query])?;
        res.pop()
            .ok_or_else(|| anyhow::anyhow!("query embedding unexpectedly missing"))
    }

    /// rank candidate embeddings by cosine similarity to the query embedding.
    pub fn rank_embeddings(
        &self,
        query_embedding: &[f32],
        candidates: &[Vec<f32>],
    ) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query_embedding, v)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        scored
    }

    /// convenience: embed query, then rank candidate embeddings.
    pub fn rank_candidates(
        &self,
        query: &str,
        candidates: &[Vec<f32>],
    ) -> Result<Vec<(usize, f32)>> {
        let query_vec = self.embed_query(query)?;
        Ok(self.rank_embeddings(&query_vec, candidates))
    }
}

fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let len = vec1.len().min(vec2.len());
    if len == 0 {
        return 0.0;
    }

    let dot_product: f32 = vec1
        .iter()
        .zip(vec2.iter())
        .take(len)
        .map(|(a, b)| a * b)
        .sum();

    let magnitude1: f32 = vec1.iter().take(len).map(|a| a * a).sum::<f32>().sqrt();
    let magnitude2: f32 = vec2.iter().take(len).map(|b| b * b).sum::<f32>().sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1 * magnitude2)
    }
}
