use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use fastembed::TextEmbedding;

use crate::io;

#[derive(Debug, Clone)]
pub struct Options {
    pub model: fastembed::EmbeddingModel,
    pub cache_dir: PathBuf,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: fastembed::EmbeddingModel::default(),
            cache_dir: PathBuf::from(".cache"),
        }
    }
}

impl Options {
    pub fn hash(&self) -> String {
        let mut h = blake3::Hasher::new();

        h.update(self.model.to_string().as_bytes());

        h.finalize().to_hex().to_string()
    }
}

#[derive(Default)]
pub struct EmbeddingService {
    pub options: Options,
}

impl EmbeddingService {
    pub fn new() -> Self {
        Self {
            options: Options::default(),
        }
    }

    pub fn model(mut self, model: fastembed::EmbeddingModel) -> Self {
        self.options.model = model;
        self
    }

    pub fn cache_dir(mut self, path: PathBuf) -> Self {
        self.options.cache_dir = path;
        self
    }

    pub fn build(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];

        let cache = io::Cache::new(&self.options.cache_dir, &self.options.hash());
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
            let mut model = embedding_model(&self.options);

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
}

fn embedding_model(options: &Options) -> TextEmbedding {
    let mut model_options = fastembed::InitOptions::new(options.model.clone());
    model_options.cache_dir = options.cache_dir.join(".fastembed_cache").clone();
    model_options.show_download_progress = false;

    TextEmbedding::try_new(model_options).expect("failed to initialize FastEmbed model")
}
