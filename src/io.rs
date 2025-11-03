use anyhow::{Context, Result};
use bincode::{Decode, Encode};
use blake3::Hasher;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Encode, Decode)]
struct EmbedFile {
    data: Vec<f32>,
}

pub struct Cache {
    cache_dir: PathBuf,
}

impl Cache {
    pub fn new(path: &Path, settings_hash: &str) -> Self {
        Self {
            cache_dir: path
                .join("rusty-embeddings")
                .join("embeddings")
                .join(settings_hash),
        }
    }

    pub fn key_for_text(text: &str) -> String {
        let mut h = Hasher::new();

        h.update(text.as_bytes());

        h.finalize().to_hex().to_string()
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        fs::create_dir_all(&self.cache_dir)
            .with_context(|| format!("creating cache dir {}", self.cache_dir.display()))
    }

    pub fn load(&self, key: &str) -> Option<Vec<f32>> {
        let path = self.path_for_key(key);
        let mut f = fs::File::open(&path).ok()?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).ok()?;
        let (parsed, _): (EmbedFile, _) =
            bincode::decode_from_slice(&buf, bincode::config::standard()).ok()?;
        Some(parsed.data)
    }

    pub fn save(&self, key: &str, vec: &[f32]) -> Result<()> {
        let path = self.path_for_key(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating cache shard {}", parent.display()))?;
        }
        let tmp = path.with_extension("bin.tmp");
        {
            let mut f = fs::File::create(&tmp)
                .with_context(|| format!("creating temp {}", tmp.display()))?;
            let payload = EmbedFile {
                data: Vec::from(vec),
            };
            let bytes = bincode::encode_to_vec(&payload, bincode::config::standard())?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &path)
            .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }

    fn path_for_key(&self, key_hex: &str) -> PathBuf {
        self.cache_dir.join(key_hex).with_extension("bin")
    }
}
