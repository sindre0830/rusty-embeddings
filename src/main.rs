use rusty_embeddings::EmbeddingService;

fn main() -> anyhow::Result<()> {
    // sample texts
    let texts = vec![
        "hello world".to_string(),
        "embedded caching example".to_string(),
        "another one".to_string(),
    ];

    // create embedding instance with default options
    let vectors = EmbeddingService::new()
        .model(fastembed::EmbeddingModel::GTEBaseENV15)
        .build(texts)?;

    println!("generated {} embeddings", vectors.len());
    println!("first embedding length: {}", vectors[0].len());

    Ok(())
}
