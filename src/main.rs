use rusty_embeddings::EmbeddingService;

fn main() -> anyhow::Result<()> {
    let service = EmbeddingService::new().model(fastembed::EmbeddingModel::GTEBaseENV15);

    let texts = vec![
        "The stock market closed higher after positive earnings reports.".to_string(),
        "Inflation rates are expected to decrease next quarter.".to_string(),
        "Cryptocurrency prices continue to show high volatility.".to_string(),
        "Investors are optimistic about the technology sector.".to_string(),
        "The company announced a new share buyback program.".to_string(),
    ];

    let embeddings = service.build(texts.clone())?;
    let query = "positive";

    let sorted_indices = service.rank_candidates(query, &embeddings)?;

    println!("Similarity ranking:");
    for (idx, score) in sorted_indices {
        println!("  {:<2} | score: {:.4} | {}", idx, score, texts[idx]);
    }

    Ok(())
}
