# Baseline vs Engineered — Results (K=5)

LLM backend for generation & judging: **Anthropic (Claude)**

Corpus: 40 Treasury bulletins (2010-2025), 40 evaluation questions.

| Metric | Baseline | Engineered | Delta (Eng - Base) |
| --- | --- | --- | --- |
| Hit Rate@5 | 0.325 | 0.4 | 0.075 |
| MRR | 0.128 | 0.226 | 0.098 |
| Recall@5 | 0.0 | 0.003 | 0.003 |
| Groundedness | 0.904 | 0.959 | 0.055 |
| Factual Accuracy | 0.025 | 0.025 | 0.0 |
| Hallucination Rate | 0.062 | 0.037 | -0.025 |
