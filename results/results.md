# Baseline vs Engineered — Results (K=5)

LLM backend for generation & judging: **OFFLINE deterministic heuristic**

Corpus: 48 Treasury bulletins (2022-2025), 25 evaluation questions.

| Metric | Baseline | Engineered | Delta (Eng - Base) |
| --- | --- | --- | --- |
| Hit Rate@5 | 0.76 | 1.0 | 0.24 |
| MRR | 0.547 | 1.0 | 0.453 |
| Recall@5 | 0.38 | 0.98 | 0.6 |
| Groundedness | 1.0 | 1.0 | 0.0 |
| Factual Accuracy | 0.4 | 0.96 | 0.56 |
| Hallucination Rate | 0.0 | 0.0 | 0.0 |
