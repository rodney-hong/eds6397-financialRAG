# Metadata Ablation — retrieval metrics (K=5)

All three share the same 40 questions. 'Engineered (no metadata)' is identical to 'Engineered' except the Year/Month pre-filter is disabled; the delta between them isolates the metadata filter's contribution.

| System | Hit Rate@5 | MRR | Recall@5 |
| --- | --- | --- | --- |
| Baseline | 0.325 | 0.128 | 0.217 |
| Engineered (no metadata) | 0.35 | 0.172 | 0.2 |
| Engineered | 0.4 | 0.226 | 0.258 |
