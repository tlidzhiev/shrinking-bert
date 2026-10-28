# Shrinking BERT for NER (CoNLL-2003): results so far

Task: token classification (NER) on CoNLL-2003, metric: entity-level micro F1 (`seqeval`).
Target: a model of at most 20M parameters that loses as little quality as possible compared to
fine-tuned `bert-base-cased`. All numbers below come from the
[Comet project](https://www.comet.com/timur-lidzhiev/shrinking-bert/view/new/panels).

## Results

| Run | What it is | Params, M | Steps | Val F1 (best) | Test F1 | Test loss |
|---|---|---|---|---|---|---|
| `baseline` | `bert-base-cased`, fine-tuned (the teacher) | 107.73 | 2000 | 0.9448 | **0.9009** | 0.1905 |
| `factorized_embed` | `bert-base-cased` with embeddings factorized to rank E=64 (SVD init), fine-tuned | 87.36 | 2000 | 0.9211 | **0.8714** | 0.2280 |
| `student_kd` | BERT 4 layers, hidden 384, random init, distilled from the teacher | 18.43 | 20000 | 0.7913 | **0.7137** | 0.4970 |
| `student_kd_factorized_embed` | BERT 10 layers, hidden 384, embeddings E=64, random init, distilled | 19.83 | 20000 | 0.8070 | **0.7405** | 0.5004 |

Validation F1 is the best checkpoint (`load_best_model_at_end`); test F1 is computed once for that
checkpoint. Each run is a single seed (42).
