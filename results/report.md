# Model Evaluation Report

## Summary

### Average Scores

| Model | Factuality (Mean ± SD) | Relevance (Mean ± SD) | Average (Mean ± SD) |
|-------|----------------------|----------------------|---------------------|
| Baseline | 0.778 ± 0.348 | 0.985 ± 0.092 | 0.881 ± 0.194 |
| Baseline Rag | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Fine Tuned | 0.656 ± 0.437 | 0.755 ± 0.387 | 0.706 ± 0.347 |
| Fine Tuned Rag | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Configuration

- **Baseline Model:** gpt-3.5-turbo
- **Fine-tuned Model:** ft:gpt-3.5-turbo-0125:personal:myown-medassist:BagH0Z8U
- **RAG Enabled:** False
- **Evaluation Timestamp:** 2025-05-25 19:11:49

## Detailed Results

The detailed evaluation results are available in JSON format at `evaluation_results.json`.
