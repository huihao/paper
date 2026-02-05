# Training Compute-Optimal Large Language Models (Chinchilla) (Hoffmann et al., 2022)

## Paper Summary

Chinchilla demonstrated that token count matters more than parameter count for a fixed compute budget, revising the scaling laws from Kaplan et al.

**Paper Link:** https://arxiv.org/abs/2203.15556

## Key Finding: The 20:1 Rule

For compute-optimal training:
- Train on approximately **20 tokens per parameter**
- Parameters and data should scale equally with compute: N ∝ C^0.5, D ∝ C^0.5
- Most pre-Chinchilla models were severely undertrained

## Comparison with Kaplan et al.

| Aspect | Kaplan (2020) | Chinchilla (2022) |
|--------|--------------|-------------------|
| N scaling | N ∝ C^0.73 | N ∝ C^0.50 |
| D scaling | D ∝ C^0.27 | D ∝ C^0.50 |
| Focus | Bigger models | More data |

## How Undertrained Were Previous Models?

| Model | Parameters | Actual Tokens | Optimal Tokens |
|-------|-----------|---------------|----------------|
| GPT-3 | 175B | 300B | 3.5T |
| Gopher | 280B | 300B | 5.6T |
| Chinchilla | 70B | 1.4T | 1.4T ✓ |

## The Chinchilla Result

- Chinchilla (70B) matched Gopher (280B) performance
- Trained on 4x more tokens
- 4x faster at inference
- Used the same compute budget

## Implementation

```bash
python chinchilla.py
```

This implementation demonstrates:
- Chinchilla scaling law calculations
- Optimal compute allocation
- Model efficiency analysis
- Comparison with Kaplan's recommendations

## Practical Implications

1. **Data Quality Matters**: Need high-quality data at scale
2. **Smaller Can Be Better**: Well-trained small models beat undertrained large ones
3. **Inference Efficiency**: Smaller models are cheaper to deploy
4. **Data-Centric AI**: Shifted focus from model architecture to data curation

## Post-Chinchilla Era

After Chinchilla, the industry moved to "overtrained" models:

| Model | Tokens/Param Ratio |
|-------|-------------------|
| Chinchilla (optimal) | 20:1 |
| LLaMA 7B | 143:1 |
| LLaMA 65B | 22:1 |
| Mistral 7B | ~1000:1 |

Why overtrain?
- Inference is more expensive than training at scale
- Smaller, well-trained models are cheaper to deploy
- Diminishing returns are acceptable for better inference economics

## Why This Paper Matters

- Fundamentally changed how models are trained
- Showed data scaling was undervalued
- Enabled efficient open-source models (LLaMA, Mistral)
- Shifted focus to data quality and curation
