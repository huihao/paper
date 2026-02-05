# Textbooks Are All You Need (Gunasekar et al., 2023)

## Paper Summary

This paper demonstrated that high-quality synthetic data allows small models (1.3B) to outperform much larger ones (15B+), shifting focus from scale to data quality.

**Paper Link:** https://arxiv.org/abs/2306.11644

## The Key Finding

**Phi-1 (1.3B)** outperforms **StarCoder (15.5B)** on coding benchmarks!

| Model | Params | HumanEval | MBPP |
|-------|--------|-----------|------|
| StarCoder | 15.5B | 33.6% | 43.6% |
| CodeLlama | 7B | 29.3% | 41.4% |
| **Phi-1** | **1.3B** | **50.6%** | **55.5%** |

The secret: "Textbook quality" training data.

## Implementation

```bash
python textbooks.py
```

This implementation demonstrates:
- Textbook-style data generation
- Quality assessment metrics
- Data filtering by quality
- Comparison with traditional approaches

## The Textbook Philosophy

### Traditional Approach
- Scrape the internet
- Train on everything
- Bigger model = better results

### Textbook Approach
- Curate high-quality content
- Generate synthetic educational data
- Quality > Quantity

## What Makes Data "Textbook Quality"?

1. **Educational**: Teaches, not just contains
2. **Progressive**: Simple to complex
3. **Clear**: No ambiguity
4. **Correct**: No errors
5. **Diverse**: Covers topic space systematically

## The Phi Model Series

| Model | Params | Focus | Performance |
|-------|--------|-------|-------------|
| Phi-1 | 1.3B | Python | Beats 15B code models |
| Phi-1.5 | 1.3B | General | Beats 10B models |
| Phi-2 | 2.7B | General | Matches 13B+ |
| Phi-3 | 3.8B | General | Matches Llama-3-8B |

Each version improves **data quality**, not just size.

## Why This Paper Matters

1. **Changed Narrative**: Size isn't everything
2. **Data-Centric AI**: Quality over quantity
3. **Efficiency**: Smaller models, better data
4. **Synthetic Data**: LLMs generating training data
5. **Edge Deployment**: Small, capable models
