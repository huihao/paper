# Mixtral of Experts (Mistral AI, 2024)

## Paper Summary

Mixtral is an open-weight MoE that proved sparse models can match dense model quality while running at much lower inference cost.

**Paper Link:** https://arxiv.org/abs/2401.04088

## Key Achievement

**Mixtral 8x7B**:
- 46.7B total parameters
- 12.9B active per token (like running a 13B model)
- Matches or beats LLaMA-2 70B quality

**5x cheaper inference** for equivalent quality!

## Architecture

```
8 Experts × 7B parameters each = 46.7B total
Top-2 routing: Each token uses 2 experts
Active: 12.9B parameters per forward pass
```

## Implementation

```bash
python mixtral.py
```

This implementation demonstrates:
- Top-2 expert routing
- SwiGLU expert FFNs
- Parameter efficiency calculation
- Routing statistics

## Model Variants

| Model | Total | Active | Quality vs |
|-------|-------|--------|-----------|
| Mixtral 8x7B | 46.7B | 12.9B | ≈ LLaMA-2 70B |
| Mixtral 8x22B | 176B | 39B | ≈ GPT-4 (some tasks) |

## Why Top-2?

- Simple: exactly 2 experts per token
- Balanced: enough capacity without waste
- Stable: no complex capacity limits needed

## Key Innovations

1. **Open-Weight MoE**: First truly open high-quality MoE
2. **Practical Design**: Simple top-2 routing
3. **SwiGLU Experts**: Modern architecture
4. **Sliding Window Attention**: Efficient long context
5. **Quality-Cost Trade-off**: Dense quality at sparse cost

## Comparison with Dense Models

| Model | Params | Inference Cost | Quality |
|-------|--------|----------------|---------|
| LLaMA-2 70B | 70B | 70B | Baseline |
| Mixtral 8x7B | 46.7B | 12.9B | ≈ Same |
| Cost Savings | - | 5.4x | - |

## Why This Paper Matters

1. **Proved MoE Quality**: Ended debate about MoE vs dense
2. **Open Weights**: Community can build on it
3. **Practical**: Actually deployable at scale
4. **Template**: Influenced subsequent MoE designs
