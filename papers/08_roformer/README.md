# RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)

## Paper Summary

RoFormer introduced Rotary Position Embedding (RoPE), which became the standard positional encoding for modern long-context LLMs.

**Paper Link:** https://arxiv.org/abs/2104.09864

## What is RoPE?

RoPE encodes position by rotating query and key vectors in 2D subspaces.

### The Rotation Formula

For position m and dimension pair (2i, 2i+1):

```
[q_{2i}  ]     [cos(mθ_i)  -sin(mθ_i)] [q_{2i}  ]
[q_{2i+1}]  =  [sin(mθ_i)   cos(mθ_i)] [q_{2i+1}]
```

where θ_i = 10000^(-2i/d)

### Key Insight

When computing attention:
```
R(q, pos_i)^T · R(k, pos_j) depends only on (i - j)
```

The relative position is naturally encoded in the dot product!

## Why RoPE is Better

| Feature | Absolute Position | RoPE |
|---------|------------------|------|
| Encoding | Additive | Multiplicative (rotation) |
| Relative position | Implicit | Explicit in attention |
| Extrapolation | Poor | Excellent |
| Parameters | Learned or fixed | None (formula-based) |

## Implementation

```bash
python rope.py
```

This implementation demonstrates:
- Rotation in 2D subspaces
- Relative position encoding
- Extrapolation capability
- Integration with attention

## Key Benefits

1. **Natural Relative Position**: The dot product of rotated vectors encodes relative position
2. **Extrapolation**: Works on sequences longer than training
3. **No Parameters**: Uses a fixed formula, no learning needed
4. **Efficient**: Simple to implement and fast to compute

## Used In

RoPE became the de facto standard:
- LLaMA (all versions)
- Mistral
- Qwen
- Yi
- Code Llama
- Falcon
- And nearly every modern open LLM

## Extensions

- **Extended Context**: YaRN, ALiBi-style interpolation
- **NTK-aware Scaling**: For very long contexts (100k+ tokens)
- **Dynamic Scaling**: Adjust base frequency during inference

## Why This Paper Matters

RoPE solved the long-context problem elegantly:
- Previous methods struggled beyond training length
- RoPE extrapolates naturally due to its mathematical structure
- Enabled the era of 100k+ context models
