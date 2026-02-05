# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

## Paper Summary

This paper ignited modern MoE research by showing that conditional computation at scale actually works.

**Paper Link:** https://arxiv.org/abs/1701.06538

## The Key Insight

Instead of using all parameters for every input, only activate a subset of "experts" per token.

```
Dense: Every token → All parameters
MoE: Every token → Only top-k experts
```

## How It Works

### 1. Gating Network
```python
scores = softmax(x @ W_gate + noise)
top_k = select_highest(scores, k)
```

### 2. Expert Selection
- Only k experts are activated per token
- Gates are renormalized among selected experts

### 3. Weighted Combination
```python
output = sum(gate[i] * expert[i](x) for i in top_k)
```

## Implementation

```bash
python sparse_moe.py
```

This implementation demonstrates:
- Noisy top-k gating
- Expert selection and routing
- Load balancing loss
- Parameter efficiency calculation

## Why Add Noise?

During training, noise encourages exploration:
- Discovers which experts work best for which inputs
- Prevents early expert collapse
- Helps with load balancing

## Load Balancing

Without balance: One expert dominates, others wasted
With balance: Experts share load, capacity utilized

Loss = CV(importance)² + CV(load)²

## Dense vs MoE Comparison

| Aspect | Dense (1T) | MoE (1T total, top-4) |
|--------|------------|----------------------|
| Total params | 1T | 1T |
| Active params | 1T | ~60B |
| FLOPs/token | 6T | 360B |
| Compute savings | 1x | 16x |

## Why This Paper Matters

1. **Introduced sparsity at scale**
2. **Proved conditional computation works**
3. **Enabled trillion-parameter models**
4. **Foundation for all modern MoE work**

Key insight: Capacity ≠ Compute
