# GShard: Scaling Giant Models with Conditional Computation

## Paper Summary

GShard pioneered scaling MoE models to 600B parameters with automatic sharding and load balancing techniques.

**Paper Link:** https://arxiv.org/abs/2006.16668

## Key Numbers

| Metric | Value |
|--------|-------|
| Total Parameters | 600B |
| Active per token | ~10B |
| Experts per layer | 2048 |
| Experts activated | 2/token |
| Training devices | 2048 TPU v3 |
| Training time | 4 days |

## Key Innovations

### 1. Automatic Sharding
Compiler-based tensor distribution:
```python
embedding = split(embedding_weights, dim=1)  # Partition
experts = split(expert_weights, dim=0)       # Distribute
```
No manual communication code needed!

### 2. Expert Capacity
Limit tokens per expert per batch:
```
capacity = (batch_size * seq_len * k) / num_experts * factor
```
Prevents overload, maintains balance.

### 3. Auxiliary Loss
Encourage uniform routing:
```
L_aux = α * sum(fraction_i * probability_i)
```

## Implementation

```bash
python gshard.py
```

## Architecture

Every other FFN layer becomes MoE:
```
Token → Router → Top-2 of 2048 experts → Output
```

Only 0.1% of experts activated per token!

## Results

- SOTA on 100+ language pairs
- 13.5 BLEU improvement on low-resource
- Near-human translation quality

## Why This Matters

1. **Scale**: First 600B model
2. **Automation**: Compiler-based sharding
3. **Practical**: Load balancing works
4. **Foundation**: Influenced all MoE work
