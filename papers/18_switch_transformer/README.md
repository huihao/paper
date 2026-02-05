# Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2021)

## Paper Summary

Switch Transformers simplified MoE routing by using single-expert activation (k=1), enabling stable training of trillion-parameter models.

**Paper Link:** https://arxiv.org/abs/2101.03961

## Key Simplification

```
Original MoE: Token → Top-k experts (k=2,4)
Switch:       Token → Single expert (k=1)
```

This simple change enabled 1.6T parameter models!

## Implementation

```bash
python switch_transformer.py
```

This implementation demonstrates:
- k=1 routing mechanism
- Capacity factor handling
- Load balancing
- Token dropping

## Switch Routing

```python
logits = x @ W_gate
probs = softmax(logits)
expert_idx = argmax(probs)
output = gate_prob * expert(x)
```

Each token routed to exactly ONE expert.

## Load Balancing Loss

```
L_aux = α × Σᵢ (fᵢ × Pᵢ)
```
- fᵢ = fraction of tokens to expert i
- Pᵢ = average routing probability to expert i

## Capacity Factor

```
capacity = (tokens / num_experts) × capacity_factor
```
- 1.0: Balanced (some drops)
- 1.25: 25% overflow allowed
- 2.0: Generous buffer

Excess tokens are dropped!

## Results

| Model | Parameters | Quality | Speed |
|-------|-----------|---------|-------|
| T5-Base | 223M | Baseline | 1.0x |
| Switch-Base-8 | 1.4B | +4.4% | 1.3x |
| Switch-Base-64 | 7.4B | +6.7% | 1.4x |

## Why This Matters

1. **Simplicity**: k=1 works!
2. **Trillion Scale**: First public 1.6T model
3. **Stability**: Key training insights
4. **Foundation**: Influenced modern MoE
