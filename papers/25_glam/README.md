# GLaM: Generalist Language Model (Du et al., 2022)

## Paper Summary

GLaM validated MoE (Mixture of Experts) scaling economics, demonstrating that massive total parameters with small active parameter counts can match or exceed dense model quality at lower cost.

**Paper Link:** https://arxiv.org/abs/2112.06905

## Key Numbers

| Metric | GPT-3 | GLaM 1.7T |
|--------|-------|-----------|
| Total Parameters | 175B | 1,200B |
| Active Parameters | 175B | 96.6B |
| Training FLOPs | 3.14e23 | 1.76e23 |
| Training Energy | 1,287 MWh | 456 MWh |
| Zero-shot Avg | 55.4% | 55.0% |
| One-shot Avg | 56.5% | 57.0% |

**GLaM matches GPT-3 quality with 1/3 the energy!**

## Architecture

MoE Layer Structure:
```
Token → Router → Select 2 of 64 experts → Weighted sum → Output
```

- **64 experts** per layer
- **2 experts** activated per token
- **1.2T total** parameters
- **96.6B active** parameters

## Implementation

```bash
python glam.py
```

This implementation demonstrates:
- MoE architecture concepts
- Routing mechanism
- Load balancing
- Efficiency calculations

## Efficiency Gain

```
Capacity/Compute Ratio: 12.4x
- 12x more parameters for knowledge storage
- Same compute cost as smaller model
```

## Expert Routing

1. Compute router logits for each expert
2. Select top-2 experts
3. Apply softmax to get weights
4. Combine expert outputs

Load balancing auxiliary loss prevents expert collapse.

## Why This Paper Matters

1. **Economics**: Proved MoE scaling works at scale
2. **Efficiency**: 3x energy savings over dense
3. **Quality**: Matched GPT-3 with fewer FLOPs
4. **Scale**: Demonstrated 1.2T trainable
5. **Foundation**: Enabled Mixtral, etc.
