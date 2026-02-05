# Scaling Laws for Neural Language Models (Kaplan et al., 2020)

## Paper Summary

This paper established the first clean empirical framework for understanding how language model performance scales with parameters, data, and compute.

**Paper Link:** https://arxiv.org/abs/2001.08361

## Key Findings

### Power Law Relationships

The paper found that test loss follows smooth power laws:

```
L(N) = (N_c / N)^α_N   where α_N ≈ 0.076
L(D) = (D_c / D)^α_D   where α_D ≈ 0.095
L(C) = (C_c / C)^α_C   where α_C ≈ 0.050
```

Where:
- N = number of parameters
- D = number of training tokens
- C = compute budget (in PetaFLOP-days)

### Optimal Compute Allocation

For a fixed compute budget C:
- Optimal parameters: N ∝ C^0.73
- Optimal data: D ∝ C^0.27

**Key insight**: Most compute should go to making models bigger, not training longer.

### Sample Efficiency

Larger models are more sample efficient:
- A 10x larger model achieves the same loss with ~3x less data
- This suggests scaling parameters is more efficient than scaling data

## Implementation

```bash
python scaling_laws.py
```

This implementation demonstrates:
- Loss prediction from scaling laws
- Optimal compute allocation
- Sample efficiency calculations
- Experiment simulation

## Key Equations

### Combined Loss
```python
L(N, D) ≈ [(N_c/N)^(α_N/α_D) + D_c/D]^α_D
```

### Compute for Training
```python
C ≈ 6ND  # FLOPs for training
```

## Practical Implications

1. **Predictable Scaling**: You can extrapolate final loss from smaller runs
2. **Bigger is Better**: With more compute, make the model bigger
3. **Diminishing Returns**: More data alone has limited benefit
4. **No Phase Transitions**: Performance improves smoothly

## Important Caveat

⚠️ **Chinchilla Revision (2022)**: Later work showed these allocations were suboptimal:
- Kaplan suggested N ∝ C^0.73, D ∝ C^0.27
- Chinchilla showed closer to N ∝ C^0.5, D ∝ C^0.5
- Most models trained before Chinchilla were undertrained

## Why This Paper Matters

- First rigorous scaling framework for LLMs
- Enabled compute-optimal training decisions
- Showed that scaling is remarkably predictable
- Laid foundation for understanding emergent capabilities
