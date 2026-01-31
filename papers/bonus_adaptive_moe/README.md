# Adaptive Mixtures of Local Experts (Jacobs et al., 1991)

## Paper Summary

The original Mixture of Experts paper that introduced the foundational concepts of expert networks, gating networks, and competitive learning.

**Paper Link:** https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf

## The Core Idea

Instead of one big network, use multiple "expert" networks, each specializing on different parts of the input space.

```
Input → [Expert 1] ─┐
      → [Expert 2] ─┼→ Gate → Weighted Output
      → [Expert 3] ─┘
```

## Key Components

### Expert Networks
Simple neural networks that specialize on subsets of data.

### Gating Network
Decides which expert to use for each input:
```
gate_probs = softmax(x @ W_gate + b_gate)
output = sum(gate_prob_i * expert_i(x))
```

## Implementation

```bash
python adaptive_moe.py
```

This implementation demonstrates:
- Expert networks (simple MLPs)
- Gating mechanism (softmax routing)
- Weighted expert combination
- Specialization behavior

## Why Specialization Works

1. **Divide and conquer**: Complex function → Simple subfunctions
2. **Competition**: Experts compete to explain each input
3. **Efficiency**: Each expert only needs to solve a subproblem
4. **Generalization**: Specialized experts generalize better

## Legacy

This 1991 paper introduced ideas still used in:
- **GShard** (2020): 600B parameters
- **Switch Transformer** (2021): Trillion parameters
- **Mixtral** (2024): Open-weight MoE
- **GPT-4** (rumored): MoE architecture

## Why This Matters

1. **Origin**: Where MoE began
2. **Core Ideas**: Still used 30+ years later
3. **Simplicity**: Elegant mathematical framework
4. **Foundation**: All modern MoE builds on this
