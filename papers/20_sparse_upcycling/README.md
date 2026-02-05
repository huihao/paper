# Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints

## Paper Summary

Sparse upcycling is a practical technique for converting trained dense models into MoE models, enabling compute reuse and faster convergence.

**Paper Link:** https://arxiv.org/abs/2212.05055

## The Key Idea

Start with a trained dense model, replicate its weights to create experts, and continue training.

```
Dense FFN (trained) → Copy to N experts → Add noise → Continue training
```

## Implementation

```bash
python sparse_upcycling.py
```

This implementation demonstrates:
- Weight replication to experts
- Noise injection for differentiation
- Router initialization
- Parameter impact analysis

## The Upcycling Process

### Step 1: Start with Dense Model
```
Dense FFN: up_proj, down_proj
(Already trained, contains useful features)
```

### Step 2: Replicate to Experts
```
Expert 0 = copy(dense) + noise
Expert 1 = copy(dense) + noise
...
Expert N = copy(dense) + noise
```

### Step 3: Add Router
```
Router = random_init()
(Will learn which inputs → which experts)
```

### Step 4: Continue Training
- Experts specialize from same starting point
- Router learns optimal routing
- Much faster than from-scratch MoE

## Benefits

| Aspect | From Scratch | Upcycling |
|--------|-------------|-----------|
| Training time | Full | ~50% less |
| Stability | Challenging | Easier |
| Compute reuse | None | Yes |
| Expert collapse | Common | Rare |

## Practical Applications

1. **Compute Reuse**: Don't waste dense training
2. **Iterative Scaling**: Dense → MoE → larger MoE
3. **Risk Reduction**: Start from working model
4. **Experimentation**: Try different configs quickly

## Used In Practice

- DeepSeek-MoE
- Qwen-MoE
- Many research models

## Why This Paper Matters

1. **Actually Practical**: Used in production
2. **Efficient**: Reuse training investment
3. **Lower Risk**: Iterate from working models
4. **Foundation**: Influenced modern MoE work
