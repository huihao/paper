# PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)

## Paper Summary

A masterclass in large-scale training orchestration, training a 540B parameter model across 6144 TPU v4 chips.

**Paper Link:** https://arxiv.org/abs/2204.02311

## Model Configurations

| Model | Params | Layers | d_model | TPU Chips |
|-------|--------|--------|---------|-----------|
| PaLM 8B | 8B | 32 | 4096 | 256 |
| PaLM 62B | 62B | 64 | 8192 | 1024 |
| PaLM 540B | 540B | 118 | 18432 | 6144 |

## Architecture Innovations

### 1. Parallel Attention and FFN
```
Standard: x → Attention → Add → FFN → Add
PaLM:     x → Attention ─┐
          x → FFN ───────┼→ Add
```
15% faster training!

### 2. SwiGLU Activation
Better quality than ReLU/GELU

### 3. Multi-Query Attention
Shared key-value heads at 540B scale

### 4. RoPE Position Encoding
Good long-sequence extrapolation

## Implementation

```bash
python palm.py
```

This implementation demonstrates:
- Model configurations
- Parallelism strategies
- Training efficiency metrics
- Emergent abilities

## Training Scale

- **TPUs**: 6144 TPU v4 chips (2 pods)
- **Tokens**: 780 billion
- **MFU**: 46.2% (Model FLOPS Utilization)
- **Time**: ~2 months

## Pathways System

2D parallelism across 6144 chips:
- 12 model shards × 256 data replicas per pod
- Cross-pod training with efficient all-reduce

## Emergent Abilities

Abilities that only appear at 540B scale:
1. Joke explanation
2. Multi-step arithmetic with CoT
3. Complex code understanding
4. Strong multi-lingual capabilities

## Why This Paper Matters

1. **Scale Proof**: 540B models are trainable efficiently
2. **Infrastructure**: Pathways enabled unprecedented scale
3. **Architecture**: Parallel attention + SwiGLU became standard
4. **Emergent Abilities**: Documented new phenomena
5. **Systems Triumph**: As much about infrastructure as algorithms
