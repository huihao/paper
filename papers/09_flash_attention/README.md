# FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)

## Paper Summary

FlashAttention is an IO-aware attention algorithm that never materializes the full N×N attention matrix, enabling much longer context windows and faster training.

**Paper Link:** https://arxiv.org/abs/2205.14135

## The Problem

Standard attention requires O(N²) memory for the attention matrix:
- 4096 tokens → 64 MB just for attention weights
- 16384 tokens → 1 GB just for attention weights
- 100k tokens → impossible with standard approach

## The Solution: Tiling + Online Softmax

### Key Ideas

1. **Block-wise Computation**: Process Q, K, V in blocks that fit in GPU SRAM
2. **Online Softmax**: Combine block results without storing full matrix
3. **IO-Aware**: Minimize HBM (slow) reads/writes

### Algorithm Sketch

```
for each Q block:
    initialize: m = -inf, l = 0, o = 0
    for each K, V block:
        compute block attention scores
        update running softmax (m, l)
        accumulate output (o)
    normalize output
```

## Memory Comparison

| Sequence Length | Standard | FlashAttention | Savings |
|----------------|----------|----------------|---------|
| 1024 | 4 MB | 0.03 MB | 133x |
| 4096 | 64 MB | 0.13 MB | 492x |
| 16384 | 1 GB | 0.5 MB | 2000x |

## Implementation

```bash
python flash_attention.py
```

This implementation demonstrates:
- Standard vs FlashAttention comparison
- Memory usage analysis
- Online softmax algorithm
- IO complexity explanation

## FlashAttention-2 Improvements

1. **Better Parallelization**: Parallelize over sequence, not just batch/heads
2. **Reduced Non-MatMul FLOPs**: Fewer scalar operations
3. **Better Memory Access**: More reuse of K, V in shared memory

Result: 2x faster than FlashAttention-1

## Impact

FlashAttention changed what was possible:

| Before FlashAttention | After FlashAttention |
|----------------------|---------------------|
| 2-4k context typical | 100k+ context possible |
| Long context = slow | Long context = efficient |
| O(N²) memory | O(N) memory |
| Sparse attention compromises | Exact attention |

## Why This Paper Matters

1. **Enabled Long Context**: GPT-4 (128k), Claude (100k+) would be impractical without it
2. **No Quality Trade-off**: Unlike sparse attention, gives exact same output
3. **Widely Adopted**: Now in PyTorch, Transformers, JAX, and all major frameworks
4. **Changed Thinking**: Showed algorithmic innovation can beat hardware brute force
