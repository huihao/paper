# Hierarchical Mixtures of Experts (Jordan & Jacobs, 1994)

## Paper Summary

Extended the original MoE with a tree-structured gating network for multi-level specialization.

**Paper Link:** https://www.cs.toronto.edu/~hinton/absps/hme.pdf

## The Hierarchy

```
                 [Root Gate]
                /           \
          [Gate L]       [Gate R]
          /     \         /     \
       [E1]   [E2]     [E3]   [E4]
```

Path probabilities:
- P(E1) = P(L) × P(LL|L)
- P(E2) = P(L) × P(LR|L)
- etc.

## Why Hierarchy?

1. **Coarse-to-fine**: Progressive decisions
2. **Scalability**: 2^depth experts
3. **Interpretability**: Clear decision paths
4. **Efficiency**: Prune unlikely branches

## Implementation

```bash
python hierarchical_moe.py
```

This implementation demonstrates:
- Tree-structured gating
- Path probability computation
- Expert leaf networks
- Hierarchical routing

## Example: Image Classification

```
Root: Animal or Object?
├── Animal
│   ├── Mammal
│   │   ├── Cat Expert
│   │   └── Dog Expert
│   └── Bird Expert
└── Object
    ├── Vehicle Expert
    └── Furniture Expert
```

Each expert specializes on a narrow category!

## Comparison with Flat MoE

| Aspect | Flat MoE | Hierarchical |
|--------|----------|--------------|
| Decision | Single gate | Multi-level |
| Experts | N gates | log(N) depth |
| Inference | All gates | Prune paths |
| Interpretability | Moderate | High |

## Why This Matters

1. **Multi-level**: Hierarchical decisions
2. **Scalable**: Logarithmic gating depth
3. **Interpretable**: Decision paths visible
4. **Foundation**: Neural + tree hybrids
