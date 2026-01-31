# Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet

## Paper Summary

The biggest leap in mechanistic interpretability - using sparse autoencoders to decompose neural networks into millions of interpretable features.

**Paper Link:** https://www.anthropic.com/research/scaling-monosemanticity

## The Problem: Polysemanticity

Individual neurons respond to many unrelated things:
```
Neuron 4792 activates for:
- The color "blue"
- The number "7"  
- References to sadness
- Some French words
```

This makes interpretation nearly impossible!

## The Solution: Sparse Autoencoders

Train a sparse autoencoder on model activations:
1. **Sparse**: Most features inactive at any time
2. **Interpretable**: One meaning per feature
3. **Comprehensive**: Explain all model behavior

## Implementation

```bash
python monosemanticity.py
```

This implementation demonstrates:
- Sparse autoencoder architecture
- Feature extraction
- Feature statistics
- Sparsity control

## Example Features Discovered

| Feature | Description | Frequency |
|---------|-------------|-----------|
| 12847 | Golden Gate Bridge | 0.01% |
| 23456 | Recursive code patterns | 0.12% |
| 45678 | Sycophantic agreement | 0.34% |
| 67890 | Safety/refusal concepts | 0.05% |

Each feature is:
- Monosemantic (one meaning)
- Interpretable (human-understandable)
- Steerable (can amplify or suppress)

## Applications

1. **Understanding**: See which features activate
2. **Steering**: Amplify or suppress features
3. **Safety**: Monitor harmful behavior features
4. **Red Teaming**: Find adversarial patterns

## Scale Achieved

- Model: Claude 3 Sonnet (billions of parameters)
- Features extracted: **34 million**
- All interpretable and steerable

## Why This Paper Matters

1. **Breakthrough**: First large-scale interpretability success
2. **Safety**: Path to understanding AI behavior
3. **Science**: What do neural networks actually learn?
4. **Future**: Foundation for interpretable AI
