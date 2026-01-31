# The Illustrated Transformer (Jay Alammar, 2018)

## Overview

This is not a research paper but a highly influential blog post that provides visual, intuitive explanations of the Transformer architecture.

**Blog Post:** https://jalammar.github.io/illustrated-transformer/

## Why This Matters

Jay Alammar's blog post has become the de facto starting point for anyone learning about Transformers. It breaks down complex concepts into digestible visual explanations.

## Key Concepts Illustrated

### 1. Embedding + Positional Encoding
- Words → Dense vectors
- Position information added via sine/cosine functions

### 2. Self-Attention Visualization
```
Query × Key^T → Scores → Softmax → Weights × Value → Output
```

Each word can "look at" all other words and decide which are most relevant.

### 3. Multi-Head Attention
Multiple attention "heads" allow the model to:
- Focus on different aspects simultaneously
- Capture various types of relationships (syntactic, semantic, positional)

### 4. Add & Normalize
- Residual connections help gradient flow
- Layer normalization stabilizes training

### 5. Feed-Forward Network
- Applied position-wise (same network to each position)
- Expands and compresses representations

## Implementation

This folder contains an educational implementation with verbose output:

```bash
python transformer_visual.py
```

The code prints step-by-step explanations of:
- Tensor shapes at each stage
- Attention weight computations
- How information flows through the network

## Learning Approach

This implementation prioritizes understanding over efficiency:
- Detailed print statements at each step
- Small dimensions for easy mental math
- Comments explaining each operation

## Example Output

```
📐 Step 1: Compute Q @ K^T (dot product similarity)
   Score matrix shape: (3, 3)
   Raw scores:
   [[1.0   0.0   0.0  ]
    [0.0   1.0   0.0  ]
    [0.0   0.0   1.25 ]]

📏 Step 2: Scale by √d_k = √4 ≈ 2.000
   Scaled scores: ...

🎯 Step 3: Apply softmax to get attention weights
   ...
```

## Use This For

- Learning how Transformers work
- Teaching others about attention mechanisms
- Debugging by understanding intermediate values
- Building intuition before diving into optimized implementations
