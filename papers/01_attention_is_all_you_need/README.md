# Attention Is All You Need (Vaswani et al., 2017)

## Paper Summary

The original Transformer paper that revolutionized NLP and became the foundation for all modern large language models.

**Paper Link:** https://arxiv.org/abs/1706.03762

## Key Contributions

1. **Self-Attention Mechanism**: Allows the model to attend to all positions in the input sequence simultaneously
2. **Multi-Head Attention**: Enables the model to jointly attend to information from different representation subspaces
3. **Positional Encoding**: Uses sinusoidal functions to inject sequence order information
4. **Encoder-Decoder Architecture**: While most modern LLMs are decoder-only, the original paper proposed the full encoder-decoder structure

## Architecture Components

```
Input → Embedding + Positional Encoding
         ↓
    [Encoder Layers × N]
         ↓
    Encoder Output
         ↓
    [Decoder Layers × N]
         ↓
    Linear → Softmax → Output
```

### Encoder Layer
- Multi-Head Self-Attention
- Add & Norm
- Feed-Forward Network
- Add & Norm

### Decoder Layer
- Masked Multi-Head Self-Attention
- Add & Norm
- Multi-Head Cross-Attention (with encoder output)
- Add & Norm
- Feed-Forward Network
- Add & Norm

## Key Equations

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

### Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Implementation

This folder contains a pure NumPy implementation of the Transformer architecture:

```bash
python transformer.py
```

## Usage Example

```python
from transformer import Transformer, scaled_dot_product_attention

# Create transformer model
model = Transformer(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048
)

# Forward pass
logits = model(src_tokens, tgt_tokens)
```

## Why This Paper Matters

- Introduced the architecture that powers GPT, BERT, LLaMA, and all modern LLMs
- Showed that attention alone (without recurrence or convolution) is sufficient for sequence modeling
- Enabled parallel training, making it much faster than RNNs
- The "Attention Is All You Need" phrase became iconic in the field
