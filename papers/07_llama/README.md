# LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)

## Paper Summary

LLaMA triggered the open-weight era and introduced architectural defaults that became standard practice for modern LLMs.

**Paper Link:** https://arxiv.org/abs/2302.13971

## Key Architectural Innovations

### 1. RMSNorm (Root Mean Square Normalization)
```python
RMSNorm(x) = x / RMS(x) * γ
where RMS(x) = sqrt(mean(x²))
```
- Simpler than LayerNorm (no mean centering)
- Slightly more computationally efficient
- Works just as well in practice

### 2. SwiGLU Activation
```python
SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```
- Gated activation function
- Uses 3 weight matrices instead of 2
- Shown to improve performance in PaLM

### 3. Rotary Position Embeddings (RoPE)
- Encodes position through rotation of Q/K vectors
- Better extrapolation to longer sequences
- No separate position embedding needed

### 4. Pre-norm Architecture
- LayerNorm before each sub-layer (not after)
- More stable training at scale

## Model Sizes

| Model | Params | Layers | Hidden | Heads | FFN |
|-------|--------|--------|--------|-------|-----|
| 7B | 6.7B | 32 | 4096 | 32 | 11008 |
| 13B | 13B | 40 | 5120 | 40 | 13824 |
| 33B | 33B | 60 | 6656 | 52 | 17920 |
| 65B | 65B | 80 | 8192 | 64 | 22016 |

## Implementation

```bash
python llama.py
```

This implementation demonstrates:
- RMSNorm normalization
- SwiGLU activation function
- Rotary Position Embeddings
- Complete LLaMA block

## Training Details

- **Tokens**: 1-1.4 trillion tokens
- **Data**: Publicly available data only
- **Context**: 2048 tokens
- **Tokenizer**: SentencePiece (32k vocab)

## Why This Paper Matters

### 1. Triggered Open-Weight Era
Before LLaMA, most capable LLMs were closed. LLaMA showed open models can compete.

### 2. Set Architectural Standards
- RMSNorm, SwiGLU, RoPE became the default
- Nearly all subsequent open models follow this template

### 3. Enabled Research Community
- Fine-tuning research exploded (Alpaca, Vicuna, etc.)
- Spawned an entire ecosystem of derivative models

### 4. Demonstrated Efficiency
- 7B model competitive with much larger closed models
- Proved quality of training matters as much as scale

## Comparison with GPT-3

| Aspect | GPT-3 | LLaMA |
|--------|-------|-------|
| Normalization | LayerNorm | RMSNorm |
| Position Encoding | Learned | RoPE |
| FFN Activation | GELU | SwiGLU |
| Architecture | Pre-norm | Pre-norm |
| Weights | Closed | Open |

## Impact on the Field

LLaMA's release was a pivotal moment:
1. Proved open-weight models could match proprietary ones
2. Architectural choices (RMSNorm, SwiGLU, RoPE) became industry standard
3. Enabled the fine-tuning revolution (LoRA, QLoRA, etc.)
4. Set the template for LLaMA 2, Mistral, Qwen, and others
