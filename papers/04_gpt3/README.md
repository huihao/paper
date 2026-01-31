# Language Models are Few-Shot Learners (GPT-3) (Brown et al., 2020)

## Paper Summary

GPT-3 demonstrated that very large language models can perform tasks with just a few examples in the prompt, without any gradient updates. This established in-context learning as a major paradigm.

**Paper Link:** https://arxiv.org/abs/2005.14165

## Key Contributions

### 1. In-Context Learning
The model learns from examples provided in the prompt:
- **Zero-shot**: Task description only
- **One-shot**: Task description + one example
- **Few-shot**: Task description + multiple examples

### 2. Scale
- 175 billion parameters
- Trained on 300 billion tokens
- 96 layers, 96 attention heads, 12288 embedding dimension

### 3. Emergent Capabilities
Larger models show abilities not present in smaller models:
- Arithmetic
- Word unscrambling
- Novel word usage

## Model Sizes

| Model | Parameters | Layers | d_model | Heads |
|-------|-----------|--------|---------|-------|
| Small | 125M | 12 | 768 | 12 |
| Medium | 350M | 24 | 1024 | 16 |
| Large | 760M | 24 | 1536 | 16 |
| XL | 1.3B | 24 | 2048 | 24 |
| 175B | 175B | 96 | 12288 | 96 |

## Implementation

```bash
python gpt3.py
```

This implementation demonstrates:
- In-context learning prompt construction
- Decoder-only transformer architecture
- Autoregressive generation with top-k sampling
- Scaling analysis

## Few-Shot Prompt Example

```
Translate English to French.

Input: Hello, how are you?
Output: Bonjour, comment allez-vous?

Input: Thank you very much.
Output: Merci beaucoup.

Input: The weather is beautiful today.
Output:
```

The model generates the translation by pattern matching from examples!

## Why This Paper Matters

1. **Paradigm Shift**: Showed that prompting could replace fine-tuning for many tasks
2. **Opened the Door**: Led directly to ChatGPT and the AI assistant era
3. **Scale as Strategy**: Demonstrated that bigger models have qualitatively new abilities
4. **General-Purpose AI**: One model serving many different tasks

## Training Details

- **Dataset**: 300B tokens from filtered Common Crawl, books, Wikipedia
- **Training**: ~3.14E23 FLOPs (estimated $4.6M in compute)
- **Context Length**: 2048 tokens
- **Architecture**: Decoder-only transformer with pre-layer normalization

## Impact on the Field

GPT-3 changed how we think about language models:
- From task-specific fine-tuning → prompt engineering
- From small, specialized models → large, general-purpose models
- From static evaluation → dynamic, context-dependent behavior
