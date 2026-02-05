# BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)

## Paper Summary

BERT (Bidirectional Encoder Representations from Transformers) introduced a new paradigm for NLP pre-training using bidirectional context.

**Paper Link:** https://arxiv.org/abs/1810.04805

## Key Contributions

### 1. Bidirectional Pre-training
- Unlike GPT (left-to-right) or ELMo (concatenated left-to-right and right-to-left)
- BERT uses true bidirectional context through masking

### 2. Pre-training Objectives

#### Masked Language Modeling (MLM)
- Randomly mask 15% of input tokens
- Of masked tokens:
  - 80% → [MASK] token
  - 10% → random token
  - 10% → unchanged
- Predict the original tokens

#### Next Sentence Prediction (NSP)
- Given sentence A and B
- Predict if B follows A (50/50 split)
- Helps with sentence-level understanding

### 3. Input Representation
```
[CLS] Sentence A [SEP] Sentence B [SEP]
  ↓       ↓        ↓       ↓        ↓
Token Embedding + Segment Embedding + Position Embedding
```

## Architecture

- **BERT-Base**: 12 layers, 768 hidden, 12 heads, 110M parameters
- **BERT-Large**: 24 layers, 1024 hidden, 16 heads, 340M parameters

## Implementation

```bash
python bert.py
```

This implementation demonstrates:
- Token, segment, and position embeddings
- Masked Language Modeling with proper masking strategy
- Next Sentence Prediction task
- Pre-training loss computation

## Code Example

```python
from bert import BertModel

# Create BERT model
model = BertModel(
    vocab_size=30522,
    hidden_size=768,
    num_layers=12,
    num_heads=12
)

# Pre-training step
losses = model.pretrain_step(
    token_ids=np.array([101, 50, 23, 103, 89, 102, 45, 78, 102]),
    segment_ids=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
    is_next=True
)

print(f"MLM Loss: {losses['mlm_loss']}")
print(f"NSP Loss: {losses['nsp_loss']}")
```

## Impact on Modern LLMs

While modern LLMs (GPT, LLaMA) are decoder-only:
- BERT's masked language modeling influenced later work
- Representation learning concepts remain relevant
- Fine-tuning paradigm became standard practice
- Encoder architectures still used for embeddings (e.g., sentence-transformers)

## Why This Paper Matters

- Demonstrated the power of pre-training + fine-tuning
- Showed bidirectional context improves understanding
- Set numerous benchmarks (GLUE, SQuAD)
- Influenced all subsequent NLP pre-training work
