# The Smol Training Playbook (Hugging Face, 2025)

## Overview

A practical end-to-end handbook for efficiently training language models from 1B to 7B parameters.

**Reference:** https://huggingface.co/spaces/HuggingFaceSmol/smol-training-playbook

## Key Principles

1. **Start Small, Scale Up** - Validate before scaling
2. **Data Quality > Quantity** - Clean, curated data
3. **Infrastructure Efficiency** - Maximize utilization
4. **Reproducibility** - Version control everything

## Sample Training Recipes

| Model | Batch Size | LR | Tokens | Hardware | Time |
|-------|-----------|-----|--------|----------|------|
| 1B | 2M | 3e-4 | 300B | 8x A100 | ~3 days |
| 3B | 2M | 3e-4 | 500B | 16x A100 | ~1 week |
| 7B | 4M | 3e-4 | 1T | 32x A100 | ~2 weeks |

## Modern Architecture Defaults

| Component | Default |
|-----------|---------|
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Position | RoPE |
| Attention | GQA or MQA |

## Implementation

```bash
python playbook.py
```

This implementation demonstrates:
- Data preparation checklist
- Architecture choices
- Hyperparameter selection
- Training infrastructure
- Monitoring and debugging
- Evaluation guide

## Data Preparation Checklist

- [ ] Collect and curate sources
- [ ] Clean and filter (dedupe, quality filter)
- [ ] Format and tokenize
- [ ] Create domain mixtures
- [ ] Validate and decontaminate

## Training Hyperparameters

```python
# Standard 7B configuration
config = {
    'peak_lr': 3e-4,
    'warmup_steps': 2000,
    'batch_size': 4_000_000,  # tokens
    'total_tokens': '1-2T',
    'optimizer': 'AdamW',
    'betas': (0.9, 0.95),
    'weight_decay': 0.1,
    'gradient_clip': 1.0,
}
```

## Why This Matters

1. **Practical**: End-to-end training guide
2. **Modern**: Up-to-date best practices
3. **Efficient**: Maximize hardware usage
4. **Accessible**: Democratize LLM training
