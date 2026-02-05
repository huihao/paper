# T5: Text-to-Text Transfer Transformer (Raffel et al., 2019)

## Paper Summary

T5 introduced a unified text-to-text framework where every NLP task is framed as converting input text to output text.

**Paper Link:** https://arxiv.org/abs/1910.10683

## The Text-to-Text Framework

Every task becomes: Input Text → Model → Output Text

| Task | Input | Output |
|------|-------|--------|
| Translation | translate English to German: Hello | Hallo |
| Summarization | summarize: Long article... | Brief summary |
| Classification | sst2 sentence: Great movie | positive |
| Q&A | question: Capital? context: France... | Paris |

## Model Configurations

| Model | Params | Layers | d_model |
|-------|--------|--------|---------|
| T5-Small | 60M | 6 | 512 |
| T5-Base | 220M | 12 | 768 |
| T5-Large | 770M | 24 | 1024 |
| T5-3B | 3B | 24 | 1024 |
| T5-11B | 11B | 24 | 1024 |

## Implementation

```bash
python t5.py
```

## Key Ablation Findings

1. **Architecture**: Encoder-decoder slightly better than decoder-only
2. **Pre-training**: Span corruption > Language modeling
3. **Data**: C4 dataset (750GB cleaned web text)
4. **Scale**: Bigger models better across all tasks

## C4 Dataset

- 750GB of cleaned English text
- From Common Crawl
- Deduplicated and filtered

## Why This Matters

1. **Unified Framework**: One approach for all NLP
2. **Systematic Study**: Rigorous ablations
3. **C4 Dataset**: Foundation for many models
4. **Transfer Learning**: Multi-task benefits
