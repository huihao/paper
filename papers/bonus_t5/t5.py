#!/usr/bin/env python3
"""
T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
(Raffel et al., 2019)

Unified text-to-text framework for all NLP tasks.

Paper: https://arxiv.org/abs/1910.10683
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class T5Config:
    """T5 model configurations"""
    name: str
    params: str
    layers: int
    d_model: int
    d_ff: int
    heads: int


T5_CONFIGS = {
    'small': T5Config('T5-Small', '60M', 6, 512, 2048, 8),
    'base': T5Config('T5-Base', '220M', 12, 768, 3072, 12),
    'large': T5Config('T5-Large', '770M', 24, 1024, 4096, 16),
    '3B': T5Config('T5-3B', '3B', 24, 1024, 16384, 32),
    '11B': T5Config('T5-11B', '11B', 24, 1024, 65536, 128),
}


class T5Framework:
    """
    T5: Text-to-Text Transfer Transformer
    
    Key insight: ALL NLP tasks can be framed as text-to-text problems.
    
    Input: "translate English to German: That is good"
    Output: "Das ist gut"
    
    Input: "summarize: <article text>"
    Output: "<summary>"
    
    Input: "classify: <text>"
    Output: "positive" / "negative"
    """
    
    @staticmethod
    def describe_framework():
        return """
T5: Unified Text-to-Text Framework

Every NLP task becomes:
    Input Text → Model → Output Text

Examples:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Task         │ Input                        │ Output           │
    ├─────────────────────────────────────────────────────────────────┤
    │ Translation  │ translate English to German: │ Das ist gut      │
    │              │ That is good                 │                  │
    ├─────────────────────────────────────────────────────────────────┤
    │ Summarize    │ summarize: The article says..│ Brief summary    │
    ├─────────────────────────────────────────────────────────────────┤
    │ Classification│ sst2 sentence: great movie  │ positive         │
    ├─────────────────────────────────────────────────────────────────┤
    │ Q&A          │ question: What is the        │ Paris            │
    │              │ capital? context: France...  │                  │
    └─────────────────────────────────────────────────────────────────┘

Benefits:
    1. Same model for all tasks
    2. Same loss function (cross-entropy on text)
    3. Same training procedure
    4. Transfer learning across tasks
"""

    @staticmethod
    def describe_architecture():
        return """
T5 Architecture:

Encoder-Decoder Transformer:
    Input: Task prefix + Input text
    Encoder: Bidirectional attention over input
    Decoder: Autoregressive generation of output

    ┌───────────────────────────────────────────┐
    │ Input: "translate English to German: cat" │
    └─────────────────┬─────────────────────────┘
                      ↓
    ┌───────────────────────────────────────────┐
    │         Encoder (Bidirectional)           │
    │  [translate] [English] [to] [German] [cat]│
    └─────────────────┬─────────────────────────┘
                      ↓
    ┌───────────────────────────────────────────┐
    │         Decoder (Autoregressive)          │
    │  [<start>] → [Katze] → [<end>]           │
    └───────────────────────────────────────────┘
                      ↓
    ┌───────────────────────────────────────────┐
    │ Output: "Katze"                           │
    └───────────────────────────────────────────┘

Key Modifications:
    1. Relative position embeddings
    2. Pre-norm (LayerNorm before attention)
    3. No bias terms (except LayerNorm)
"""


def print_ablation_findings():
    """Print key findings from T5 ablation study"""
    print("\n📊 T5 Ablation Study Findings")
    print("=" * 70)
    
    print("""
The T5 paper is famous for its systematic ablation study.

Key Findings:

1. Architecture:
   - Encoder-decoder slightly better than decoder-only for many tasks
   - Shared embeddings between encoder/decoder helps
   
2. Pre-training Objective:
   - Span corruption (like BERT) > Language modeling
   - 15% corruption rate optimal
   - Average span length of 3 tokens
   
3. Data:
   - C4 (Colossal Clean Crawled Corpus) created
   - Web text with quality filtering
   - English only, deduplicated
   
4. Training:
   - Multi-task mixing: Equal examples per task (not proportional)
   - Temperature-based task mixing helps
   - More training always helps (but diminishing returns)
   
5. Scale:
   - Bigger models better across all tasks
   - 11B model state-of-the-art at time

C4 Dataset:
    ┌────────────────────────────────────────┐
    │ Size: 750GB of cleaned text           │
    │ Source: Common Crawl                  │
    │ Filtering:                            │
    │   - English only                      │
    │   - Removed short pages               │
    │   - Removed boilerplate               │
    │   - Deduplicated                      │
    │   - Removed offensive content         │
    └────────────────────────────────────────┘
""")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("T5: Exploring the Limits of Transfer Learning")
    print("(Raffel et al., 2019)")
    print("=" * 70)
    
    print(T5Framework.describe_framework())
    print(T5Framework.describe_architecture())
    
    # Print configurations
    print("\n📊 T5 Model Configurations")
    print("-" * 70)
    print(f"{'Model':<12} {'Params':<8} {'Layers':<8} {'d_model':<10} {'Heads':<8}")
    print("-" * 70)
    for name, config in T5_CONFIGS.items():
        print(f"{config.name:<12} {config.params:<8} {config.layers:<8} "
              f"{config.d_model:<10} {config.heads:<8}")
    
    print_ablation_findings()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Unified Framework: All NLP as text-to-text
    
    2. Systematic Study: Rigorous ablations
    
    3. C4 Dataset: Foundation for many models
    
    4. Transfer Learning: Proved multi-task helps
    
    5. Encoder-Decoder: Showed architecture value
    
    Key Insight:
        Framing all NLP as text-to-text simplifies
        everything: one model, one loss, one pipeline.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
