#!/usr/bin/env python3
"""
PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)
A masterclass in large-scale training orchestration across thousands of accelerators.

Paper: https://arxiv.org/abs/2204.02311
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PaLMConfig:
    """PaLM model configurations"""
    name: str
    params: float  # in billions
    layers: int
    heads: int
    d_model: int
    d_ff: int
    tpu_chips: int


PALM_CONFIGS = {
    '8B': PaLMConfig('PaLM 8B', 8, 32, 16, 4096, 16384, 256),
    '62B': PaLMConfig('PaLM 62B', 62, 64, 32, 8192, 32768, 1024),
    '540B': PaLMConfig('PaLM 540B', 540, 118, 48, 18432, 73728, 6144),
}


class PathwaysSystem:
    """
    The Pathways System for Large-Scale Training
    
    Key innovations:
    1. Efficient 2D parallelism (data + model)
    2. Cross-pod training (multiple TPU pods)
    3. Efficient all-reduce across heterogeneous networks
    4. Pipeline-free parallelism for better utilization
    """
    
    @staticmethod
    def describe_parallelism():
        return """
Pathways Parallelism Strategy:

Data Parallelism:
    - Replicate model across devices
    - Each device processes different data
    - Sync gradients after each step
    
Model Parallelism:
    - Split model layers across devices
    - Each device holds part of the model
    - Activations passed between devices
    
PaLM 540B Configuration:
    ┌─────────────────────────────────────────────────┐
    │ Total TPU v4 Chips: 6144                        │
    │ TPU Pods: 2 (3072 chips each)                   │
    │ Model Shards: 12 per replica                    │
    │ Data Parallel Replicas: 256                     │
    │ Training Efficiency: 46.2% MFU                  │
    └─────────────────────────────────────────────────┘

2D Parallelism Layout:
    Device Layout: 12 × 256 = 3072 chips per pod
    
    ┌────┬────┬────┬────┬────┬─...─┬────┐ ← Data parallel (256)
    │ L0 │ L0 │ L0 │ L0 │ L0 │     │ L0 │
    ├────┼────┼────┼────┼────┼─...─┼────┤
    │ L1 │ L1 │ L1 │ L1 │ L1 │     │ L1 │
    ├────┼────┼────┼────┼────┼─...─┼────┤
    │ ...│ ...│ ...│ ...│ ...│     │ ...│
    ├────┼────┼────┼────┼────┼─...─┼────┤
    │L11 │L11 │L11 │L11 │L11 │     │L11 │
    └────┴────┴────┴────┴────┴─...─┴────┘
      ↑
      Model parallel (12 shards)
"""

    @staticmethod
    def compute_efficiency_metrics(config: PaLMConfig) -> Dict:
        """Compute training efficiency metrics"""
        # TPU v4 peak FLOPS per chip
        peak_flops_per_chip = 275e12  # 275 TFLOPS bfloat16
        
        # Total theoretical peak
        total_peak_flops = peak_flops_per_chip * config.tpu_chips
        
        # Observed MFU (Model FLOPS Utilization) from paper
        mfu = 0.462 if config.params == 540 else 0.50
        
        # Actual achieved FLOPS
        achieved_flops = total_peak_flops * mfu
        
        return {
            'peak_pflops': total_peak_flops / 1e15,
            'achieved_pflops': achieved_flops / 1e15,
            'mfu': mfu,
            'chips': config.tpu_chips
        }


class PaLMArchitecture:
    """
    PaLM Architecture Details
    
    Key choices:
    1. Decoder-only transformer
    2. SwiGLU activation
    3. Parallel attention + FFN
    4. Multi-query attention (for 540B)
    5. RoPE position encoding
    6. No bias terms
    """
    
    @staticmethod
    def describe_architecture():
        return """
PaLM Architecture Innovations:

1. Parallel Attention and FFN:
   Standard Transformer:
       x → Attention → Add → FFN → Add
   
   PaLM:
       x → Attention ─┐
       x → FFN ───────┼→ Add
                     ↓
                  output
   
   15% faster training!

2. SwiGLU Activation:
   FFN(x) = SiLU(x·W_gate) ⊙ (x·W_up) · W_down
   
   Better quality than ReLU or GELU

3. Multi-Query Attention (540B):
   - Shared key-value heads across attention heads
   - Reduces memory for long sequences
   - Slight quality trade-off worth it at scale

4. RoPE Position Encoding:
   - Rotary position embeddings
   - Good extrapolation to longer sequences

5. No Bias Terms:
   - Slightly better training stability
   - Fewer parameters
"""


def print_palm_configs():
    """Print PaLM model configurations"""
    print("\n📊 PaLM Model Configurations")
    print("=" * 70)
    
    print(f"{'Model':<12} {'Params':<10} {'Layers':<8} {'Heads':<8} {'d_model':<10} {'TPUs':<8}")
    print("-" * 70)
    
    for name, config in PALM_CONFIGS.items():
        print(f"{config.name:<12} {config.params}B{'':<5} {config.layers:<8} "
              f"{config.heads:<8} {config.d_model:<10} {config.tpu_chips:<8}")


def print_training_details():
    """Print training details"""
    print("\n📊 PaLM 540B Training Details")
    print("=" * 70)
    
    config = PALM_CONFIGS['540B']
    metrics = PathwaysSystem.compute_efficiency_metrics(config)
    
    print(f"""
Training Configuration:
    - Model: PaLM 540B
    - TPU v4 Chips: {config.tpu_chips:,}
    - Training Tokens: 780 billion
    - Sequence Length: 2048
    - Batch Size: 2048 sequences (4M tokens per batch)
    
Efficiency:
    - Peak PFLOPS: {metrics['peak_pflops']:.1f}
    - Achieved PFLOPS: {metrics['achieved_pflops']:.1f}
    - MFU (Model FLOPS Utilization): {metrics['mfu']:.1%}
    
Training Time:
    - Total training time: ~2 months
    - Including restarts and debugging
""")


def show_emergent_abilities():
    """Show emergent abilities discovered"""
    print("\n📊 Emergent Abilities in PaLM")
    print("=" * 70)
    
    print("""
PaLM discovered several emergent abilities:

1. Joke Explanation:
   Small models: Cannot explain jokes
   PaLM 540B: Can explain why jokes are funny
   
2. Multi-step Arithmetic:
   Small models: Fail at multi-step math
   PaLM 540B: Can do chain-of-thought arithmetic
   
3. Code Understanding:
   PaLM can explain code, fix bugs, translate between languages
   
4. Multi-lingual:
   Trained primarily on English, but learns multilingual abilities
   
5. Few-shot Learning:
   One of the first models to show strong few-shot across many tasks

Breakthrough Finding:
   Some abilities only appear at very large scale (100B+)
   This is called "emergent behavior" - not present in smaller models
""")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("PaLM: Scaling Language Modeling with Pathways")
    print("Chowdhery et al., 2022")
    print("=" * 70)
    
    print(PathwaysSystem.describe_parallelism())
    print(PaLMArchitecture.describe_architecture())
    print_palm_configs()
    print_training_details()
    show_emergent_abilities()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Scale Proof: Showed 540B models are trainable
    
    2. Efficiency: 46% MFU on 6144 chips is impressive
    
    3. Architecture Insights: Parallel attention + SwiGLU
    
    4. Emergent Abilities: Discovered new phenomena
    
    5. Infrastructure: Pathways enabled cross-pod training
    
    Key Insight:
        Large-scale training is as much about infrastructure
        as it is about algorithms. PaLM was a systems triumph.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
