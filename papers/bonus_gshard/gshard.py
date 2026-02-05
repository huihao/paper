#!/usr/bin/env python3
"""
GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding
(Lepikhin et al., 2020)

Pioneered scaling MoE to 600B parameters with efficient sharding.

Paper: https://arxiv.org/abs/2006.16668
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class GShardConfig:
    """GShard model configuration"""
    total_params: float  # billions
    num_experts: int
    expert_capacity: int
    d_model: int
    layers: int


class GShard:
    """
    GShard: Scaling Giant MoE Models
    
    Key innovations:
    1. Mixture of Experts at scale (600B params)
    2. Automatic model parallelism compiler
    3. Expert capacity and load balancing
    4. Auxiliary losses for balanced routing
    """
    
    @staticmethod
    def describe_architecture():
        return """
GShard Architecture:

The Translation Model:
    ┌─────────────────────────────────────────────────┐
    │ Encoder                                         │
    │   - Standard attention layers                   │
    │   - Every other FFN → MoE layer                │
    │   - 2048 experts per MoE layer                 │
    └─────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────┐
    │ Decoder                                         │
    │   - Standard attention layers                   │
    │   - Cross-attention to encoder                 │
    │   - Every other FFN → MoE layer                │
    │   - 2048 experts per MoE layer                 │
    └─────────────────────────────────────────────────┘

MoE Layer Detail:
    Input token → Router → Top-2 experts → Weighted output
    
    With 2048 experts:
    - Each token activates 2 experts (0.1% of total)
    - Massive capacity with sparse activation
"""

    @staticmethod
    def describe_sharding():
        return """
GShard Automatic Sharding:

Challenge:
    How to distribute 600B parameters across thousands of TPUs?

Solution: Compiler-based automatic sharding

SPMD Annotation:
    Programmer annotates tensor dimensions:
    - replicate(tensor)  # Copy to all devices
    - split(tensor, dim)  # Partition along dimension
    - Compiler infers communication patterns

Example:
    # Weight matrix [vocab, d_model]
    embedding = split(embedding_weights, dim=1)
    # Each device holds part of embedding dimension
    
    # Expert weights [num_experts, d_ff, d_model]  
    experts = split(expert_weights, dim=0)
    # Each device holds subset of experts

Benefits:
    1. No manual communication code
    2. Optimal data placement
    3. Scales to thousands of devices
    4. Works with existing code
"""

    @staticmethod
    def describe_load_balancing():
        return """
Expert Load Balancing:

Problem:
    If all tokens route to same experts:
    - Unbalanced compute
    - Some experts overtrained
    - Wasted capacity

Solution 1: Expert Capacity
    Each expert can only handle C tokens per batch
    C = (batch_size * seq_len * k) / num_experts * capacity_factor
    
    Overflow tokens are dropped or sent to second-choice expert

Solution 2: Auxiliary Loss
    L_aux = α * (sum over experts of: fraction_i * probability_i)
    
    Encourages uniform distribution of tokens to experts

GShard Results:
    With these techniques:
    - 96%+ expert utilization
    - No expert collapse
    - Stable training at 600B scale
"""


def print_gshard_specs():
    """Print GShard model specifications"""
    print("\n📊 GShard Model Specifications")
    print("=" * 70)
    
    print("""
GShard 600B Translation Model:

    ┌─────────────────────────────────────────────┐
    │ Total Parameters:  600 billion              │
    │ Active per token:  ~10 billion              │
    │ Experts per layer: 2048                     │
    │ Experts activated: 2 per token              │
    │ MoE layers:        Every other layer        │
    │ Training devices:  2048 TPU v3 cores        │
    │ Training time:     4 days                   │
    └─────────────────────────────────────────────┘

Performance:
    - State-of-the-art on 100+ language pairs
    - 13.5 BLEU improvement on low-resource languages
    - Near-human quality on many pairs
""")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("GShard: Scaling Giant Models with Conditional Computation")
    print("(Lepikhin et al., 2020)")
    print("=" * 70)
    
    print(GShard.describe_architecture())
    print(GShard.describe_sharding())
    print(GShard.describe_load_balancing())
    print_gshard_specs()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Scale: First 600B parameter model
    
    2. Automatic Sharding: Compiler-based distribution
    
    3. Load Balancing: Practical MoE training
    
    4. Translation: SOTA on 100+ language pairs
    
    5. Infrastructure: Influenced all later MoE work
    
    Key Insight:
        Automatic sharding + expert capacity makes
        MoE training practical at unprecedented scale.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
