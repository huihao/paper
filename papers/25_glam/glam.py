#!/usr/bin/env python3
"""
GLaM: Generalist Language Model (Du et al., 2022)
Validated MoE scaling economics with massive total parameters but small active counts.

Paper: https://arxiv.org/abs/2112.06905
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class GLaMConfig:
    """GLaM model configuration"""
    name: str
    total_params: float  # Total params in billions
    active_params: float  # Active params per token in billions
    num_experts: int
    experts_per_token: int
    layers: int


GLAM_CONFIGS = {
    'GLaM-base': GLaMConfig('GLaM Base', 143, 8, 64, 2, 32),
    'GLaM-1.7T': GLaMConfig('GLaM 1.7T', 1200, 96.6, 64, 2, 64),
}


class GLaMArchitecture:
    """
    GLaM Architecture: Generalist Language Model
    
    Key insight: Use Mixture of Experts (MoE) to achieve
    large total parameter count while keeping inference
    cost manageable.
    
    GLaM 1.7T:
    - 1.2T total parameters
    - 96.6B active per forward pass
    - 64 experts per layer
    - 2 experts activated per token
    """
    
    def __init__(self, config: GLaMConfig):
        self.config = config
    
    @staticmethod
    def describe_architecture():
        return """
GLaM Architecture:

Standard Dense Model:
    All parameters used for every token.
    Cost scales linearly with size.
    
    Forward Pass:
        Token → All Parameters → Output
        Cost: O(total_params)

GLaM (MoE Model):
    Only a subset of experts activated per token.
    Total params can be huge, active params small.
    
    Forward Pass:
        Token → Router → Selected Experts (2 of 64) → Output
        Cost: O(active_params) << O(total_params)

Layer Structure:
    ┌─────────────────────────────────────────────┐
    │ Multi-Head Attention                        │
    ├─────────────────────────────────────────────┤
    │ MoE Layer:                                  │
    │   ┌──────────────────────────────────────┐  │
    │   │ Router: Choose 2 of 64 experts       │  │
    │   ├──────────────────────────────────────┤  │
    │   │ Expert 1 │ Expert 2 │ ... │ Expert 64│  │
    │   │  (FFN)   │  (FFN)   │     │  (FFN)   │  │
    │   └──────────────────────────────────────┘  │
    │   Weighted sum of 2 selected experts        │
    └─────────────────────────────────────────────┘

Result:
    1.2T parameters for knowledge capacity
    96.6B compute cost per token
    ~12x more capacity than compute
"""


def compute_efficiency_gain(config: GLaMConfig) -> Dict:
    """Compute the efficiency gain of MoE"""
    capacity_ratio = config.total_params / config.active_params
    expert_utilization = config.experts_per_token / config.num_experts
    
    return {
        'capacity_ratio': capacity_ratio,
        'expert_utilization': expert_utilization,
        'total_params': config.total_params,
        'active_params': config.active_params,
        'compute_savings': 1 - (1 / capacity_ratio)
    }


def compare_dense_vs_moe():
    """Compare dense and MoE models"""
    print("\n📊 Dense vs MoE Comparison")
    print("=" * 70)
    
    print("""
GLaM vs Dense Comparison:

┌─────────────────────────────────────────────────────────────────────┐
│ Model        │ Total Params │ Active Params │ Capacity │ FLOPs     │
├─────────────────────────────────────────────────────────────────────┤
│ Dense 137B   │ 137B         │ 137B          │ 137B     │ 1.0x      │
│ GLaM 1.7T    │ 1,200B       │ 96.6B         │ 1,200B   │ 0.7x      │
└─────────────────────────────────────────────────────────────────────┘

Key Insight:
    GLaM has 8.8x more parameters but uses 0.7x the FLOPs!
    
    This means:
    - More knowledge storage capacity
    - Lower inference cost per token
    - Better quality per FLOP

Training Cost:
    GLaM uses 1/3 the training energy of GPT-3
    while achieving better or comparable quality.
""")


def show_training_economics():
    """Show the training economics of GLaM"""
    print("\n📊 Training Economics")
    print("=" * 70)
    
    config = GLAM_CONFIGS['GLaM-1.7T']
    efficiency = compute_efficiency_gain(config)
    
    print(f"""
GLaM 1.7T Training Economics:

Model Configuration:
    Total Parameters: {config.total_params:.1f}B
    Active Parameters: {config.active_params:.1f}B
    Number of Experts: {config.num_experts}
    Experts per Token: {config.experts_per_token}
    
Efficiency Metrics:
    Capacity/Compute Ratio: {efficiency['capacity_ratio']:.1f}x
    Expert Utilization: {efficiency['expert_utilization']*100:.1f}%
    Compute Savings: {efficiency['compute_savings']*100:.1f}%

Training Comparison:
    ┌────────────────────────────────────────────┐
    │ Metric          │ GPT-3    │ GLaM 1.7T   │
    ├────────────────────────────────────────────┤
    │ Params (total)  │ 175B     │ 1,200B      │
    │ Params (active) │ 175B     │ 96.6B       │
    │ Training FLOPs  │ 3.14e23  │ 1.76e23     │
    │ Energy (MWh)    │ 1,287    │ 456         │
    │ Zero-shot Avg   │ 55.4%    │ 55.0%       │
    │ One-shot Avg    │ 56.5%    │ 57.0%       │
    └────────────────────────────────────────────┘
    
    GLaM matches GPT-3 quality with 1/3 the energy!
""")


def explain_routing():
    """Explain the routing mechanism"""
    print("\n📊 Expert Routing")
    print("=" * 70)
    
    print("""
How Token Routing Works:

1. Compute Router Logits:
   router_logits = token_hidden @ router_weights
   # Shape: [num_experts]

2. Select Top-k Experts:
   top_k_experts = top_k(router_logits, k=2)
   # Select 2 of 64 experts

3. Compute Expert Weights:
   weights = softmax(selected_logits)
   # Normalize selected expert contributions

4. Apply Experts:
   output = sum(weight_i * expert_i(token) for i in top_k)

Load Balancing:
    To ensure all experts are used:
    - Add auxiliary loss encouraging uniform routing
    - Prevents expert collapse (all tokens to same expert)
    
    Auxiliary Loss:
    L_aux = α * sum(fraction_i * probability_i)
    
    Where:
    - fraction_i = fraction of tokens routed to expert i
    - probability_i = average routing probability to expert i
""")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("GLaM: Generalist Language Model (Du et al., 2022)")
    print("=" * 70)
    
    print(GLaMArchitecture.describe_architecture())
    compare_dense_vs_moe()
    show_training_economics()
    explain_routing()
    
    # Show configurations
    print("\n📊 GLaM Configurations")
    print("-" * 70)
    print(f"{'Model':<15} {'Total':<12} {'Active':<12} {'Experts':<10} {'Top-k':<8}")
    print("-" * 70)
    for name, config in GLAM_CONFIGS.items():
        print(f"{config.name:<15} {config.total_params}B{'':<6} "
              f"{config.active_params}B{'':<6} {config.num_experts:<10} {config.experts_per_token:<8}")
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Economics Validation: Proved MoE scaling works
    
    2. Energy Efficiency: 3x less energy than dense models
    
    3. Quality Proof: Matched GPT-3 with fewer active params
    
    4. Scale Demonstration: 1.2T parameters trainable
    
    5. Foundation: Paved the way for Mixtral, etc.
    
    Key Insight:
        Total parameters ≠ Compute cost
        With MoE, you can have both scale and efficiency.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
