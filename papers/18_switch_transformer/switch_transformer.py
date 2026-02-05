#!/usr/bin/env python3
"""
Switch Transformers: Scaling to Trillion Parameter Models
(Fedus et al., 2021)

Simplified MoE routing using single-expert activation for stable training.

Paper: https://arxiv.org/abs/2101.03961
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class SwitchConfig:
    """Switch Transformer configuration"""
    num_experts: int
    d_model: int
    d_ff: int
    capacity_factor: float
    expert_dropout: float


class SwitchLayer:
    """
    Switch Transformer MoE Layer
    
    Key simplifications over original MoE:
    1. Route to exactly 1 expert (not top-k)
    2. Simpler load balancing loss
    3. Expert dropout for regularization
    4. Capacity factor to limit expert load
    """
    
    def __init__(self, config: SwitchConfig):
        self.config = config
        
        # Expert networks
        self.experts = [
            {
                'W1': np.random.randn(config.d_model, config.d_ff) * 0.02,
                'W2': np.random.randn(config.d_ff, config.d_model) * 0.02,
            }
            for _ in range(config.num_experts)
        ]
        
        # Simple gating weights (no noise)
        self.W_gate = np.random.randn(config.d_model, config.num_experts) * 0.02
    
    def compute_switch_gate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple switch routing: each token goes to exactly 1 expert.
        
        Args:
            x: Input [batch, d_model]
            
        Returns:
            gate_probs: Probability of selected expert [batch, 1]
            expert_indices: Selected expert index [batch]
        """
        # Compute gating logits
        logits = np.matmul(x, self.W_gate)  # [batch, num_experts]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Select single expert (argmax)
        expert_indices = np.argmax(probs, axis=-1)  # [batch]
        
        # Get probability of selected expert
        gate_probs = probs[np.arange(x.shape[0]), expert_indices]
        
        return gate_probs, expert_indices
    
    def forward_expert(self, x: np.ndarray, expert_idx: int) -> np.ndarray:
        """Forward through single expert"""
        W1 = self.experts[expert_idx]['W1']
        W2 = self.experts[expert_idx]['W2']
        hidden = np.maximum(0, np.matmul(x, W1))
        return np.matmul(hidden, W2)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Switch layer forward pass.
        
        Each token routed to exactly one expert.
        Output scaled by gate probability.
        """
        batch_size = x.shape[0]
        gate_probs, expert_indices = self.compute_switch_gate(x)
        
        output = np.zeros_like(x)
        
        # Group tokens by expert
        expert_counts = np.zeros(self.config.num_experts)
        
        for i in range(batch_size):
            expert_idx = expert_indices[i]
            expert_counts[expert_idx] += 1
            
            # Check capacity
            capacity = int(batch_size / self.config.num_experts * self.config.capacity_factor)
            if expert_counts[expert_idx] <= capacity:
                expert_output = self.forward_expert(x[i:i+1], expert_idx)
                output[i:i+1] = gate_probs[i] * expert_output
            # else: token is dropped (output stays zero)
        
        aux_info = {
            'expert_counts': expert_counts,
            'dropped_tokens': int(np.sum(expert_counts > capacity))
        }
        
        return output, aux_info
    
    @staticmethod
    def describe_architecture():
        return """
Switch Transformer Architecture:

Key Simplification: Route to EXACTLY 1 expert

Original MoE (Shazeer 2017):
    Token → Top-k experts (k=2 or 4)
    - More compute per token
    - Complex routing logic
    
Switch Transformer:
    Token → Single expert (k=1)
    - Simpler, faster
    - Same quality at scale
    
    ┌─────────────────────────────────────────────────┐
    │ Token: "cat"                                    │
    │   ↓                                             │
    │ Router: softmax(x @ W_gate)                     │
    │   ↓                                             │
    │ Select: argmax → Expert 7                       │
    │   ↓                                             │
    │ Output: gate_prob * Expert_7(x)                │
    └─────────────────────────────────────────────────┘

Why It Works:
    1. Each expert sees more tokens (not shared with top-k others)
    2. Simpler gradient flow
    3. Lower memory for routing
    4. Better scaling properties
"""

    @staticmethod
    def describe_load_balancing():
        return """
Switch Transformer Load Balancing:

The Problem:
    If all tokens go to one expert:
    - That expert is overloaded
    - Others are undertrained
    - Wasted capacity

Solution: Auxiliary Loss

    L_aux = α × Σᵢ fᵢ × Pᵢ
    
    Where:
    - fᵢ = fraction of tokens routed to expert i
    - Pᵢ = average routing probability to expert i
    
    This loss encourages:
    - Uniform token distribution across experts
    - Minimizes both imbalance and confidence

Capacity Factor:
    Each expert can handle at most:
    
    capacity = (tokens_per_batch / num_experts) × capacity_factor
    
    capacity_factor = 1.0: Exactly balanced (some drops)
    capacity_factor = 1.25: 25% overflow allowed
    capacity_factor = 2.0: 100% overflow (rarely needed)
    
    Tokens exceeding capacity are DROPPED.
    Their gradient flows through the gate probability only.
"""


def demonstrate_switch_routing():
    """Demonstrate switch routing"""
    print("\n📊 Switch Routing Demo")
    print("-" * 40)
    
    config = SwitchConfig(
        num_experts=8,
        d_model=64,
        d_ff=256,
        capacity_factor=1.25,
        expert_dropout=0.0
    )
    
    switch = SwitchLayer(config)
    
    # Sample tokens
    x = np.random.randn(16, 64)  # 16 tokens
    
    gate_probs, expert_indices = switch.compute_switch_gate(x)
    
    print("Switch routing for 16 tokens (to 1 of 8 experts):")
    print("-" * 50)
    for i in range(16):
        print(f"  Token {i:2d} → Expert {expert_indices[i]} (prob: {gate_probs[i]:.3f})")
    
    # Expert distribution
    unique, counts = np.unique(expert_indices, return_counts=True)
    print(f"\nExpert load distribution:")
    for e, c in zip(unique, counts):
        print(f"  Expert {e}: {c} tokens {'⚠️ (overloaded)' if c > 2 else ''}")


def show_switch_results():
    """Show Switch Transformer results"""
    print("\n📊 Switch Transformer Results")
    print("=" * 70)
    
    print("""
Scaling Comparison:

┌─────────────────────────────────────────────────────────────────────┐
│ Model                 │ Parameters  │ Quality    │ Speed Gain      │
├─────────────────────────────────────────────────────────────────────┤
│ T5-Base               │ 223M        │ Baseline   │ 1.0x            │
│ Switch-Base-8         │ 1.4B        │ +4.4%      │ 1.3x            │
│ Switch-Base-64        │ 7.4B        │ +6.7%      │ 1.4x            │
├─────────────────────────────────────────────────────────────────────┤
│ T5-XXL                │ 11B         │ Baseline   │ 1.0x            │
│ Switch-XXL-128        │ 395B        │ +4.5%      │ 1.0x (!)        │
└─────────────────────────────────────────────────────────────────────┘

Key Finding:
    Switch-C (1.6T parameters) trained stably!
    First publicly documented trillion-parameter model.
    
    The k=1 routing simplification was crucial for stability.
""")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("Switch Transformers: Scaling to Trillion Parameters")
    print("(Fedus et al., 2021)")
    print("=" * 70)
    
    print(SwitchLayer.describe_architecture())
    print(SwitchLayer.describe_load_balancing())
    demonstrate_switch_routing()
    show_switch_results()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Simplification: k=1 routing works!
    
    2. Trillion Scale: First public 1.6T model
    
    3. Stability: Key insights for MoE training
    
    4. Efficiency: Same quality, less compute
    
    5. Foundation: Influenced Mixtral, GLaM, etc.
    
    Key Insight:
        Sometimes simpler is better.
        Routing to 1 expert instead of k experts
        enabled stable training at unprecedented scale.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
