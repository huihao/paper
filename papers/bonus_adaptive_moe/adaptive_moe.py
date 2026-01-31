#!/usr/bin/env python3
"""
Adaptive Mixtures of Local Experts (Jacobs et al., 1991)
The original Mixture of Experts paper.

Paper: https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf
"""

import numpy as np
from typing import List, Tuple


class AdaptiveMixtureOfExperts:
    """
    Original Mixture of Experts (1991)
    
    The foundational paper that introduced:
    - Expert networks specialized on subsets of data
    - Gating network to route inputs to experts
    - Competitive learning between experts
    
    This is where it all began!
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        hidden_dim: int = 32
    ):
        self.num_experts = num_experts
        
        # Expert networks (simple MLPs)
        self.expert_weights = [
            {
                'W1': np.random.randn(input_dim, hidden_dim) * 0.1,
                'b1': np.zeros(hidden_dim),
                'W2': np.random.randn(hidden_dim, output_dim) * 0.1,
                'b2': np.zeros(output_dim)
            }
            for _ in range(num_experts)
        ]
        
        # Gating network
        self.gate_W = np.random.randn(input_dim, num_experts) * 0.1
        self.gate_b = np.zeros(num_experts)
    
    def forward_expert(self, x: np.ndarray, expert_idx: int) -> np.ndarray:
        """Forward pass through one expert"""
        W1 = self.expert_weights[expert_idx]['W1']
        b1 = self.expert_weights[expert_idx]['b1']
        W2 = self.expert_weights[expert_idx]['W2']
        b2 = self.expert_weights[expert_idx]['b2']
        
        # Simple MLP: x -> hidden -> output
        hidden = np.maximum(0, np.matmul(x, W1) + b1)  # ReLU
        output = np.matmul(hidden, W2) + b2
        return output
    
    def forward_gate(self, x: np.ndarray) -> np.ndarray:
        """Compute gating weights (which expert to use)"""
        logits = np.matmul(x, self.gate_W) + self.gate_b
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full forward pass.
        
        Returns:
            output: Weighted combination of expert outputs
            gate_probs: Probability of each expert
        """
        # Get expert outputs
        expert_outputs = [
            self.forward_expert(x, i) 
            for i in range(self.num_experts)
        ]
        expert_outputs = np.stack(expert_outputs, axis=1)  # [batch, experts, out]
        
        # Get gating weights
        gate_probs = self.forward_gate(x)  # [batch, experts]
        
        # Weighted combination
        output = np.sum(
            expert_outputs * gate_probs[:, :, np.newaxis],
            axis=1
        )
        
        return output, gate_probs
    
    @staticmethod
    def describe_concept():
        return """
Adaptive Mixtures of Local Experts (1991):

The Original Idea:
    Instead of one big network, use multiple "expert" networks,
    each specializing on different parts of the input space.
    
    ┌─────────────────────────────────────────────────┐
    │                    Input x                      │
    │                      │                          │
    │        ┌─────────────┼─────────────┐           │
    │        ↓             ↓             ↓           │
    │   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
    │   │Expert 1 │   │Expert 2 │   │Expert 3 │     │
    │   │(Region A)│   │(Region B)│   │(Region C)│     │
    │   └────┬────┘   └────┬────┘   └────┬────┘     │
    │        │             │             │           │
    │        ↓             ↓             ↓           │
    │   ┌───────────────────────────────────────┐    │
    │   │     Gating Network (decides weights)  │    │
    │   │        g₁ = 0.1, g₂ = 0.7, g₃ = 0.2   │    │
    │   └───────────────────────────────────────┘    │
    │                      │                          │
    │        Output = g₁·E₁ + g₂·E₂ + g₃·E₃         │
    └─────────────────────────────────────────────────┘

Key Insights:
    1. Specialization: Each expert learns a subset of the problem
    2. Competition: Experts compete to explain each input
    3. Soft routing: Weighted combination, not hard selection
    4. End-to-end training: Gate and experts train together

The gating network learns to:
    - Recognize which expert is best for each input
    - Partition the input space among experts
    - Combine expert outputs appropriately
"""


def demonstrate_specialization():
    """Demonstrate expert specialization"""
    print("\n📊 Expert Specialization Demo")
    print("-" * 40)
    
    # Create MoE with 4 experts
    moe = AdaptiveMixtureOfExperts(
        input_dim=2,
        output_dim=1,
        num_experts=4
    )
    
    # Test on different regions of input space
    test_points = [
        np.array([[1.0, 1.0]]),   # Quadrant 1
        np.array([[-1.0, 1.0]]),  # Quadrant 2
        np.array([[-1.0, -1.0]]), # Quadrant 3
        np.array([[1.0, -1.0]]),  # Quadrant 4
    ]
    
    print("Input Region → Expert Probabilities:")
    print("-" * 50)
    for i, x in enumerate(test_points, 1):
        _, gate_probs = moe.forward(x)
        probs_str = ", ".join([f"E{j+1}:{p:.2f}" for j, p in enumerate(gate_probs[0])])
        print(f"  Quadrant {i} ({x[0]}): [{probs_str}]")
    
    print("\nNote: In trained MoE, different experts would specialize")
    print("on different input regions!")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("Adaptive Mixtures of Local Experts (Jacobs et al., 1991)")
    print("The Original Mixture of Experts Paper")
    print("=" * 70)
    
    print(AdaptiveMixtureOfExperts.describe_concept())
    demonstrate_specialization()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Origin: The first Mixture of Experts paper
    
    2. Core Ideas: Specialization + Gating still used today
    
    3. Competitive Learning: Experts compete for inputs
    
    4. Foundation: All modern MoE builds on this
    
    5. Simplicity: Elegant solution to complex problems
    
    Key Insight:
        Divide and conquer: Let different experts
        specialize on different parts of the problem.
        This is the core of all modern MoE systems.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
