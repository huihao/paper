#!/usr/bin/env python3
"""
Hierarchical Mixtures of Experts (Jordan & Jacobs, 1994)
Extended MoE with hierarchical gating structure.

Paper: https://www.cs.toronto.edu/~hinton/absps/hme.pdf
"""

import numpy as np
from typing import List, Dict, Tuple


class HierarchicalMoE:
    """
    Hierarchical Mixtures of Experts (1994)
    
    Extension of original MoE with:
    - Tree-structured gating network
    - Multiple levels of specialization
    - Soft decision trees for routing
    
    Structure:
                    [Gate 0]
                   /        \
              [Gate 1]    [Gate 2]
              /    \      /    \
           [E1]  [E2]  [E3]  [E4]
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 2,
        experts_per_leaf: int = 1
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        
        # Number of leaf experts = 2^depth
        self.num_leaves = 2 ** depth
        
        # Gating networks at each level
        # Each internal node has its own gate
        self.gates = self._init_gates()
        
        # Expert networks at leaves
        self.experts = self._init_experts()
    
    def _init_gates(self) -> Dict:
        """Initialize hierarchical gates"""
        gates = {}
        for level in range(self.depth):
            num_gates = 2 ** level
            for g in range(num_gates):
                gate_id = f"L{level}_G{g}"
                gates[gate_id] = {
                    'W': np.random.randn(self.input_dim, 2) * 0.1,
                    'b': np.zeros(2)
                }
        return gates
    
    def _init_experts(self) -> List[Dict]:
        """Initialize leaf expert networks"""
        experts = []
        hidden_dim = 32
        for _ in range(self.num_leaves):
            experts.append({
                'W1': np.random.randn(self.input_dim, hidden_dim) * 0.1,
                'b1': np.zeros(hidden_dim),
                'W2': np.random.randn(hidden_dim, self.output_dim) * 0.1,
                'b2': np.zeros(self.output_dim)
            })
        return experts
    
    def forward_gate(self, x: np.ndarray, gate_id: str) -> np.ndarray:
        """Forward through a single gate"""
        W = self.gates[gate_id]['W']
        b = self.gates[gate_id]['b']
        logits = np.matmul(x, W) + b
        # Softmax for left/right probability
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def forward_expert(self, x: np.ndarray, expert_idx: int) -> np.ndarray:
        """Forward through a leaf expert"""
        e = self.experts[expert_idx]
        hidden = np.maximum(0, np.matmul(x, e['W1']) + e['b1'])
        return np.matmul(hidden, e['W2']) + e['b2']
    
    def compute_leaf_probabilities(self, x: np.ndarray) -> np.ndarray:
        """
        Compute probability of reaching each leaf expert.
        
        This traverses the tree and multiplies gate probabilities
        along each path to get the final routing probability.
        """
        batch_size = x.shape[0]
        # Start with probability 1 at root
        probs = np.ones((batch_size, 1))
        
        for level in range(self.depth):
            new_probs = []
            num_gates = 2 ** level
            
            for g in range(num_gates):
                gate_id = f"L{level}_G{g}"
                gate_probs = self.forward_gate(x, gate_id)  # [batch, 2]
                
                # Multiply parent probability by left/right probs
                parent_prob = probs[:, g:g+1]
                left_prob = parent_prob * gate_probs[:, 0:1]
                right_prob = parent_prob * gate_probs[:, 1:2]
                new_probs.extend([left_prob, right_prob])
            
            probs = np.concatenate(new_probs, axis=1)
        
        return probs  # [batch, num_leaves]
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full forward pass through hierarchical MoE.
        
        Returns:
            output: Weighted combination of all expert outputs
            leaf_probs: Probability of each leaf expert
        """
        # Get probability of each leaf
        leaf_probs = self.compute_leaf_probabilities(x)
        
        # Get outputs from all experts
        expert_outputs = [
            self.forward_expert(x, i) 
            for i in range(self.num_leaves)
        ]
        expert_outputs = np.stack(expert_outputs, axis=1)
        
        # Weighted combination
        output = np.sum(
            expert_outputs * leaf_probs[:, :, np.newaxis],
            axis=1
        )
        
        return output, leaf_probs
    
    @staticmethod
    def describe_concept():
        return """
Hierarchical Mixtures of Experts (1994):

The Hierarchical Extension:
    Original MoE: Flat structure, all experts equal
    Hierarchical MoE: Tree structure, progressive specialization
    
    Binary Tree Structure (depth=2):
    
                        ┌─────────────┐
                        │   Root Gate │
                        │  P(L), P(R) │
                        └──────┬──────┘
                       ┌───────┴───────┐
                       ↓               ↓
                 ┌─────────┐     ┌─────────┐
                 │ Gate L  │     │ Gate R  │
                 │P(LL,LR) │     │P(RL,RR) │
                 └────┬────┘     └────┬────┘
                 ┌────┴────┐     ┌────┴────┐
                 ↓         ↓     ↓         ↓
            ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
            │Expert 1│ │Expert 2│ │Expert 3│ │Expert 4│
            │  (LL)  │ │  (LR)  │ │  (RL)  │ │  (RR)  │
            └────────┘ └────────┘ └────────┘ └────────┘

    Path probabilities:
        P(Expert 1) = P(L) × P(LL|L)
        P(Expert 2) = P(L) × P(LR|L)
        P(Expert 3) = P(R) × P(RL|R)
        P(Expert 4) = P(R) × P(RR|R)

Benefits of Hierarchy:
    1. Coarse-to-fine decisions
    2. Exponential expert capacity (2^depth)
    3. Interpretable decision paths
    4. Efficient inference (can prune paths)

Example: Image Classification
    Root: Is it an animal or object?
    Level 1: If animal, is it mammal or bird?
    Level 2: If mammal, is it cat or dog?
    
    Each expert specializes on a narrow category!
"""


def demonstrate_hierarchy():
    """Demonstrate hierarchical routing"""
    print("\n📊 Hierarchical Routing Demo")
    print("-" * 40)
    
    hme = HierarchicalMoE(
        input_dim=2,
        output_dim=1,
        depth=2
    )
    
    # Test points in different quadrants
    test_points = [
        ("Top-Right", np.array([[2.0, 2.0]])),
        ("Top-Left", np.array([[-2.0, 2.0]])),
        ("Bottom-Left", np.array([[-2.0, -2.0]])),
        ("Bottom-Right", np.array([[2.0, -2.0]])),
    ]
    
    print("Input Region → Leaf Expert Probabilities:")
    print("-" * 50)
    for name, x in test_points:
        _, leaf_probs = hme.forward(x)
        probs_str = ", ".join([f"E{i+1}:{p:.2f}" for i, p in enumerate(leaf_probs[0])])
        print(f"  {name:<15}: [{probs_str}]")
    
    print("\nIn a trained model, the tree would partition")
    print("the input space hierarchically!")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("Hierarchical Mixtures of Experts (Jordan & Jacobs, 1994)")
    print("=" * 70)
    
    print(HierarchicalMoE.describe_concept())
    demonstrate_hierarchy()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Hierarchy: Introduced multi-level gating
    
    2. Scalability: Exponential experts with logarithmic depth
    
    3. Interpretability: Clear decision paths
    
    4. Efficiency: Can prune unlikely paths
    
    5. Foundation: Influenced decision tree + neural network hybrids
    
    Key Insight:
        Complex decisions can be decomposed into
        a hierarchy of simpler decisions.
        
        This is the divide-and-conquer principle
        applied to neural network gating.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
