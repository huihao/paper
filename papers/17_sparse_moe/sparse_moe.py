#!/usr/bin/env python3
"""
Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
(Shazeer et al., 2017)

The paper that ignited modern MoE - conditional computation at scale.

Paper: https://arxiv.org/abs/1701.06538
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class NoisyTopKGating:
    """
    Noisy Top-K Gating Mechanism
    
    The original MoE gating from Shazeer et al. 2017.
    
    Key innovations:
    1. Sparsity: Only activate top-k experts per token
    2. Noise: Add noise during training for exploration
    3. Load balancing: Auxiliary loss to prevent expert collapse
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_experts: int,
        k: int = 4,
        noise_std: float = 1.0
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        
        # Gating network weights
        self.W_gate = np.random.randn(input_dim, num_experts) * 0.01
        self.W_noise = np.random.randn(input_dim, num_experts) * 0.01
    
    def forward(
        self, 
        x: np.ndarray, 
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Compute gating values.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            training: Whether we're training (add noise)
            
        Returns:
            gates: Gating weights (batch_size, k)
            indices: Expert indices (batch_size, k)
            aux_info: Auxiliary information for loss
        """
        batch_size = x.shape[0]
        
        # Compute clean gating scores
        clean_logits = np.matmul(x, self.W_gate)  # (batch_size, num_experts)
        
        if training:
            # Add noise for exploration
            noise_logits = np.matmul(x, self.W_noise)
            noise_std = softmax(noise_logits) * self.noise_std
            noise = np.random.randn(batch_size, self.num_experts) * noise_std
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
        
        # Top-k selection
        top_k_indices = np.argsort(noisy_logits, axis=-1)[:, -self.k:]
        
        # Get top-k logits and compute gates
        top_k_logits = np.take_along_axis(noisy_logits, top_k_indices, axis=-1)
        gates = softmax(top_k_logits, axis=-1)
        
        # Compute auxiliary info for load balancing
        aux_info = self._compute_aux_info(clean_logits, top_k_indices)
        
        return gates, top_k_indices, aux_info
    
    def _compute_aux_info(
        self, 
        logits: np.ndarray, 
        selected_indices: np.ndarray
    ) -> dict:
        """Compute information for load balancing loss"""
        batch_size = logits.shape[0]
        
        # Count how many times each expert is selected
        expert_counts = np.zeros(self.num_experts)
        for batch_idx in range(batch_size):
            for exp_idx in selected_indices[batch_idx]:
                expert_counts[exp_idx] += 1
        
        # Importance (sum of gates for each expert)
        importance = np.mean(softmax(logits, axis=-1), axis=0)
        
        return {
            'expert_counts': expert_counts,
            'importance': importance,
            'load_balance': np.std(expert_counts)  # Lower is better
        }


class Expert:
    """A single expert (FFN)"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.02
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.02
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, np.matmul(x, self.W1))  # ReLU
        return np.matmul(hidden, self.W2)


class SparseMoELayer:
    """
    Sparsely-Gated Mixture-of-Experts Layer
    
    The key idea: conditional computation.
    Instead of using all parameters for every input,
    only activate a subset of "experts" per token.
    
    Benefits:
    - Massive total parameter count
    - Constant compute per token
    - Specialized experts for different inputs
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        k: int = 4
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        
        # Gating network
        self.gating = NoisyTopKGating(input_dim, num_experts, k)
        
        # Expert networks
        self.experts = [
            Expert(input_dim, hidden_dim, input_dim)
            for _ in range(num_experts)
        ]
    
    def __call__(
        self, 
        x: np.ndarray, 
        training: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            training: Whether we're training
            
        Returns:
            output: Output tensor (batch_size, input_dim)
            aux_info: Auxiliary information
        """
        batch_size = x.shape[0]
        
        # Get gating values
        gates, indices, aux_info = self.gating.forward(x, training)
        
        # Compute output for each sample
        output = np.zeros((batch_size, self.input_dim))
        
        for batch_idx in range(batch_size):
            for k_idx in range(self.k):
                expert_idx = indices[batch_idx, k_idx]
                gate_value = gates[batch_idx, k_idx]
                
                expert_output = self.experts[expert_idx](x[batch_idx:batch_idx+1])
                output[batch_idx] += gate_value * expert_output.squeeze()
        
        return output, aux_info
    
    def count_active_params(self) -> dict:
        """Count active vs total parameters"""
        # Expert parameters
        expert_params = self.experts[0].W1.size + self.experts[0].W2.size
        total_expert_params = expert_params * self.num_experts
        active_expert_params = expert_params * self.k
        
        # Gating parameters
        gating_params = self.gating.W_gate.size + self.gating.W_noise.size
        
        return {
            'total_params': total_expert_params + gating_params,
            'active_per_token': active_expert_params + gating_params,
            'efficiency': self.k / self.num_experts
        }


def load_balancing_loss(aux_info: dict, num_experts: int) -> float:
    """
    Compute load balancing loss.
    
    We want experts to be used equally to prevent:
    1. Expert collapse (one expert handles everything)
    2. Wasted capacity (some experts never used)
    
    L_importance = cv(importance)^2
    L_load = cv(load)^2
    """
    importance = aux_info['importance']
    expert_counts = aux_info['expert_counts']
    
    # Coefficient of variation for importance
    cv_importance = np.std(importance) / (np.mean(importance) + 1e-10)
    
    # Coefficient of variation for load
    cv_load = np.std(expert_counts) / (np.mean(expert_counts) + 1e-10)
    
    # Combined loss
    loss = cv_importance ** 2 + cv_load ** 2
    
    return loss


def demo_moe_basics():
    """Demonstrate MoE basics"""
    print("=" * 70)
    print("Sparsely-Gated Mixture-of-Experts (Shazeer et al., 2017)")
    print("=" * 70)
    
    print("\n📚 The Key Insight:")
    print("-" * 40)
    print("""
    Instead of using all parameters for every input:
    → Only activate a subset of "experts" per token
    
    Dense Model:
        Every token uses all 175B parameters
        Compute = O(total_params)
    
    MoE Model:
        Total: 1 trillion parameters across 64 experts
        Each token uses only 4 experts (~60B params)
        Compute = O(active_params) << O(total_params)
    """)
    
    # Create MoE layer
    input_dim = 64
    hidden_dim = 256
    num_experts = 16
    k = 4
    
    moe = SparseMoELayer(input_dim, hidden_dim, num_experts, k)
    
    # Sample input
    batch_size = 8
    x = np.random.randn(batch_size, input_dim)
    
    print("\n📊 MoE Layer Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of experts: {num_experts}")
    print(f"  Active experts (k): {k}")
    
    # Forward pass
    output, aux_info = moe(x, training=True)
    
    print(f"\n📊 Forward Pass Results:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expert usage distribution: {aux_info['expert_counts']}")
    print(f"  Load balance (lower is better): {aux_info['load_balance']:.4f}")
    
    # Parameter efficiency
    param_info = moe.count_active_params()
    print(f"\n📊 Parameter Efficiency:")
    print(f"  Total parameters: {param_info['total_params']:,}")
    print(f"  Active per token: {param_info['active_per_token']:,}")
    print(f"  Efficiency ratio: {param_info['efficiency']:.1%}")


def explain_gating():
    """Explain the gating mechanism"""
    print("\n" + "=" * 70)
    print("📚 Understanding Noisy Top-K Gating")
    print("=" * 70)
    
    print("""
The Gating Mechanism:

1. Compute Gating Scores:
   H(x) = softmax(x · W_g + noise)
   
2. Select Top-K Experts:
   Keep only k highest-scoring experts
   
3. Renormalize Gates:
   Normalize weights among selected experts

Why Noise?
   - Encourages exploration during training
   - Helps discover which experts work best
   - Prevents early expert collapse

Why Load Balancing Loss?
   Without it:
   - One expert might dominate
   - Other experts become useless
   - Waste of capacity
   
   With it:
   - Experts share the load
   - All capacity is utilized
   - Better specialization
""")


def compare_dense_vs_moe():
    """Compare dense and MoE models"""
    print("\n" + "=" * 70)
    print("📊 Dense vs MoE Comparison")
    print("=" * 70)
    
    print("""
Example: 1 Trillion Parameter Model

Dense Model (GPT-3 style):
┌──────────────────────────────────────┐
│  1T parameters                       │
│  All used for every token            │
│  Compute: ~6T FLOPs per token        │
└──────────────────────────────────────┘

MoE Model (64 experts, top-4):
┌────────────────────────────────────────────────────────┐
│  1T total parameters (64 experts × ~15B each)         │
│  Only 4 experts active per token (~60B)               │
│  Compute: ~360B FLOPs per token                       │
│  16x less compute for same parameter count!           │
└────────────────────────────────────────────────────────┘

Trade-offs:
+ Massive capacity increase
+ Same inference compute
+ Experts can specialize
- Harder to train
- Need load balancing
- Communication overhead in distributed training
""")


def demo():
    """Main demonstration"""
    demo_moe_basics()
    explain_gating()
    compare_dense_vs_moe()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Introduced sparsity at scale
    
    2. Showed conditional computation works
    
    3. Enabled trillion-parameter models
    
    4. Foundation for Switch Transformers, Mixtral, etc.
    
    5. Key idea: Capacity ≠ Compute
       You can have enormous capacity while 
       keeping compute tractable.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
