#!/usr/bin/env python3
"""
Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints
(Komatsuzaki et al., 2022)

Practical technique for converting dense checkpoints into MoE models.

Paper: https://arxiv.org/abs/2212.05055
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UpcycleConfig:
    """Configuration for upcycling"""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_experts: int = 8
    num_active: int = 2


class DenseFFN:
    """Dense Feed-Forward Network (source)"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.up_proj = np.random.randn(hidden_size, intermediate_size) * 0.02
        self.down_proj = np.random.randn(intermediate_size, hidden_size) * 0.02
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, np.matmul(x, self.up_proj))
        return np.matmul(hidden, self.down_proj)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            'up_proj': self.up_proj.copy(),
            'down_proj': self.down_proj.copy()
        }


class SparseUpcycler:
    """
    Sparse Upcycling: Convert Dense to MoE
    
    Key insight: Start with a trained dense model and convert it to MoE.
    - Replicate FFN weights to create experts
    - Add random noise for differentiation
    - Continue training to specialize experts
    
    Benefits:
    - Reuse compute from dense training
    - Faster convergence than training from scratch
    - Experts start from good initialization
    """
    
    def __init__(self, config: UpcycleConfig):
        self.config = config
    
    def upcycle_ffn(
        self,
        dense_ffn: DenseFFN,
        noise_scale: float = 0.01
    ) -> List[Dict[str, np.ndarray]]:
        """
        Convert a dense FFN to multiple expert FFNs.
        
        Strategy:
        1. Copy dense weights to each expert
        2. Add small random noise for differentiation
        3. The router is initialized from scratch
        
        Args:
            dense_ffn: The source dense FFN
            noise_scale: Scale of noise to add for differentiation
            
        Returns:
            List of expert weight dictionaries
        """
        base_weights = dense_ffn.get_weights()
        expert_weights = []
        
        for i in range(self.config.num_experts):
            # Copy weights
            expert = {
                'up_proj': base_weights['up_proj'].copy(),
                'down_proj': base_weights['down_proj'].copy()
            }
            
            # Add noise for differentiation
            expert['up_proj'] += np.random.randn(*expert['up_proj'].shape) * noise_scale
            expert['down_proj'] += np.random.randn(*expert['down_proj'].shape) * noise_scale
            
            expert_weights.append(expert)
        
        return expert_weights
    
    def initialize_router(self) -> np.ndarray:
        """
        Initialize router weights.
        
        The router starts random since we want the model to
        learn which inputs should go to which experts.
        """
        return np.random.randn(self.config.hidden_size, self.config.num_experts) * 0.02
    
    def estimate_additional_params(self) -> Dict:
        """Estimate parameter increase from upcycling"""
        dense_params = self.config.hidden_size * self.config.intermediate_size * 2  # up + down
        
        moe_params = dense_params * self.config.num_experts
        router_params = self.config.hidden_size * self.config.num_experts
        
        return {
            'dense_ffn_params': dense_params,
            'moe_total_params': moe_params + router_params,
            'multiplier': (moe_params + router_params) / dense_params,
            'active_multiplier': (dense_params * self.config.num_active + router_params) / dense_params
        }


def explain_upcycling_process():
    """Explain the upcycling process"""
    print("=" * 70)
    print("Sparse Upcycling: Dense → MoE Conversion")
    print("Komatsuzaki et al., 2022")
    print("=" * 70)
    
    print("""
The Upcycling Process:

Step 1: Start with Trained Dense Model
    ┌────────────────────────────────┐
    │  Dense FFN (trained)           │
    │  [up_proj, down_proj]          │
    └────────────────────────────────┘

Step 2: Replicate to N Experts
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Expert 0 │ │ Expert 1 │ │ Expert 2 │ │ Expert 3 │
    │ (copy)   │ │ (copy)   │ │ (copy)   │ │ (copy)   │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘
          ↓           ↓           ↓           ↓
    Add small random noise for differentiation

Step 3: Add Router
    ┌────────────────────────────────┐
    │  Router (random init)          │
    │  Routes tokens to top-k experts│
    └────────────────────────────────┘

Step 4: Continue Training
    - Router learns which expert for which input
    - Experts specialize from same starting point
    - Much faster than training MoE from scratch!
""")


def demonstrate_upcycling():
    """Demonstrate the upcycling process"""
    print("\n📊 Upcycling Demonstration")
    print("-" * 40)
    
    config = UpcycleConfig(
        hidden_size=64,
        intermediate_size=256,
        num_experts=8,
        num_active=2
    )
    
    # Create dense FFN (pretend it's trained)
    dense_ffn = DenseFFN(config.hidden_size, config.intermediate_size)
    
    print(f"Dense FFN weights shape:")
    print(f"  up_proj: {dense_ffn.up_proj.shape}")
    print(f"  down_proj: {dense_ffn.down_proj.shape}")
    
    # Upcycle
    upcycler = SparseUpcycler(config)
    expert_weights = upcycler.upcycle_ffn(dense_ffn, noise_scale=0.01)
    
    print(f"\nAfter upcycling:")
    print(f"  Number of experts: {len(expert_weights)}")
    
    # Check similarity between experts
    print(f"\nExpert similarity (should be high initially):")
    for i in range(min(3, len(expert_weights))):
        for j in range(i+1, min(3, len(expert_weights))):
            up_sim = np.corrcoef(
                expert_weights[i]['up_proj'].flatten(),
                expert_weights[j]['up_proj'].flatten()
            )[0, 1]
            print(f"  Expert {i} vs Expert {j}: r={up_sim:.4f}")
    
    # Parameter efficiency
    param_info = upcycler.estimate_additional_params()
    print(f"\nParameter Impact:")
    print(f"  Dense FFN params: {param_info['dense_ffn_params']:,}")
    print(f"  MoE total params: {param_info['moe_total_params']:,}")
    print(f"  Parameter multiplier: {param_info['multiplier']:.1f}x")
    print(f"  Active multiplier: {param_info['active_multiplier']:.1f}x")


def compare_training_approaches():
    """Compare upcycling vs training from scratch"""
    print("\n" + "=" * 70)
    print("📊 Upcycling vs Training from Scratch")
    print("=" * 70)
    
    print("""
Training MoE from Scratch:
    ┌─────────────────────────────────────────────────┐
    │  Initialize randomly                            │
    │  Train for T steps                              │
    │  Total compute: T × C                           │
    │                                                 │
    │  Challenges:                                    │
    │  - Expert collapse (some experts unused)        │
    │  - Training instability                         │
    │  - Slow convergence                             │
    └─────────────────────────────────────────────────┘

Sparse Upcycling:
    ┌─────────────────────────────────────────────────┐
    │  Phase 1: Train dense model (T_dense steps)     │
    │  Phase 2: Upcycle to MoE                        │
    │  Phase 3: Continue training (T_continue steps) │
    │                                                 │
    │  Benefits:                                      │
    │  - Reuse dense training compute                 │
    │  - Experts start specialized                    │
    │  - More stable training                         │
    │  - Faster to useful MoE                         │
    └─────────────────────────────────────────────────┘

Empirical Results (from paper):
    - Upcycled MoE matches from-scratch with ~50% less training
    - Better sample efficiency
    - More stable training dynamics
    - Easier to scale to many experts
""")


def explain_practical_benefits():
    """Explain practical benefits"""
    print("\n" + "=" * 70)
    print("📚 Practical Benefits of Upcycling")
    print("=" * 70)
    
    print("""
1. Compute Reuse:
   You've already trained a 7B model? Great!
   Convert it to 8x7B MoE and continue training.
   Don't waste the original training investment.

2. Iterative Scaling:
   Train dense → upcycle → train MoE → upcycle more
   Each step builds on previous work.

3. Risk Reduction:
   Dense models are well-understood.
   Start there, convert when confident.

4. Experimentation:
   Easy to try different MoE configurations
   from the same dense checkpoint.

5. Deployment Flexibility:
   Keep dense for simple cases.
   Use MoE for complex queries.
   Same base knowledge.

Used in Practice:
    - DeepSeek-MoE (from DeepSeek dense)
    - Qwen-MoE (from Qwen dense)
    - Many internal research models
""")


def demo():
    """Main demonstration"""
    explain_upcycling_process()
    demonstrate_upcycling()
    compare_training_approaches()
    explain_practical_benefits()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Practical Technique: Actually used in production
    
    2. Compute Efficiency: Reuse existing training investment
    
    3. Lower Risk: Iterate from working dense models
    
    4. Faster Experimentation: Try MoE without full retraining
    
    5. Foundation for Modern MoE: Influenced DeepSeek, Qwen
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
