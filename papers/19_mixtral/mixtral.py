#!/usr/bin/env python3
"""
Mixtral of Experts (Mistral AI, 2024)
Open-weight MoE that proves sparse models can match dense quality at lower inference cost.

Paper: https://arxiv.org/abs/2401.04088
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MixtralConfig:
    """Mixtral 8x7B configuration"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    num_experts: int = 8
    num_active_experts: int = 2
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768


class MixtralRouter:
    """
    Mixtral Top-2 Router
    
    Routes each token to 2 of 8 experts.
    Uses softmax gating with learned routing.
    """
    
    def __init__(self, hidden_size: int, num_experts: int, num_active: int = 2):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_active = num_active
        
        # Router weights
        self.gate = np.random.randn(hidden_size, num_experts) * 0.02
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route tokens to top-2 experts.
        
        Args:
            x: Input tensor (seq_len, hidden_size)
            
        Returns:
            expert_indices: (seq_len, 2)
            expert_weights: (seq_len, 2) - normalized weights
        """
        # Compute router logits
        logits = np.matmul(x, self.gate)  # (seq_len, num_experts)
        
        # Get top-2 indices
        top_k_indices = np.argsort(logits, axis=-1)[:, -self.num_active:]
        
        # Get top-2 logits and compute softmax weights
        top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits, axis=-1, keepdims=True))
        weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return top_k_indices, weights


class MixtralExpert:
    """
    Mixtral Expert (SwiGLU FFN)
    
    Each expert is a SwiGLU feed-forward network.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # SwiGLU projections
        self.gate_proj = np.random.randn(hidden_size, intermediate_size) * 0.02
        self.up_proj = np.random.randn(hidden_size, intermediate_size) * 0.02
        self.down_proj = np.random.randn(intermediate_size, hidden_size) * 0.02
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """SwiGLU forward pass"""
        gate = self.silu(np.matmul(x, self.gate_proj))
        up = np.matmul(x, self.up_proj)
        return np.matmul(gate * up, self.down_proj)
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation"""
        return x * (1 / (1 + np.exp(-x)))


class MixtralMoELayer:
    """
    Mixtral MoE Layer
    
    Key design choices:
    - 8 experts total
    - Top-2 routing (always use exactly 2 experts)
    - SwiGLU expert architecture
    - No auxiliary load balancing loss at inference
    """
    
    def __init__(self, config: MixtralConfig):
        self.config = config
        
        # Router
        self.router = MixtralRouter(
            config.hidden_size,
            config.num_experts,
            config.num_active_experts
        )
        
        # Experts
        self.experts = [
            MixtralExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ]
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through MoE layer.
        
        Each token is processed by exactly 2 experts,
        with outputs weighted by router probabilities.
        """
        seq_len = x.shape[0]
        
        # Route
        expert_indices, expert_weights = self.router.forward(x)
        
        # Compute output
        output = np.zeros_like(x)
        
        for token_idx in range(seq_len):
            for k in range(self.config.num_active_experts):
                expert_idx = expert_indices[token_idx, k]
                weight = expert_weights[token_idx, k]
                
                expert_out = self.experts[expert_idx](x[token_idx:token_idx+1])
                output[token_idx] += weight * expert_out.squeeze()
        
        # Compute routing statistics
        stats = self._compute_stats(expert_indices)
        
        return output, stats
    
    def _compute_stats(self, expert_indices: np.ndarray) -> Dict:
        """Compute routing statistics"""
        # Count expert usage
        flat_indices = expert_indices.flatten()
        counts = np.bincount(flat_indices, minlength=self.config.num_experts)
        
        return {
            'expert_counts': counts,
            'most_used': np.argmax(counts),
            'least_used': np.argmin(counts),
            'load_std': np.std(counts)
        }
    
    def count_parameters(self) -> Dict:
        """Count total and active parameters"""
        # Expert parameters
        expert_params = (
            self.config.hidden_size * self.config.intermediate_size +  # gate_proj
            self.config.hidden_size * self.config.intermediate_size +  # up_proj
            self.config.intermediate_size * self.config.hidden_size    # down_proj
        )
        
        total_expert_params = expert_params * self.config.num_experts
        active_expert_params = expert_params * self.config.num_active_experts
        
        # Router parameters
        router_params = self.config.hidden_size * self.config.num_experts
        
        return {
            'total_params': total_expert_params + router_params,
            'active_params': active_expert_params + router_params,
            'compression': (total_expert_params + router_params) / (active_expert_params + router_params)
        }


def demonstrate_mixtral():
    """Demonstrate Mixtral MoE"""
    print("=" * 70)
    print("Mixtral of Experts (Mistral AI, 2024)")
    print("=" * 70)
    
    config = MixtralConfig(
        hidden_size=64,  # Small for demo
        intermediate_size=256,
        num_experts=8,
        num_active_experts=2
    )
    
    layer = MixtralMoELayer(config)
    
    print("\n📊 Mixtral Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Total experts: {config.num_experts}")
    print(f"  Active experts per token: {config.num_active_experts}")
    
    # Sample input
    seq_len = 16
    x = np.random.randn(seq_len, config.hidden_size)
    
    # Forward pass
    output, stats = layer(x)
    
    print(f"\n📊 Forward Pass Results:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expert usage: {stats['expert_counts']}")
    print(f"  Most used expert: {stats['most_used']}")
    print(f"  Load std: {stats['load_std']:.2f}")
    
    # Parameter efficiency
    param_info = layer.count_parameters()
    print(f"\n📊 Parameter Efficiency:")
    print(f"  Total parameters: {param_info['total_params']:,}")
    print(f"  Active per token: {param_info['active_params']:,}")
    print(f"  Compression ratio: {param_info['compression']:.1f}x")


def compare_mixtral_variants():
    """Compare Mixtral model variants"""
    print("\n" + "=" * 70)
    print("📊 Mixtral Model Variants")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ Model             │ Total Params │ Active Params │ Experts │ Top-K │
├─────────────────────────────────────────────────────────────────────┤
│ Mixtral 8x7B      │ 46.7B        │ 12.9B         │ 8       │ 2     │
│ Mixtral 8x22B     │ 176B         │ 39B           │ 8       │ 2     │
│                   │              │               │         │       │
│ Comparison:       │              │               │         │       │
│ LLaMA-2 70B       │ 70B          │ 70B (dense)   │ -       │ -     │
│ Mistral 7B        │ 7B           │ 7B (dense)    │ -       │ -     │
└─────────────────────────────────────────────────────────────────────┘

Key Insight:
    Mixtral 8x7B with 46.7B total parameters
    runs at ~13B active parameters
    but matches or beats LLaMA-2 70B!
    
Inference Cost:
    - Mixtral 8x7B: Similar to 13B dense model
    - Quality: Similar to 70B dense model
    - Cost savings: ~5x for equivalent quality
""")


def explain_mixtral_innovations():
    """Explain what makes Mixtral special"""
    print("\n" + "=" * 70)
    print("📚 Mixtral Key Innovations")
    print("=" * 70)
    
    print("""
1. Open-Weight MoE:
   - First truly open high-quality MoE model
   - Weights publicly available
   - Enabled research and deployment

2. Practical Top-2 Design:
   - Simple: exactly 2 experts per token
   - No complex routing or capacity limits
   - Easy to implement and optimize

3. SwiGLU Experts:
   - Modern architecture (like LLaMA)
   - Each expert is a full SwiGLU FFN
   - High quality per expert

4. Quality-Cost Trade-off:
   - Dense 70B quality at 13B inference cost
   - Makes large models practical
   - Game changer for deployment

5. Sliding Window Attention:
   - 4k sliding window
   - Efficient for long sequences
   - Can handle 32k context

6. No Auxiliary Loss at Inference:
   - Load balancing only during training
   - Clean inference without overhead
   - Simpler deployment
""")


def demo():
    """Main demonstration"""
    demonstrate_mixtral()
    compare_mixtral_variants()
    explain_mixtral_innovations()
    
    print("\n" + "=" * 70)
    print("📚 Why Mixtral Matters")
    print("=" * 70)
    print("""
    1. Proved MoE Quality: Sparse models can match dense quality
    
    2. Open Weights: Community can study, deploy, fine-tune
    
    3. Practical Design: Simple top-2 routing that just works
    
    4. Cost Efficiency: 5x cheaper inference for same quality
    
    5. Template for Industry: Influenced Qwen, DeepSeek, etc.
    
    Key Takeaway:
        You can have your cake and eat it too.
        Large capacity (46.7B) with small compute (13B).
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
