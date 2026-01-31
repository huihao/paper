#!/usr/bin/env python3
"""
LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
Implementation of key architectural components: RMSNorm, SwiGLU, and RoPE.

Paper: https://arxiv.org/abs/2302.13971
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class LLaMAConfig:
    """LLaMA model configuration"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    @classmethod
    def llama_7b(cls):
        return cls()
    
    @classmethod
    def llama_13b(cls):
        return cls(
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=40,
            num_attention_heads=40
        )
    
    @classmethod
    def llama_65b(cls):
        return cls(
            hidden_size=8192,
            intermediate_size=22016,
            num_hidden_layers=80,
            num_attention_heads=64
        )


class RMSNorm:
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    Key difference from LayerNorm:
    - No mean centering (only variance normalization)
    - Simpler and slightly more efficient
    - Works just as well in practice
    
    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²))
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(hidden_size)  # Learnable scale (γ)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RMSNorm
        
        Args:
            x: Input tensor (..., hidden_size)
            
        Returns:
            Normalized tensor
        """
        # Compute RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        
        # Normalize and scale
        return (x / rms) * self.weight


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Sigmoid Linear Unit) activation, also known as Swish
    
    SiLU(x) = x * sigmoid(x)
    
    Used in LLaMA's SwiGLU variant
    """
    return x * (1 / (1 + np.exp(-x)))


class SwiGLU:
    """
    SwiGLU Activation (Swish-Gated Linear Unit)
    
    Key innovation:
    - Combines gating with SiLU activation
    - SwiGLU(x, W, V, W2) = (SiLU(xW) ⊙ xV) W2
    - Shown to improve performance in PaLM and LLaMA
    
    Note: This uses 3 weight matrices instead of 2 for standard FFN
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Three projections: gate, up, down
        self.gate_proj = np.random.randn(hidden_size, intermediate_size) * 0.02
        self.up_proj = np.random.randn(hidden_size, intermediate_size) * 0.02
        self.down_proj = np.random.randn(intermediate_size, hidden_size) * 0.02
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply SwiGLU
        
        SwiGLU(x) = (SiLU(x @ gate_proj) ⊙ (x @ up_proj)) @ down_proj
        """
        gate = silu(np.matmul(x, self.gate_proj))
        up = np.matmul(x, self.up_proj)
        return np.matmul(gate * up, self.down_proj)


class RotaryPositionalEmbedding:
    """
    Rotary Position Embedding (RoPE)
    
    Key innovations:
    - Encodes position through rotation of query/key vectors
    - Relative position information is naturally incorporated
    - Excellent extrapolation to longer sequences
    
    For position m and dimension i:
    - cos(m * θ_i) and sin(m * θ_i)
    where θ_i = 10000^(-2i/d)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        self.freqs = self._compute_freqs()
        self.cos_cached, self.sin_cached = self._compute_cache()
    
    def _compute_freqs(self) -> np.ndarray:
        """Compute rotation frequencies"""
        # θ_i = 10000^(-2i/d) for i = 0, 1, ..., d/2-1
        i = np.arange(0, self.dim, 2)
        freqs = 1.0 / (self.theta ** (i / self.dim))
        return freqs
    
    def _compute_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute cos and sin for all positions"""
        positions = np.arange(self.max_seq_len)
        
        # Shape: (max_seq_len, dim/2)
        angles = np.outer(positions, self.freqs)
        
        # Duplicate for pairs: shape becomes (max_seq_len, dim)
        cos_cached = np.cos(np.repeat(angles, 2, axis=-1))
        sin_cached = np.sin(np.repeat(angles, 2, axis=-1))
        
        return cos_cached, sin_cached
    
    def rotate_half(self, x: np.ndarray) -> np.ndarray:
        """Rotate half the hidden dimensions"""
        x1 = x[..., ::2]   # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Interleave -x2 and x1
        rotated = np.empty_like(x)
        rotated[..., ::2] = -x2
        rotated[..., 1::2] = x1
        
        return rotated
    
    def __call__(self, x: np.ndarray, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply rotary embeddings to input
        
        Args:
            x: Input tensor (batch, seq_len, dim) or (seq_len, dim)
            positions: Position indices (optional, defaults to 0, 1, 2, ...)
            
        Returns:
            Rotated tensor
        """
        if x.ndim == 2:
            seq_len = x.shape[0]
        else:
            seq_len = x.shape[1]
        
        if positions is None:
            positions = np.arange(seq_len)
        
        # Get cached values for these positions
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        return x * cos + self.rotate_half(x) * sin


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LLaMAAttention:
    """
    LLaMA Attention with RoPE
    
    Key features:
    - Rotary Position Embeddings (RoPE)
    - Multi-head attention
    - Causal masking
    """
    
    def __init__(self, config: LLaMAConfig):
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Projections
        self.q_proj = np.random.randn(config.hidden_size, config.hidden_size) * 0.02
        self.k_proj = np.random.randn(config.hidden_size, config.hidden_size) * 0.02
        self.v_proj = np.random.randn(config.hidden_size, config.hidden_size) * 0.02
        self.o_proj = np.random.randn(config.hidden_size, config.hidden_size) * 0.02
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(
            self.head_dim, 
            config.max_position_embeddings,
            config.rope_theta
        )
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply attention with RoPE
        
        Args:
            x: Input (seq_len, hidden_size)
            
        Returns:
            Output (seq_len, hidden_size)
        """
        seq_len = x.shape[0]
        
        # Project Q, K, V
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # Reshape for multi-head (simplified, treating as single head here)
        # In practice, reshape to (seq_len, num_heads, head_dim)
        
        # Apply RoPE to Q and K
        Q = self.rope(Q)
        K = self.rope(K)
        
        # Attention scores
        scores = np.matmul(Q, K.T) / np.sqrt(self.head_dim)
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask
        
        # Softmax and weighted sum
        attn_weights = softmax(scores, axis=-1)
        output = np.matmul(attn_weights, V)
        
        # Output projection
        return np.matmul(output, self.o_proj)


class LLaMABlock:
    """
    Single LLaMA Transformer Block
    
    Architecture (pre-norm):
    1. RMSNorm → Attention → Add
    2. RMSNorm → SwiGLU FFN → Add
    """
    
    def __init__(self, config: LLaMAConfig):
        self.attention = LLaMAAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through LLaMA block"""
        # Attention with residual
        normed = self.input_layernorm(x)
        attn_out = self.attention(normed)
        x = x + attn_out
        
        # MLP with residual
        normed = self.post_attention_layernorm(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out
        
        return x


def demo_rmsnorm():
    """Demonstrate RMSNorm"""
    print("\n📊 RMSNorm Demonstration")
    print("-" * 40)
    
    hidden_size = 8
    rmsnorm = RMSNorm(hidden_size)
    
    # Create input
    x = np.random.randn(4, hidden_size)
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    
    # Apply RMSNorm
    output = rmsnorm(x)
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Verify RMS is approximately 1
    rms = np.sqrt(np.mean(output ** 2, axis=-1))
    print(f"RMS of each row: {rms.round(4)}")
    
    print("\n💡 Key insight: RMSNorm normalizes by root mean square, not mean+std")


def demo_swiglu():
    """Demonstrate SwiGLU"""
    print("\n📊 SwiGLU Demonstration")
    print("-" * 40)
    
    hidden_size = 8
    intermediate_size = 24
    swiglu = SwiGLU(hidden_size, intermediate_size)
    
    x = np.random.randn(4, hidden_size)
    print(f"Input shape: {x.shape}")
    
    output = swiglu(x)
    print(f"Output shape: {output.shape}")
    
    print("\n💡 SwiGLU formula: (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down")
    print("   Uses 3 projections instead of 2 in standard FFN")
    print("   Intermediate size is 2/3 of what you'd use with standard FFN")


def demo_rope():
    """Demonstrate Rotary Position Embeddings"""
    print("\n📊 Rotary Position Embeddings (RoPE) Demonstration")
    print("-" * 40)
    
    dim = 8
    rope = RotaryPositionalEmbedding(dim, max_seq_len=100)
    
    # Create Q and K vectors at different positions
    q = np.random.randn(1, dim)
    k = np.random.randn(1, dim)
    
    print(f"Query shape: {q.shape}, Key shape: {k.shape}")
    
    # Rotate at position 0
    q_pos0 = rope(q, positions=np.array([0]))
    k_pos0 = rope(k, positions=np.array([0]))
    
    # Rotate at position 5
    q_pos5 = rope(q, positions=np.array([5]))
    k_pos5 = rope(k, positions=np.array([5]))
    
    # The key insight: relative position is preserved
    print("\n💡 RoPE encodes position through rotation")
    print("   - Dot product encodes relative position")
    print("   - Extrapolates well to longer sequences")
    print("   - No need for separate position embeddings")
    
    # Show that rotation changes with position
    print(f"\nVector at pos 0: {q_pos0[0, :4].round(3)}")
    print(f"Vector at pos 5: {q_pos5[0, :4].round(3)}")


def demo_llama_block():
    """Demonstrate full LLaMA block"""
    print("\n📊 LLaMA Block Demonstration")
    print("-" * 40)
    
    # Small config for demo
    config = LLaMAConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        max_position_embeddings=128
    )
    
    block = LLaMABlock(config)
    
    seq_len = 10
    x = np.random.randn(seq_len, config.hidden_size)
    
    print(f"Input shape: {x.shape}")
    
    output = block(x)
    print(f"Output shape: {output.shape}")
    
    print("\n💡 LLaMA block combines:")
    print("   1. RMSNorm (simpler than LayerNorm)")
    print("   2. Attention with RoPE (rotary position)")
    print("   3. SwiGLU FFN (gated activation)")


def demo():
    """Main demonstration"""
    print("=" * 80)
    print("LLaMA: Open and Efficient Foundation Language Models")
    print("Touvron et al., 2023")
    print("=" * 80)
    
    print("\n📚 LLaMA's Key Architectural Innovations:")
    print("-" * 40)
    print("""
    1. RMSNorm: Simpler normalization (no mean centering)
    2. SwiGLU: Gated activation function in FFN
    3. RoPE: Rotary Position Embeddings for better extrapolation
    4. Pre-norm: Normalization before each sub-layer
    5. No bias: Removes bias terms for efficiency
    """)
    
    demo_rmsnorm()
    demo_swiglu()
    demo_rope()
    demo_llama_block()
    
    print("\n" + "=" * 80)
    print("📊 LLaMA Model Sizes")
    print("-" * 40)
    
    configs = [
        ("LLaMA-7B", LLaMAConfig.llama_7b()),
        ("LLaMA-13B", LLaMAConfig.llama_13b()),
        ("LLaMA-65B", LLaMAConfig.llama_65b()),
    ]
    
    print(f"{'Model':<15} {'Hidden':<10} {'Layers':<10} {'Heads':<10} {'FFN':<10}")
    print("-" * 55)
    
    for name, config in configs:
        print(f"{name:<15} {config.hidden_size:<10} {config.num_hidden_layers:<10} "
              f"{config.num_attention_heads:<10} {config.intermediate_size:<10}")
    
    print("\n📚 Training Details:")
    print("-" * 40)
    print("""
    - Trained on 1-1.4 trillion tokens
    - Open weights (though not fully open-source)
    - Efficient: 7B competitive with much larger models
    - Set new baseline for open-weight models
    """)
    
    print("\n📚 Impact:")
    print("-" * 40)
    print("""
    1. Triggered the open-weight era
    2. Architectural choices became the new default
    3. Enabled fine-tuning research (Alpaca, Vicuna, etc.)
    4. Proved open models can compete with closed ones
    """)
    print("=" * 80)


if __name__ == "__main__":
    demo()
