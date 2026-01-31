#!/usr/bin/env python3
"""
RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
Complete implementation of Rotary Position Embedding (RoPE).

Paper: https://arxiv.org/abs/2104.09864
"""

import numpy as np
from typing import Tuple, Optional
import math


class RotaryPositionEmbedding:
    """
    Rotary Position Embedding (RoPE)
    
    Key insight: Encode position through rotation in 2D subspaces.
    
    For a d-dimensional vector, we split it into d/2 pairs.
    Each pair (x_{2i}, x_{2i+1}) is rotated by angle θ_i * position.
    
    Benefits:
    1. Relative position is naturally encoded in attention
    2. Excellent extrapolation to longer sequences
    3. No additional parameters to learn
    4. Works with any attention mechanism
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 4096,
        base: float = 10000.0
    ):
        """
        Initialize RoPE
        
        Args:
            dim: Dimension of the embeddings (must be even)
            max_seq_len: Maximum sequence length to cache
            base: Base for frequency computation (default 10000)
        """
        assert dim % 2 == 0, "Dimension must be even"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies: θ_i = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
        self.inv_freq = inv_freq
        
        # Cache cos and sin for all positions
        self._build_cache()
    
    def _build_cache(self):
        """Precompute cos and sin for all positions"""
        positions = np.arange(self.max_seq_len)
        
        # Outer product: (seq_len,) × (dim/2,) → (seq_len, dim/2)
        freqs = np.outer(positions, self.inv_freq)
        
        # Create rotation matrices components
        # We need both cos and sin for each frequency
        self.cos_cache = np.cos(freqs)  # (seq_len, dim/2)
        self.sin_cache = np.sin(freqs)  # (seq_len, dim/2)
    
    def rotate_half(self, x: np.ndarray) -> np.ndarray:
        """
        Rotate half of the dimensions
        
        Split x into [x1, x2] and return [-x2, x1]
        This is the "rotation" operation in 2D.
        """
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        return np.concatenate([-x2, x1], axis=-1)
    
    def apply_rotary_pos_emb(
        self, 
        x: np.ndarray, 
        positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply rotary position embedding to input tensor.
        
        The rotation formula:
        [x_1, x_2] → [x_1 * cos(θ) - x_2 * sin(θ), x_1 * sin(θ) + x_2 * cos(θ)]
        
        Which can be rewritten as:
        x_rotated = x * cos + rotate_half(x) * sin
        
        Args:
            x: Input tensor (..., seq_len, dim)
            positions: Position indices (seq_len,), defaults to [0, 1, 2, ...]
            
        Returns:
            Rotated tensor with same shape as input
        """
        seq_len = x.shape[-2]
        
        if positions is None:
            positions = np.arange(seq_len)
        
        # Get cached cos and sin
        cos = self.cos_cache[positions]  # (seq_len, dim/2)
        sin = self.sin_cache[positions]  # (seq_len, dim/2)
        
        # Expand to full dimension by repeating each frequency twice
        cos = np.repeat(cos, 2, axis=-1)  # (seq_len, dim)
        sin = np.repeat(sin, 2, axis=-1)  # (seq_len, dim)
        
        # Apply rotation
        return x * cos + self.rotate_half(x) * sin


class RoPEAttention:
    """
    Multi-head attention with Rotary Position Embedding
    """
    
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 4096):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Projections
        self.W_q = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_k = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_v = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_o = np.random.randn(hidden_size, hidden_size) * 0.02
        
        # RoPE for each head
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
    
    def __call__(self, x: np.ndarray, causal: bool = True) -> np.ndarray:
        """
        Apply attention with RoPE
        
        Args:
            x: Input (seq_len, hidden_size)
            causal: Whether to use causal masking
            
        Returns:
            Output (seq_len, hidden_size)
        """
        seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = np.matmul(x, self.W_q)  # (seq_len, hidden_size)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim)
        V = V.reshape(seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to each head
        for h in range(self.num_heads):
            Q[:, h, :] = self.rope.apply_rotary_pos_emb(Q[:, h, :].reshape(seq_len, 1, self.head_dim)).squeeze(1)
            K[:, h, :] = self.rope.apply_rotary_pos_emb(K[:, h, :].reshape(seq_len, 1, self.head_dim)).squeeze(1)
        
        # Compute attention per head
        outputs = []
        for h in range(self.num_heads):
            q_h = Q[:, h, :]  # (seq_len, head_dim)
            k_h = K[:, h, :]
            v_h = V[:, h, :]
            
            # Attention scores
            scores = np.matmul(q_h, k_h.T) / np.sqrt(self.head_dim)
            
            # Causal mask
            if causal:
                mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
                scores = scores + mask
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            
            # Weighted sum
            output = np.matmul(weights, v_h)
            outputs.append(output)
        
        # Concatenate heads
        concat = np.concatenate(outputs, axis=-1)
        
        # Output projection
        return np.matmul(concat, self.W_o)


def visualize_rope_rotation():
    """Visualize how RoPE rotates vectors at different positions"""
    print("\n📊 Visualizing RoPE Rotation")
    print("-" * 50)
    
    # Create simple 2D example
    dim = 2
    rope = RotaryPositionEmbedding(dim, max_seq_len=100)
    
    # Initial vector pointing along x-axis
    v = np.array([[1.0, 0.0]])  # (1, 2)
    
    print(f"Initial vector: {v[0]}")
    print("\nRotated vectors at different positions:")
    
    for pos in [0, 1, 2, 5, 10, 20]:
        v_expanded = v.reshape(1, 1, dim)
        rotated = rope.apply_rotary_pos_emb(v_expanded, positions=np.array([pos]))
        rotated = rotated.reshape(-1)
        angle = np.arctan2(rotated[1], rotated[0]) * 180 / np.pi
        print(f"  Position {pos:2d}: [{rotated[0]:7.4f}, {rotated[1]:7.4f}] (angle: {angle:7.2f}°)")
    
    print("\n💡 Key insight: Position is encoded as a rotation angle!")


def demonstrate_relative_position():
    """Show that RoPE encodes relative position in dot products"""
    print("\n📊 Relative Position Encoding")
    print("-" * 50)
    
    dim = 8
    rope = RotaryPositionEmbedding(dim, max_seq_len=100)
    
    # Create two random vectors
    q = np.random.randn(1, 1, dim)
    k = np.random.randn(1, 1, dim)
    
    print("Dot product of Q and K at different relative positions:")
    print("(showing that relative position matters, not absolute)")
    
    # Test: dot(q@pos_i, k@pos_j) depends on (i-j)
    for i in range(0, 10, 2):
        for j in range(0, 10, 2):
            q_rot = rope.apply_rotary_pos_emb(q, positions=np.array([i]))
            k_rot = rope.apply_rotary_pos_emb(k, positions=np.array([j]))
            
            dot = np.sum(q_rot * k_rot)
            print(f"  q@{i} · k@{j} (rel={i-j:+2d}): {dot:8.4f}")
        print()
    
    print("💡 Vectors at the same relative distance have similar dot products!")


def compare_with_absolute_position():
    """Compare RoPE with absolute position embeddings"""
    print("\n📊 RoPE vs Absolute Position Embeddings")
    print("-" * 50)
    
    print("""
    Absolute Position Embeddings (original Transformer):
    - Add position embedding to token embedding: x + PE[pos]
    - Fixed or learned lookup table
    - Hard to extrapolate beyond training length
    - Position info can be "washed out" in deep networks
    
    Rotary Position Embeddings (RoPE):
    - Rotate Q and K by position-dependent angles
    - Relative position encoded in attention scores directly
    - Excellent extrapolation properties
    - Position info preserved through attention computation
    
    Mathematical difference:
    - Absolute: attention(x_i + p_i, x_j + p_j)
    - RoPE: attention(R(x_i, i), R(x_j, j)) where R is rotation
    
    In RoPE, the dot product of rotated vectors:
    R(x_i, i)^T · R(x_j, j) depends only on (x_i, x_j) and (i - j)
    """)


def demo_extrapolation():
    """Demonstrate RoPE's extrapolation capability"""
    print("\n📊 Extrapolation Capability")
    print("-" * 50)
    
    # RoPE with short training length
    train_length = 32
    rope = RotaryPositionEmbedding(dim=8, max_seq_len=train_length, base=10000.0)
    
    # Can easily extend to longer sequences
    rope_extended = RotaryPositionEmbedding(dim=8, max_seq_len=128, base=10000.0)
    
    print(f"Trained on sequences up to length: {train_length}")
    print(f"Testing extrapolation to length: 128")
    
    # Check that cos/sin values are consistent for positions within training range
    test_pos = 10
    original = (rope.cos_cache[test_pos], rope.sin_cache[test_pos])
    extended = (rope_extended.cos_cache[test_pos], rope_extended.sin_cache[test_pos])
    
    print(f"\nCos values at position {test_pos}:")
    print(f"  Original:  {original[0].round(4)}")
    print(f"  Extended:  {extended[0].round(4)}")
    print(f"  Match: {np.allclose(original[0], extended[0])}")
    
    print("\n💡 RoPE uses same rotation formula for all positions!")
    print("   No learned parameters → perfect extrapolation")


def demo():
    """Main demonstration"""
    print("=" * 80)
    print("RoFormer: Enhanced Transformer with Rotary Position Embedding")
    print("Su et al., 2021")
    print("=" * 80)
    
    print("\n📚 What is RoPE?")
    print("-" * 40)
    print("""
    Rotary Position Embedding encodes position by rotating
    the query and key vectors in 2D subspaces.
    
    Key formula:
    For position m and dimension pair (2i, 2i+1):
    
    [q_{2i}  ]     [cos(mθ_i)  -sin(mθ_i)] [q_{2i}  ]
    [q_{2i+1}]  =  [sin(mθ_i)   cos(mθ_i)] [q_{2i+1}]
    
    where θ_i = 10000^(-2i/d)
    
    This means each pair of dimensions is rotated by a position-dependent angle.
    """)
    
    visualize_rope_rotation()
    demonstrate_relative_position()
    compare_with_absolute_position()
    demo_extrapolation()
    
    print("\n" + "=" * 80)
    print("📚 Summary: Why RoPE Became Standard")
    print("=" * 80)
    print("""
    1. Encodes relative position naturally in attention
    2. No additional parameters to learn
    3. Excellent extrapolation to longer sequences
    4. Works seamlessly with any attention variant
    5. Computationally efficient
    
    Used in: LLaMA, Mistral, Qwen, Yi, and nearly all modern LLMs
    """)
    print("=" * 80)


if __name__ == "__main__":
    demo()
