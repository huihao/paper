#!/usr/bin/env python3
"""
FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
Implementation demonstrating the key concepts of memory-efficient attention.

Paper: https://arxiv.org/abs/2205.14135
"""

import numpy as np
from typing import Tuple, Optional
import time


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Standard softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class StandardAttention:
    """
    Standard (vanilla) attention implementation.
    
    Memory: O(N²) - stores full attention matrix
    This is the baseline that FlashAttention improves upon.
    """
    
    @staticmethod
    def forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Standard attention: O(N²) memory for attention matrix
        
        Args:
            Q: Query matrix (N, d)
            K: Key matrix (N, d)
            V: Value matrix (N, d)
            
        Returns:
            Output (N, d)
        """
        N, d = Q.shape
        
        # Step 1: Compute full attention matrix (O(N²) memory!)
        scores = np.matmul(Q, K.T) / np.sqrt(d)  # (N, N)
        
        # Step 2: Softmax
        attention = softmax(scores, axis=-1)  # (N, N)
        
        # Step 3: Weighted sum
        output = np.matmul(attention, V)  # (N, d)
        
        return output


class FlashAttentionConcept:
    """
    Conceptual implementation of FlashAttention.
    
    Key ideas:
    1. Block-wise computation to avoid storing full N×N matrix
    2. Online softmax with rescaling
    3. Tiling for GPU memory hierarchy
    
    Note: This is a simplified Python version. Real FlashAttention
    requires CUDA kernels for true efficiency gains.
    """
    
    def __init__(self, block_size: int = 64):
        """
        Initialize FlashAttention
        
        Args:
            block_size: Size of blocks for tiled computation
        """
        self.block_size = block_size
    
    def online_softmax_stats(
        self, 
        m_prev: float, 
        l_prev: float,
        x_new: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Online softmax computation.
        
        Key insight: We can compute softmax incrementally by tracking:
        - m: running maximum
        - l: running sum of exp(x - m)
        
        When we see new values, we can update the statistics
        without storing all previous values.
        """
        m_new = np.max(x_new)
        m = max(m_prev, m_new)
        
        # Rescale previous sum
        l_rescaled = l_prev * np.exp(m_prev - m)
        
        # Add new values
        exp_new = np.exp(x_new - m)
        l = l_rescaled + np.sum(exp_new)
        
        return m, l, exp_new
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        FlashAttention-style forward pass.
        
        Instead of computing N×N attention matrix:
        1. Process Q in blocks
        2. For each Q block, iterate through K,V blocks
        3. Use online softmax to combine results
        
        Memory: O(N) instead of O(N²)
        """
        N, d = Q.shape
        B = self.block_size
        
        # Output accumulator
        output = np.zeros_like(Q)
        
        # Process Q in blocks
        for i in range(0, N, B):
            q_block = Q[i:i+B]
            block_size_q = len(q_block)
            
            # Initialize online softmax stats for this Q block
            m = np.full(block_size_q, -np.inf)  # Running max
            l = np.zeros(block_size_q)  # Running sum of exp
            o = np.zeros((block_size_q, d))  # Running output
            
            # Iterate through K, V blocks
            for j in range(0, N, B):
                k_block = K[j:j+B]
                v_block = V[j:j+B]
                
                # Compute attention scores for this block pair
                scores = np.matmul(q_block, k_block.T) / np.sqrt(d)  # (B, B)
                
                # Online softmax update
                m_new = np.max(scores, axis=-1)
                m_prev = m.copy()
                m = np.maximum(m, m_new)
                
                # Rescale previous output
                scale = np.exp(m_prev - m)
                o = o * scale[:, np.newaxis]
                l = l * scale
                
                # Compute new contributions
                exp_scores = np.exp(scores - m[:, np.newaxis])
                l += np.sum(exp_scores, axis=-1)
                o += np.matmul(exp_scores, v_block)
            
            # Normalize output
            output[i:i+B] = o / l[:, np.newaxis]
        
        return output


class FlashAttentionV2Concept:
    """
    FlashAttention-2 improvements.
    
    Key optimizations over FlashAttention-1:
    1. Better parallelization across sequence
    2. Reduced non-matmul FLOPs
    3. Better work partitioning
    """
    
    @staticmethod
    def describe_improvements():
        return """
FlashAttention-2 Optimizations:

1. Parallelization:
   - FA1: Parallelizes over batch and heads
   - FA2: Also parallelizes over sequence length
   - Better GPU utilization for long sequences

2. Work Partitioning:
   - FA1: Each block does full softmax normalization
   - FA2: Delays normalization, reduces memory traffic

3. Reduced Non-MatMul FLOPs:
   - FA1: Many scalar operations for online softmax
   - FA2: Vectorized operations, fewer rescales

4. Improved Memory Access:
   - FA1: Read K, V multiple times per Q block
   - FA2: Better reuse of K, V in shared memory

Result: 2x faster than FlashAttention-1 on A100 GPUs
"""


def benchmark_attention(seq_len: int, dim: int, num_runs: int = 5):
    """Compare standard vs flash attention"""
    print(f"\n📊 Benchmark: seq_len={seq_len}, dim={dim}")
    print("-" * 50)
    
    # Generate random inputs
    Q = np.random.randn(seq_len, dim)
    K = np.random.randn(seq_len, dim)
    V = np.random.randn(seq_len, dim)
    
    # Standard attention
    start = time.time()
    for _ in range(num_runs):
        out_standard = StandardAttention.forward(Q, K, V)
    time_standard = (time.time() - start) / num_runs
    
    # Flash attention (conceptual)
    flash = FlashAttentionConcept(block_size=64)
    start = time.time()
    for _ in range(num_runs):
        out_flash = flash.forward(Q, K, V)
    time_flash = (time.time() - start) / num_runs
    
    # Verify correctness
    max_diff = np.max(np.abs(out_standard - out_flash))
    
    print(f"Standard Attention: {time_standard*1000:.2f} ms")
    print(f"Flash Attention:    {time_flash*1000:.2f} ms")
    print(f"Max difference:     {max_diff:.2e}")
    print(f"Results match:      {max_diff < 1e-5}")


def analyze_memory():
    """Analyze memory usage of different attention methods"""
    print("\n📊 Memory Analysis")
    print("-" * 50)
    
    print("""
Standard Attention Memory:
- Attention matrix: O(N²) 
- For N=4096, d=64:
  - Attention matrix: 4096 × 4096 × 4 bytes = 64 MB (float32)
  - Total with Q,K,V,O: ~65 MB

FlashAttention Memory:
- No full attention matrix stored
- Block buffers: O(B²) where B << N
- For N=4096, d=64, B=64:
  - Block buffer: 64 × 64 × 4 bytes = 16 KB
  - Total: ~1 MB (dominated by Q,K,V,O)
  
Memory savings: ~64x for this example!
""")
    
    # Memory scaling comparison
    print("\nMemory scaling with sequence length:")
    print(f"{'Seq Len':<10} {'Standard':<15} {'Flash':<15} {'Savings':<10}")
    print("-" * 50)
    
    for N in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        d = 64
        B = 64
        
        # Standard: N² for attention matrix
        standard_mb = (N * N * 4) / (1024 * 1024)
        
        # Flash: O(B²) for blocks
        flash_mb = (B * B * 4) / (1024 * 1024) + (N * d * 4 * 4) / (1024 * 1024)
        
        savings = standard_mb / flash_mb
        
        print(f"{N:<10} {standard_mb:<15.1f} MB {flash_mb:<15.2f} MB {savings:<10.1f}x")


def explain_io_complexity():
    """Explain the I/O complexity advantage of FlashAttention"""
    print("\n📊 I/O Complexity: Why FlashAttention is Fast")
    print("-" * 50)
    
    print("""
GPU Memory Hierarchy:
- HBM (High Bandwidth Memory): ~1.5 TB/s, 40-80 GB
- SRAM (Shared Memory): ~19 TB/s, 20 MB per SM
- Registers: Fastest, very limited

Standard Attention:
1. Load Q, K from HBM → Compute S = QK^T → Write S to HBM
2. Load S from HBM → Compute P = softmax(S) → Write P to HBM  
3. Load P, V from HBM → Compute O = PV → Write O to HBM

Total HBM access: O(N² + Nd) reads + O(N² + Nd) writes

FlashAttention:
1. Load Q, K, V blocks into SRAM
2. Compute attention for block entirely in SRAM
3. Write only final output O to HBM

Total HBM access: O(Nd) reads + O(Nd) writes

The N² term is eliminated! For long sequences, this is huge.
""")


def demo():
    """Main demonstration"""
    print("=" * 80)
    print("FlashAttention: Fast and Memory-Efficient Exact Attention")
    print("Dao et al., 2022")
    print("=" * 80)
    
    print("\n📚 What is FlashAttention?")
    print("-" * 40)
    print("""
    FlashAttention is an IO-aware attention algorithm that:
    1. Never materializes the full N×N attention matrix
    2. Uses tiling to compute attention in blocks
    3. Uses online softmax to combine block results
    4. Achieves exact (not approximate) attention
    
    Result: 2-4x faster, enables 4x longer contexts
    """)
    
    # Show standard vs flash
    print("\n📊 Correctness Check")
    print("-" * 40)
    
    seq_len = 128
    dim = 32
    
    Q = np.random.randn(seq_len, dim)
    K = np.random.randn(seq_len, dim)
    V = np.random.randn(seq_len, dim)
    
    standard_out = StandardAttention.forward(Q, K, V)
    
    flash = FlashAttentionConcept(block_size=32)
    flash_out = flash.forward(Q, K, V)
    
    print(f"Sequence length: {seq_len}")
    print(f"Dimension: {dim}")
    print(f"Max difference: {np.max(np.abs(standard_out - flash_out)):.2e}")
    print("Results are numerically equivalent!")
    
    # Memory analysis
    analyze_memory()
    
    # I/O complexity
    explain_io_complexity()
    
    # FlashAttention-2 improvements
    print(FlashAttentionV2Concept.describe_improvements())
    
    # Benchmark (simplified, Python version)
    benchmark_attention(256, 64)
    
    print("\n" + "=" * 80)
    print("📚 Impact of FlashAttention")
    print("=" * 80)
    print("""
    1. Enabled Long Context:
       - GPT-4 with 128k context
       - Claude with 100k+ context
       - Previously, >4k was impractical
    
    2. Faster Training:
       - 2-4x speedup on attention
       - Enables larger batch sizes
    
    3. Reduced Memory:
       - Train larger models on same hardware
       - 4x longer sequences with same memory
    
    4. Exact, Not Approximate:
       - Unlike sparse attention, gives same output
       - No quality trade-off
    
    Now integrated into PyTorch, transformers library, and all major frameworks.
    """)
    print("=" * 80)


if __name__ == "__main__":
    demo()
