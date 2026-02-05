#!/usr/bin/env python3
"""
The Illustrated Transformer (Jay Alammar, 2018)
Educational implementation with step-by-step visualization of tensor flow.

This implementation focuses on understanding how data flows through 
the transformer, with detailed print statements and visualizations.

Blog post: https://jalammar.github.io/illustrated-transformer/
"""

import numpy as np
from typing import List, Tuple, Optional


def print_tensor_info(name: str, tensor: np.ndarray, show_values: bool = False):
    """Print tensor information for visualization"""
    print(f"\n📊 {name}")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Min: {tensor.min():.4f}, Max: {tensor.max():.4f}, Mean: {tensor.mean():.4f}")
    if show_values and tensor.size <= 16:
        print(f"   Values: {tensor.flatten()[:16].round(3)}")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class IllustratedEmbedding:
    """
    Word Embedding Layer with Visualization
    
    Converts token IDs to dense vectors.
    Each word is represented as a vector in a high-dimensional space.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        print("\n🔧 Creating Embedding Layer")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"   Embedding dimension: {embedding_dim}")
        
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.02
        print(f"   Embedding matrix shape: {self.embedding_matrix.shape}")
    
    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embeddings for token IDs"""
        print("\n📥 Embedding Lookup")
        print(f"   Input token IDs: {token_ids}")
        
        embeddings = self.embedding_matrix[token_ids]
        print_tensor_info("Embeddings", embeddings)
        
        return embeddings


class IllustratedPositionalEncoding:
    """
    Positional Encoding with Visualization
    
    Adds position information to embeddings using sine and cosine functions.
    This allows the model to understand word order.
    """
    
    def __init__(self, d_model: int, max_len: int = 100):
        print("\n🔧 Creating Positional Encoding")
        print(f"   Model dimension: {d_model}")
        print(f"   Max sequence length: {max_len}")
        
        self.pe = self._create_encoding(d_model, max_len)
        print(f"   Positional encoding shape: {self.pe.shape}")
    
    def _create_encoding(self, d_model: int, max_len: int) -> np.ndarray:
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input embeddings"""
        seq_len = x.shape[0] if x.ndim == 2 else x.shape[1]
        print(f"\n📍 Adding Positional Encoding (seq_len={seq_len})")
        
        pos_enc = self.pe[:seq_len]
        result = x + pos_enc
        
        print_tensor_info("After positional encoding", result)
        return result


def illustrated_attention_step_by_step(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    step_name: str = "Attention"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention with step-by-step visualization
    
    This function shows each step of the attention mechanism:
    1. Query-Key similarity (dot product)
    2. Scaling by sqrt(d_k)
    3. Softmax to get attention weights
    4. Weighted sum of values
    """
    print(f"\n{'='*60}")
    print(f"🔍 {step_name}: Step-by-Step Attention Computation")
    print(f"{'='*60}")
    
    d_k = query.shape[-1]
    print(f"\n📌 Input shapes:")
    print(f"   Query (Q): {query.shape}")
    print(f"   Key (K): {key.shape}")
    print(f"   Value (V): {value.shape}")
    print(f"   d_k (dimension): {d_k}")
    
    # Step 1: Compute Q @ K^T
    print(f"\n📐 Step 1: Compute Q @ K^T (dot product similarity)")
    scores = np.matmul(query, key.T)
    print(f"   Score matrix shape: {scores.shape}")
    print(f"   Raw scores:\n{scores.round(3)}")
    
    # Step 2: Scale by sqrt(d_k)
    print(f"\n📏 Step 2: Scale by √d_k = √{d_k} ≈ {np.sqrt(d_k):.3f}")
    scaled_scores = scores / np.sqrt(d_k)
    print(f"   Scaled scores:\n{scaled_scores.round(3)}")
    
    # Step 3: Apply softmax
    print(f"\n🎯 Step 3: Apply softmax to get attention weights")
    attention_weights = softmax(scaled_scores, axis=-1)
    print(f"   Attention weights (each row sums to 1):")
    print(f"   {attention_weights.round(3)}")
    print(f"   Row sums: {attention_weights.sum(axis=-1).round(3)}")
    
    # Step 4: Weighted sum of values
    print(f"\n💫 Step 4: Compute weighted sum of values")
    output = np.matmul(attention_weights, value)
    print(f"   Output shape: {output.shape}")
    print_tensor_info("Attention output", output)
    
    return output, attention_weights


class IllustratedMultiHeadAttention:
    """
    Multi-Head Attention with detailed visualization
    
    Splits the attention into multiple "heads" that can each focus on
    different aspects of the input (e.g., syntax, semantics, position).
    """
    
    def __init__(self, d_model: int, num_heads: int):
        print(f"\n🔧 Creating Multi-Head Attention")
        print(f"   Model dimension: {d_model}")
        print(f"   Number of heads: {num_heads}")
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        print(f"   Dimension per head: {self.d_k}")
        
        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def __call__(self, x: np.ndarray, show_all_heads: bool = False) -> np.ndarray:
        """Apply multi-head attention"""
        print(f"\n{'='*60}")
        print(f"🎭 Multi-Head Attention ({self.num_heads} heads)")
        print(f"{'='*60}")
        
        seq_len = x.shape[0]
        print(f"\n📥 Input shape: {x.shape}")
        
        # Project to Q, K, V
        print(f"\n📊 Step 1: Project input to Q, K, V")
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        print(f"   Q, K, V shapes: {Q.shape}")
        
        # Split into heads
        print(f"\n✂️ Step 2: Split into {self.num_heads} heads")
        Q_heads = Q.reshape(seq_len, self.num_heads, self.d_k)
        K_heads = K.reshape(seq_len, self.num_heads, self.d_k)
        V_heads = V.reshape(seq_len, self.num_heads, self.d_k)
        print(f"   Per-head shape: {Q_heads[:, 0, :].shape}")
        
        # Apply attention to each head
        print(f"\n🔄 Step 3: Apply attention to each head")
        head_outputs = []
        
        for h in range(self.num_heads):
            if show_all_heads or h == 0:
                output, weights = illustrated_attention_step_by_step(
                    Q_heads[:, h, :],
                    K_heads[:, h, :],
                    V_heads[:, h, :],
                    f"Head {h+1}"
                )
                if h == 0 and self.num_heads > 1 and not show_all_heads:
                    print(f"\n   (Showing only Head 1 for brevity. Set show_all_heads=True to see all.)")
            else:
                scores = np.matmul(Q_heads[:, h, :], K_heads[:, h, :].T) / np.sqrt(self.d_k)
                weights = softmax(scores, axis=-1)
                output = np.matmul(weights, V_heads[:, h, :])
            
            head_outputs.append(output)
        
        # Concatenate heads
        print(f"\n🔗 Step 4: Concatenate all heads")
        concat = np.concatenate(head_outputs, axis=-1)
        print(f"   Concatenated shape: {concat.shape}")
        
        # Final projection
        print(f"\n📤 Step 5: Final linear projection")
        output = np.matmul(concat, self.W_o)
        print_tensor_info("Multi-head attention output", output)
        
        return output


class IllustratedFeedForward:
    """
    Position-wise Feed-Forward Network with visualization
    
    Applied to each position independently:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int):
        print(f"\n🔧 Creating Feed-Forward Network")
        print(f"   Input/Output dimension: {d_model}")
        print(f"   Hidden dimension: {d_ff}")
        
        self.W_1 = np.random.randn(d_model, d_ff) * 0.02
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * 0.02
        self.b_2 = np.zeros(d_model)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply feed-forward network"""
        print(f"\n{'='*60}")
        print(f"🔄 Feed-Forward Network")
        print(f"{'='*60}")
        
        print(f"\n📥 Input shape: {x.shape}")
        
        # First linear + ReLU
        print(f"\n📊 Step 1: Linear transform + ReLU")
        hidden = np.matmul(x, self.W_1) + self.b_1
        print(f"   After W_1: {hidden.shape}")
        
        hidden = np.maximum(0, hidden)  # ReLU
        print(f"   After ReLU: {hidden.shape}")
        print(f"   ReLU activated {(hidden > 0).sum()}/{hidden.size} neurons")
        
        # Second linear
        print(f"\n📊 Step 2: Second linear transform")
        output = np.matmul(hidden, self.W_2) + self.b_2
        print_tensor_info("Feed-forward output", output)
        
        return output


class IllustratedTransformerBlock:
    """
    Complete Transformer Block with visualization
    
    Combines:
    1. Multi-Head Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        print(f"\n{'='*60}")
        print(f"🏗️ Creating Transformer Block")
        print(f"{'='*60}")
        
        self.attention = IllustratedMultiHeadAttention(d_model, num_heads)
        self.feed_forward = IllustratedFeedForward(d_model, d_ff)
        self.d_model = d_model
    
    def layer_norm(self, x: np.ndarray, name: str) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        normed = (x - mean) / std
        print(f"\n📏 Layer Norm ({name}): mean={x.mean():.4f} → 0, std={x.std():.4f} → 1")
        return normed
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Process through transformer block"""
        print(f"\n{'#'*60}")
        print(f"🔲 Processing Transformer Block")
        print(f"{'#'*60}")
        
        # Multi-head attention
        attn_out = self.attention(x)
        
        # Residual connection + Layer Norm
        print(f"\n➕ Residual Connection 1: x + attention(x)")
        x = x + attn_out
        x = self.layer_norm(x, "after attention")
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        
        # Residual connection + Layer Norm
        print(f"\n➕ Residual Connection 2: x + feedforward(x)")
        x = x + ff_out
        x = self.layer_norm(x, "after feed-forward")
        
        print_tensor_info("Block output", x)
        return x


def illustrate_full_transformer_flow():
    """
    Complete illustration of data flow through a Transformer
    """
    print("\n" + "=" * 80)
    print("📚 THE ILLUSTRATED TRANSFORMER - Complete Data Flow")
    print("=" * 80)
    
    # Configuration
    vocab_size = 100
    d_model = 16
    num_heads = 4
    d_ff = 64
    
    print("\n📋 Configuration:")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Model dimension: {d_model}")
    print(f"   Number of attention heads: {num_heads}")
    print(f"   Feed-forward hidden dimension: {d_ff}")
    
    # Create components
    print("\n" + "-" * 40)
    print("Step 0: Creating model components")
    print("-" * 40)
    
    embedding = IllustratedEmbedding(vocab_size, d_model)
    pos_encoding = IllustratedPositionalEncoding(d_model)
    transformer_block = IllustratedTransformerBlock(d_model, num_heads, d_ff)
    
    # Input sequence
    print("\n" + "-" * 40)
    print("Step 1: Input Processing")
    print("-" * 40)
    
    # Simulate token IDs for "The cat sat on the mat"
    token_ids = np.array([5, 12, 45, 78, 5, 91])
    print(f"\n📝 Input sentence (token IDs): {token_ids}")
    print(f"   (Simulating: 'The cat sat on the mat')")
    
    # Get embeddings
    embeddings = embedding(token_ids)
    
    # Add positional encoding
    print("\n" + "-" * 40)
    print("Step 2: Add Positional Information")
    print("-" * 40)
    
    x = pos_encoding(embeddings)
    
    # Process through transformer block
    print("\n" + "-" * 40)
    print("Step 3: Process Through Transformer Block")
    print("-" * 40)
    
    output = transformer_block(x)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"""
    Input: {token_ids.shape[0]} tokens (token IDs)
                    ↓
    Embedding: {embeddings.shape} (dense vectors)
                    ↓
    + Positional Encoding: {x.shape}
                    ↓
    Multi-Head Attention → Add & Norm
                    ↓
    Feed-Forward → Add & Norm
                    ↓
    Output: {output.shape} (contextualized representations)
    
    Each output vector now contains information about:
    - The token itself (from embedding)
    - Its position (from positional encoding)
    - Context from all other tokens (from attention)
    """)
    
    print("=" * 80)
    print("🎉 The Illustrated Transformer demo complete!")
    print("=" * 80)


def demo_single_attention():
    """Demo single attention computation for learning"""
    print("\n" + "=" * 80)
    print("🎓 ATTENTION MECHANISM - Educational Demo")
    print("=" * 80)
    
    # Small example for clarity
    print("\n📝 Let's trace attention with a small example:")
    print("   3 words, 4-dimensional embeddings")
    
    # Create simple embeddings (as if for words "I", "love", "transformers")
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],  # "I"
        [0.0, 1.0, 0.0, 0.0],  # "love"
        [0.0, 0.0, 1.0, 0.5],  # "transformers"
    ])
    
    print(f"\n📊 Input embeddings (Q = K = V in self-attention):")
    for i, word in enumerate(["I", "love", "transformers"]):
        print(f"   '{word}': {embeddings[i]}")
    
    output, weights = illustrated_attention_step_by_step(
        embeddings, embeddings, embeddings, "Self-Attention"
    )
    
    print("\n🎯 Interpretation:")
    print("   The attention weights show how much each word 'attends to' other words.")
    print("   High weight = strong relationship/importance")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run educational demos
    demo_single_attention()
    print("\n" * 2)
    illustrate_full_transformer_flow()
