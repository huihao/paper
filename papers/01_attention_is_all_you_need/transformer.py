#!/usr/bin/env python3
"""
Attention Is All You Need (Vaswani et al., 2017)
Implementation of the original Transformer architecture.

The Transformer model introduces:
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Encoder-decoder architecture

Paper: https://arxiv.org/abs/1706.03762
"""

import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding:
    """
    Sinusoidal positional encoding as described in the original paper.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._create_positional_encoding()
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create positional encoding matrix"""
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def __call__(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for a sequence length"""
        return self.pe[:seq_len]


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        query: Query matrix (batch, seq_len, d_k)
        key: Key matrix (batch, seq_len, d_k)
        value: Value matrix (batch, seq_len, d_v)
        mask: Optional mask for preventing attention to certain positions
        
    Returns:
        Output and attention weights
    """
    d_k = query.shape[-1]
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = np.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Compute output: attention_weights * V
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """
    
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_k)
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine the heads back into the last dimension"""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, d_k)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multi-head attention
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention weights
        """
        # Linear projections
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention to each head
        batch_size = query.shape[0]
        all_outputs = []
        all_weights = []
        
        for h in range(self.num_heads):
            output, weights = scaled_dot_product_attention(
                Q[:, h], K[:, h], V[:, h], mask
            )
            all_outputs.append(output)
            all_weights.append(weights)
        
        # Stack and combine heads
        output = np.stack(all_outputs, axis=1)  # (batch, heads, seq_len, d_k)
        output = self.combine_heads(output)
        
        # Final linear projection
        output = np.matmul(output, self.W_o)
        
        return output, np.stack(all_weights, axis=1)


class FeedForward:
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int):
        self.W_1 = np.random.randn(d_model, d_ff) * 0.02
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * 0.02
        self.b_2 = np.zeros(d_model)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply feed-forward network"""
        hidden = np.maximum(0, np.matmul(x, self.W_1) + self.b_1)  # ReLU
        return np.matmul(hidden, self.W_2) + self.b_2


class LayerNorm:
    """Layer Normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer:
    """
    Transformer Encoder Layer
    
    Each layer has:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply encoder layer"""
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class DecoderLayer:
    """
    Transformer Decoder Layer
    
    Each layer has:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention with encoder output
    3. Feed-forward network
    All with residual connections and layer normalization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def __call__(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        look_ahead_mask: Optional[np.ndarray] = None,
        padding_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply decoder layer"""
        # Masked self-attention
        self_attn_output, _ = self.self_attention(x, x, x, look_ahead_mask)
        x = self.norm1(x + self_attn_output)
        
        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, padding_mask)
        x = self.norm2(x + cross_attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x


class Transformer:
    """
    Full Transformer Model (Encoder-Decoder Architecture)
    
    The original Transformer as described in "Attention Is All You Need"
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.02
    
    def create_look_ahead_mask(self, size: int) -> np.ndarray:
        """Create causal mask for decoder self-attention"""
        mask = np.triu(np.ones((size, size)), k=1)
        return mask == 0  # Convert to attention mask (1 = attend, 0 = mask)
    
    def encode(self, src: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode the source sequence
        
        Args:
            src: Source token ids (batch, seq_len)
            mask: Optional padding mask
            
        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        seq_len = src.shape[1]
        
        # Token embedding + positional encoding
        x = self.embedding[src] * math.sqrt(self.d_model)
        x = x + self.pos_encoding(seq_len)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return x
    
    def decode(
        self,
        tgt: np.ndarray,
        encoder_output: np.ndarray,
        look_ahead_mask: Optional[np.ndarray] = None,
        padding_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Decode the target sequence
        
        Args:
            tgt: Target token ids (batch, seq_len)
            encoder_output: Output from encoder (batch, src_seq_len, d_model)
            look_ahead_mask: Causal mask for self-attention
            padding_mask: Padding mask for cross-attention
            
        Returns:
            Decoder output (batch, seq_len, d_model)
        """
        seq_len = tgt.shape[1]
        
        # Token embedding + positional encoding
        x = self.embedding[tgt] * math.sqrt(self.d_model)
        x = x + self.pos_encoding(seq_len)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask)
        
        return x
    
    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
        tgt_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through the transformer
        
        Args:
            src: Source token ids (batch, src_seq_len)
            tgt: Target token ids (batch, tgt_seq_len)
            src_mask: Source padding mask
            tgt_mask: Target padding mask
            
        Returns:
            Logits over vocabulary (batch, tgt_seq_len, vocab_size)
        """
        # Create look-ahead mask for decoder
        tgt_seq_len = tgt.shape[1]
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len)
        
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, look_ahead_mask, tgt_mask)
        
        # Project to vocabulary
        logits = np.matmul(decoder_output, self.output_projection)
        
        return logits


def demo():
    """Demonstrate the Transformer implementation"""
    print("=" * 80)
    print("Attention Is All You Need - Transformer Implementation Demo")
    print("=" * 80)
    
    # Create a small transformer
    vocab_size = 1000
    d_model = 64
    num_heads = 4
    
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256
    )
    
    # Create sample input
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = np.random.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = np.random.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Encoder layers: 2")
    print(f"  Decoder layers: 2")
    
    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    
    # Forward pass
    logits = transformer(src, tgt)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"  (batch_size, target_seq_len, vocab_size)")
    
    # Demonstrate attention
    print("\n" + "-" * 40)
    print("Demonstrating Scaled Dot-Product Attention:")
    print("-" * 40)
    
    Q = np.random.randn(1, 4, 64)
    K = np.random.randn(1, 4, 64)
    V = np.random.randn(1, 4, 64)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  Query/Key/Value shape: {Q.shape}")
    print(f"  Attention output shape: {output.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    print(f"  Attention weights (first batch):")
    print(f"  {weights[0].round(3)}")
    
    print("\n" + "=" * 80)
    print("Transformer implementation complete!")
    print("Key concepts: Self-attention, Multi-head attention, Encoder-Decoder")
    print("=" * 80)


if __name__ == "__main__":
    demo()
