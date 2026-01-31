#!/usr/bin/env python3
"""
BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
Implementation of BERT's key innovations: MLM and NSP pre-training objectives.

Paper: https://arxiv.org/abs/1810.04805
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import random


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class BertEmbedding:
    """
    BERT Embedding Layer
    
    Combines:
    1. Token embeddings
    2. Segment embeddings (for sentence A/B distinction)
    3. Position embeddings
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, max_position: int = 512):
        self.token_embedding = np.random.randn(vocab_size, hidden_size) * 0.02
        self.segment_embedding = np.random.randn(2, hidden_size) * 0.02  # 0 = Sentence A, 1 = Sentence B
        self.position_embedding = np.random.randn(max_position, hidden_size) * 0.02
    
    def __call__(
        self, 
        token_ids: np.ndarray, 
        segment_ids: np.ndarray
    ) -> np.ndarray:
        """
        Get BERT embeddings
        
        Args:
            token_ids: Token IDs (batch, seq_len)
            segment_ids: Segment IDs (0 or 1) (batch, seq_len)
            
        Returns:
            Combined embeddings (batch, seq_len, hidden_size)
        """
        seq_len = token_ids.shape[-1]
        
        # Get embeddings
        token_emb = self.token_embedding[token_ids]
        segment_emb = self.segment_embedding[segment_ids]
        position_emb = self.position_embedding[:seq_len]
        
        # Combine
        return token_emb + segment_emb + position_emb


class MaskedLanguageModel:
    """
    Masked Language Modeling (MLM) Head
    
    BERT's primary pre-training objective:
    - Randomly mask 15% of tokens
    - Predict the original tokens from context
    
    Of the 15% masked tokens:
    - 80% are replaced with [MASK]
    - 10% are replaced with random tokens
    - 10% are kept unchanged
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int,
        mask_token_id: int = 103,  # [MASK] token
        mask_probability: float = 0.15
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mask_token_id = mask_token_id
        self.mask_probability = mask_probability
        
        # MLM prediction head
        self.dense = np.random.randn(hidden_size, hidden_size) * 0.02
        self.layer_norm_gamma = np.ones(hidden_size)
        self.layer_norm_beta = np.zeros(hidden_size)
        self.output_weights = np.random.randn(hidden_size, vocab_size) * 0.02
        self.output_bias = np.zeros(vocab_size)
    
    def create_masked_input(
        self, 
        token_ids: np.ndarray,
        special_token_ids: List[int] = [101, 102, 0]  # [CLS], [SEP], [PAD]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create masked input for MLM training
        
        Args:
            token_ids: Original token IDs
            special_token_ids: Tokens that should not be masked
            
        Returns:
            Tuple of (masked_token_ids, mask_positions, original_tokens)
        """
        masked_ids = token_ids.copy()
        seq_len = token_ids.shape[0]
        
        mask_positions = []
        original_tokens = []
        
        for i in range(seq_len):
            if token_ids[i] in special_token_ids:
                continue
                
            if random.random() < self.mask_probability:
                mask_positions.append(i)
                original_tokens.append(token_ids[i])
                
                rand = random.random()
                if rand < 0.8:
                    # 80%: Replace with [MASK]
                    masked_ids[i] = self.mask_token_id
                elif rand < 0.9:
                    # 10%: Replace with random token
                    masked_ids[i] = random.randint(0, self.vocab_size - 1)
                # 10%: Keep original (do nothing)
        
        return masked_ids, np.array(mask_positions), np.array(original_tokens)
    
    def predict(self, hidden_states: np.ndarray, mask_positions: np.ndarray) -> np.ndarray:
        """
        Predict masked tokens
        
        Args:
            hidden_states: BERT output (seq_len, hidden_size)
            mask_positions: Positions of masked tokens
            
        Returns:
            Logits for masked positions (num_masked, vocab_size)
        """
        # Get hidden states at masked positions
        masked_hidden = hidden_states[mask_positions]
        
        # Dense + activation
        x = np.matmul(masked_hidden, self.dense)
        x = np.maximum(0, x)  # GELU approximation (using ReLU for simplicity)
        
        # Layer norm
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        x = self.layer_norm_gamma * (x - mean) / (std + 1e-6) + self.layer_norm_beta
        
        # Project to vocabulary
        logits = np.matmul(x, self.output_weights) + self.output_bias
        
        return logits
    
    def compute_loss(
        self, 
        logits: np.ndarray, 
        original_tokens: np.ndarray
    ) -> float:
        """
        Compute MLM loss (cross-entropy)
        
        Args:
            logits: Predicted logits (num_masked, vocab_size)
            original_tokens: Ground truth token IDs (num_masked,)
            
        Returns:
            Average cross-entropy loss
        """
        probs = softmax(logits, axis=-1)
        
        # Get probability of correct tokens
        correct_probs = probs[np.arange(len(original_tokens)), original_tokens]
        
        # Cross-entropy loss
        loss = -np.mean(np.log(correct_probs + 1e-10))
        
        return loss


class NextSentencePrediction:
    """
    Next Sentence Prediction (NSP) Head
    
    BERT's secondary pre-training objective:
    - Given sentence A and sentence B
    - Predict if B is the actual next sentence after A
    
    Training data:
    - 50% positive: B actually follows A
    - 50% negative: B is a random sentence
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        
        # NSP classifier (binary)
        self.classifier = np.random.randn(hidden_size, 2) * 0.02
        self.bias = np.zeros(2)
    
    def predict(self, pooled_output: np.ndarray) -> np.ndarray:
        """
        Predict if sentence B follows sentence A
        
        Args:
            pooled_output: [CLS] token representation (hidden_size,)
            
        Returns:
            Logits for [IsNext, NotNext] (2,)
        """
        logits = np.matmul(pooled_output, self.classifier) + self.bias
        return logits
    
    def compute_loss(self, logits: np.ndarray, is_next: bool) -> float:
        """
        Compute NSP loss (binary cross-entropy)
        
        Args:
            logits: Predicted logits (2,)
            is_next: True if B follows A, False otherwise
            
        Returns:
            Cross-entropy loss
        """
        probs = softmax(logits)
        label = 0 if is_next else 1
        loss = -np.log(probs[label] + 1e-10)
        return loss


class BertEncoder:
    """
    Simplified BERT Encoder (single layer for demonstration)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-attention weights
        self.W_q = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_k = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_v = np.random.randn(hidden_size, hidden_size) * 0.02
        self.W_o = np.random.randn(hidden_size, hidden_size) * 0.02
        
        # Feed-forward
        self.ff_1 = np.random.randn(hidden_size, intermediate_size) * 0.02
        self.ff_2 = np.random.randn(intermediate_size, hidden_size) * 0.02
    
    def attention(self, x: np.ndarray) -> np.ndarray:
        """Apply self-attention"""
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        scores = np.matmul(Q, K.T) / np.sqrt(self.head_dim)
        weights = softmax(scores, axis=-1)
        attn_out = np.matmul(weights, V)
        
        return np.matmul(attn_out, self.W_o)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through encoder layer"""
        # Self-attention + residual
        attn_out = self.attention(x)
        x = x + attn_out
        
        # Layer norm (simplified)
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
        
        # Feed-forward + residual
        ff_hidden = np.maximum(0, np.matmul(x, self.ff_1))  # GELU approximation
        ff_out = np.matmul(ff_hidden, self.ff_2)
        x = x + ff_out
        
        # Layer norm
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
        
        return x


class BertModel:
    """
    Complete BERT Model for Pre-training
    
    Architecture:
    - Embedding layer (token + segment + position)
    - N Transformer encoder layers
    - MLM head for masked token prediction
    - NSP head for next sentence prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position: int = 512
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embeddings
        self.embeddings = BertEmbedding(vocab_size, hidden_size, max_position)
        
        # Encoder layers (simplified to 1 layer for demo)
        self.encoder = BertEncoder(hidden_size, num_heads, intermediate_size)
        
        # Pooler for [CLS] token
        self.pooler = np.random.randn(hidden_size, hidden_size) * 0.02
        
        # Pre-training heads
        self.mlm_head = MaskedLanguageModel(vocab_size, hidden_size)
        self.nsp_head = NextSentencePrediction(hidden_size)
    
    def forward(
        self,
        token_ids: np.ndarray,
        segment_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through BERT
        
        Args:
            token_ids: Token IDs (seq_len,)
            segment_ids: Segment IDs (seq_len,)
            
        Returns:
            Tuple of (sequence_output, pooled_output)
        """
        # Get embeddings
        x = self.embeddings(token_ids, segment_ids)
        
        # Encode
        sequence_output = self.encoder(x)
        
        # Pool [CLS] token
        cls_hidden = sequence_output[0]
        pooled_output = np.tanh(np.matmul(cls_hidden, self.pooler))
        
        return sequence_output, pooled_output
    
    def pretrain_step(
        self,
        token_ids: np.ndarray,
        segment_ids: np.ndarray,
        is_next: bool
    ) -> Dict[str, float]:
        """
        Single pre-training step
        
        Args:
            token_ids: Original token IDs
            segment_ids: Segment IDs
            is_next: Whether sentence B follows sentence A
            
        Returns:
            Dictionary of losses
        """
        # Create masked input
        masked_ids, mask_positions, original_tokens = self.mlm_head.create_masked_input(token_ids)
        
        # Forward pass with masked input
        sequence_output, pooled_output = self.forward(masked_ids, segment_ids)
        
        # Compute MLM loss
        if len(mask_positions) > 0:
            mlm_logits = self.mlm_head.predict(sequence_output, mask_positions)
            mlm_loss = self.mlm_head.compute_loss(mlm_logits, original_tokens)
        else:
            mlm_loss = 0.0
        
        # Compute NSP loss
        nsp_logits = self.nsp_head.predict(pooled_output)
        nsp_loss = self.nsp_head.compute_loss(nsp_logits, is_next)
        
        # Total loss
        total_loss = mlm_loss + nsp_loss
        
        return {
            'mlm_loss': mlm_loss,
            'nsp_loss': nsp_loss,
            'total_loss': total_loss
        }


def demo():
    """Demonstrate BERT pre-training"""
    print("=" * 80)
    print("BERT: Pre-training of Deep Bidirectional Transformers")
    print("=" * 80)
    
    # Create small BERT model
    print("\n📦 Creating BERT model...")
    model = BertModel(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=256,
        max_position=128
    )
    
    print(f"   Vocabulary size: 1000")
    print(f"   Hidden size: 64")
    print(f"   Number of heads: 4")
    
    # Create sample input
    # [CLS] sentence A [SEP] sentence B [SEP]
    print("\n📝 Creating sample input...")
    
    # Simulated tokenized input
    # [CLS]=101, [SEP]=102, [MASK]=103, [PAD]=0
    sentence_a_tokens = [101, 50, 23, 67, 89, 12, 102]  # [CLS] + 5 words + [SEP]
    sentence_b_tokens = [45, 78, 34, 56, 102]  # 4 words + [SEP]
    
    token_ids = np.array(sentence_a_tokens + sentence_b_tokens)
    segment_ids = np.array([0] * len(sentence_a_tokens) + [1] * len(sentence_b_tokens))
    
    print(f"   Token IDs: {token_ids}")
    print(f"   Segment IDs: {segment_ids}")
    print(f"   Sequence length: {len(token_ids)}")
    
    # Demonstrate MLM
    print("\n🎭 Masked Language Modeling (MLM) Demo:")
    print("-" * 40)
    
    masked_ids, mask_pos, orig_tokens = model.mlm_head.create_masked_input(token_ids)
    print(f"   Original tokens: {token_ids}")
    print(f"   Masked tokens:   {masked_ids}")
    print(f"   Masked positions: {mask_pos}")
    print(f"   Original values at masked positions: {orig_tokens}")
    
    # Forward pass
    print("\n🔄 Forward pass through BERT...")
    sequence_output, pooled_output = model.forward(token_ids, segment_ids)
    print(f"   Sequence output shape: {sequence_output.shape}")
    print(f"   Pooled output shape: {pooled_output.shape}")
    
    # Pre-training step
    print("\n📊 Pre-training step (MLM + NSP):")
    print("-" * 40)
    
    # Positive example (is_next=True)
    losses_pos = model.pretrain_step(token_ids, segment_ids, is_next=True)
    print(f"   Positive example (B follows A):")
    print(f"      MLM Loss: {losses_pos['mlm_loss']:.4f}")
    print(f"      NSP Loss: {losses_pos['nsp_loss']:.4f}")
    print(f"      Total Loss: {losses_pos['total_loss']:.4f}")
    
    # Negative example (is_next=False)
    losses_neg = model.pretrain_step(token_ids, segment_ids, is_next=False)
    print(f"\n   Negative example (B is random):")
    print(f"      MLM Loss: {losses_neg['mlm_loss']:.4f}")
    print(f"      NSP Loss: {losses_neg['nsp_loss']:.4f}")
    print(f"      Total Loss: {losses_neg['total_loss']:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📚 BERT Key Innovations:")
    print("-" * 40)
    print("""
    1. Bidirectional Context: Unlike GPT (left-to-right), BERT sees
       context from both directions using the [MASK] token.
    
    2. Pre-training Objectives:
       - MLM (Masked Language Modeling): Predict masked tokens
       - NSP (Next Sentence Prediction): Understand sentence relationships
    
    3. Segment Embeddings: Distinguish between sentence A and B
    
    4. [CLS] Token: Used for classification tasks, represents
       the entire input sequence.
    
    5. Fine-tuning: Pre-trained BERT can be fine-tuned for
       various downstream tasks (QA, NER, classification, etc.)
    """)
    print("=" * 80)


if __name__ == "__main__":
    demo()
