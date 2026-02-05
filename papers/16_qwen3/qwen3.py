#!/usr/bin/env python3
"""
Qwen3 Technical Report (Yang et al., 2025)
Implementation demonstrating unified MoE with thinking and non-thinking modes.

Paper: https://arxiv.org/abs/2505.09388
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ThinkingMode(Enum):
    """Qwen3's thinking modes"""
    THINKING = "thinking"      # Extended reasoning with <think> tags
    NON_THINKING = "non_thinking"  # Direct, fast responses


@dataclass
class Qwen3Config:
    """Qwen3 model configuration"""
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    num_experts: int = 64  # For MoE variants
    num_active_experts: int = 8  # Active per token
    rope_theta: float = 1000000.0  # Extended context


class GroupedQueryAttention:
    """
    Grouped Query Attention (GQA)
    
    Used in Qwen3 for efficiency.
    Multiple query heads share the same key-value heads.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        
        # Projections
        self.q_proj = np.random.randn(hidden_size, num_attention_heads * self.head_dim) * 0.02
        self.k_proj = np.random.randn(hidden_size, num_key_value_heads * self.head_dim) * 0.02
        self.v_proj = np.random.randn(hidden_size, num_key_value_heads * self.head_dim) * 0.02
        self.o_proj = np.random.randn(num_attention_heads * self.head_dim, hidden_size) * 0.02
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply GQA"""
        seq_len = x.shape[0]
        
        # Project
        q = np.matmul(x, self.q_proj)
        k = np.matmul(x, self.k_proj)
        v = np.matmul(x, self.v_proj)
        
        # Reshape
        q = q.reshape(seq_len, self.num_attention_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_key_value_heads, self.head_dim)
        
        # Repeat KV for each group
        k = np.repeat(k, self.num_key_value_groups, axis=1)
        v = np.repeat(v, self.num_key_value_groups, axis=1)
        
        # Compute attention (simplified)
        # In practice, use flash attention
        outputs = []
        for h in range(self.num_attention_heads):
            scores = np.matmul(q[:, h], k[:, h].T) / np.sqrt(self.head_dim)
            # Causal mask
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + mask
            weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            output = np.matmul(weights, v[:, h])
            outputs.append(output)
        
        concat = np.concatenate(outputs, axis=-1)
        return np.matmul(concat, self.o_proj)


class Qwen3MoELayer:
    """
    Qwen3 Mixture of Experts Layer
    
    Key features:
    - Fine-grained experts (many small experts)
    - Top-k routing with auxiliary loss
    - Shared expert for base knowledge
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_active: int
    ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_active = num_active
        
        # Router
        self.router = np.random.randn(hidden_size, num_experts) * 0.02
        
        # Expert FFNs (simplified)
        self.expert_up = [
            np.random.randn(hidden_size, intermediate_size // num_experts) * 0.02
            for _ in range(num_experts)
        ]
        self.expert_down = [
            np.random.randn(intermediate_size // num_experts, hidden_size) * 0.02
            for _ in range(num_experts)
        ]
    
    def route(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route tokens to experts.
        Returns top-k expert indices and weights.
        """
        # Compute router logits
        logits = np.matmul(x, self.router)  # (seq_len, num_experts)
        
        # Top-k selection
        top_k_idx = np.argsort(logits, axis=-1)[:, -self.num_active:]
        
        # Softmax over selected experts
        top_k_logits = np.take_along_axis(logits, top_k_idx, axis=-1)
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits, axis=-1, keepdims=True))
        weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return top_k_idx, weights
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply MoE layer"""
        seq_len = x.shape[0]
        
        # Route
        expert_idx, expert_weights = self.route(x)
        
        # Compute expert outputs
        output = np.zeros_like(x)
        
        for i in range(seq_len):
            for j, (exp_id, weight) in enumerate(zip(expert_idx[i], expert_weights[i])):
                # Expert forward pass
                hidden = np.maximum(0, np.matmul(x[i], self.expert_up[exp_id]))
                exp_out = np.matmul(hidden, self.expert_down[exp_id])
                output[i] += weight * exp_out
        
        return output


class UnifiedThinkingController:
    """
    Unified Thinking Mode Controller
    
    Qwen3's key innovation: Same model can operate in
    thinking mode (extended CoT) or non-thinking mode (fast).
    """
    
    def __init__(self):
        self.mode = ThinkingMode.NON_THINKING
    
    def set_mode(self, mode: ThinkingMode):
        """Set the thinking mode"""
        self.mode = mode
    
    def should_think(self, query: str) -> bool:
        """
        Decide whether to use thinking mode based on query.
        
        Thinking mode is better for:
        - Complex reasoning
        - Math problems
        - Multi-step tasks
        
        Non-thinking is better for:
        - Simple questions
        - Chitchat
        - Speed-critical applications
        """
        thinking_indicators = [
            "solve", "calculate", "prove", "reason",
            "step by step", "explain why", "analyze"
        ]
        
        query_lower = query.lower()
        for indicator in thinking_indicators:
            if indicator in query_lower:
                return True
        return False
    
    def format_response(
        self,
        thinking: Optional[str],
        answer: str
    ) -> str:
        """Format response based on mode"""
        if self.mode == ThinkingMode.THINKING and thinking:
            return f"<think>\n{thinking}\n</think>\n\n{answer}"
        return answer
    
    def estimate_cost(self, thinking: bool) -> Dict:
        """Estimate cost difference between modes"""
        base_tokens = 100
        
        if thinking:
            return {
                'thinking_tokens': 500,  # Extended reasoning
                'answer_tokens': base_tokens,
                'total': 600,
                'latency': 'higher'
            }
        else:
            return {
                'thinking_tokens': 0,
                'answer_tokens': base_tokens,
                'total': base_tokens,
                'latency': 'lower'
            }


def demonstrate_thinking_modes():
    """Demonstrate thinking vs non-thinking modes"""
    print("=" * 70)
    print("Qwen3: Unified MoE with Thinking Modes")
    print("Yang et al., 2025")
    print("=" * 70)
    
    controller = UnifiedThinkingController()
    
    # Example queries
    queries = [
        ("What is 2 + 2?", False),
        ("Solve the equation: 3x + 5 = 20", True),
        ("Hello, how are you?", False),
        ("Prove that the sum of angles in a triangle is 180°", True),
    ]
    
    print("\n📊 Automatic Mode Selection:")
    print("-" * 50)
    
    for query, expected_thinking in queries:
        should_think = controller.should_think(query)
        mode = "Thinking" if should_think else "Non-Thinking"
        icon = "🧠" if should_think else "⚡"
        
        cost = controller.estimate_cost(should_think)
        
        print(f"\n{icon} Query: \"{query}\"")
        print(f"   Mode: {mode}")
        print(f"   Est. tokens: {cost['total']}")
        print(f"   Latency: {cost['latency']}")


def demonstrate_moe():
    """Demonstrate MoE routing"""
    print("\n" + "=" * 70)
    print("📊 MoE Routing Demonstration")
    print("=" * 70)
    
    config = Qwen3Config()
    
    print(f"\nQwen3 MoE Configuration:")
    print(f"  Total experts: {config.num_experts}")
    print(f"  Active experts per token: {config.num_active_experts}")
    print(f"  Activation ratio: {config.num_active_experts/config.num_experts:.1%}")
    
    # Create MoE layer
    moe = Qwen3MoELayer(
        hidden_size=64,  # Small for demo
        intermediate_size=256,
        num_experts=8,
        num_active=2
    )
    
    # Sample input
    x = np.random.randn(4, 64)  # 4 tokens
    
    # Route
    expert_idx, weights = moe.route(x)
    
    print(f"\n  Token routing (4 tokens → top-2 experts):")
    for i in range(4):
        experts = expert_idx[i]
        ws = weights[i]
        print(f"    Token {i}: Expert {experts[0]} ({ws[0]:.2f}), Expert {experts[1]} ({ws[1]:.2f})")


def explain_qwen3_innovations():
    """Explain Qwen3's key innovations"""
    print("\n" + "=" * 70)
    print("📚 Qwen3 Key Innovations")
    print("=" * 70)
    
    print("""
1. Unified MoE Architecture:
   - Dense and MoE variants from same family
   - Shared vocabulary and training approach
   - Easy to choose based on compute budget

2. Thinking Mode Toggle:
   - Same model, two operating modes
   - Thinking: Extended reasoning, higher quality
   - Non-thinking: Fast, direct answers
   - User/system can control mode

3. Fine-Grained Experts:
   - Many small experts (64-128)
   - Better specialization
   - Smoother routing

4. Grouped Query Attention (GQA):
   - Reduced KV cache memory
   - Faster inference
   - Especially for long contexts

5. Extended Context:
   - Base: 32k tokens
   - Extended: 128k with YaRN
   - RoPE theta = 1,000,000

6. Training Approach:
   - Multi-stage post-training
   - Mix of thinking and non-thinking data
   - Quality over quantity
""")


def demo():
    """Main demonstration"""
    demonstrate_thinking_modes()
    demonstrate_moe()
    explain_qwen3_innovations()
    
    print("\n" + "=" * 70)
    print("📚 Why Qwen3 Matters")
    print("=" * 70)
    print("""
    1. Unified Architecture: One family for all use cases
    
    2. Dynamic Reasoning: Trade off cost vs quality at runtime
    
    3. State-of-the-art MoE: Efficient with fine-grained routing
    
    4. Practical Design: Built for real-world deployment
    
    5. Open Weights: Community can use and learn from it
    
    Key Insight:
        The same model can be efficient (non-thinking) when you
        need speed, and thorough (thinking) when you need accuracy.
        No need for separate models!
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
