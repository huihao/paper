#!/usr/bin/env python3
"""
Language Models are Few-Shot Learners (GPT-3) (Brown et al., 2020)
Implementation demonstrating in-context learning capabilities.

Paper: https://arxiv.org/abs/2005.14165
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@dataclass
class GPT3Config:
    """GPT-3 model configuration"""
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 12288  # GPT-3 175B
    n_head: int = 96
    n_layer: int = 96
    
    @classmethod
    def gpt3_small(cls):
        """GPT-3 Small (125M)"""
        return cls(n_embd=768, n_head=12, n_layer=12)
    
    @classmethod
    def gpt3_medium(cls):
        """GPT-3 Medium (350M)"""
        return cls(n_embd=1024, n_head=16, n_layer=24)
    
    @classmethod
    def gpt3_large(cls):
        """GPT-3 Large (760M)"""
        return cls(n_embd=1536, n_head=16, n_layer=24)
    
    @classmethod
    def gpt3_xl(cls):
        """GPT-3 XL (1.3B)"""
        return cls(n_embd=2048, n_head=24, n_layer=24)
    
    @classmethod
    def gpt3_175b(cls):
        """GPT-3 175B (the full model)"""
        return cls(n_embd=12288, n_head=96, n_layer=96)


class InContextLearning:
    """
    Demonstrates In-Context Learning (ICL) - GPT-3's key contribution.
    
    In-context learning allows the model to learn from examples
    provided in the prompt, without any gradient updates.
    
    Three paradigms:
    1. Zero-shot: Task description only
    2. One-shot: Task description + one example
    3. Few-shot: Task description + multiple examples
    """
    
    def __init__(self, model_config: Optional[GPT3Config] = None):
        self.config = model_config or GPT3Config.gpt3_small()
    
    def create_zero_shot_prompt(
        self,
        task_description: str,
        input_text: str
    ) -> str:
        """
        Create a zero-shot prompt (no examples)
        
        The model must understand the task from description alone.
        """
        prompt = f"""{task_description}

Input: {input_text}
Output:"""
        return prompt
    
    def create_one_shot_prompt(
        self,
        task_description: str,
        example: Tuple[str, str],
        input_text: str
    ) -> str:
        """
        Create a one-shot prompt (single example)
        
        One example helps the model understand the expected format.
        """
        example_input, example_output = example
        prompt = f"""{task_description}

Input: {example_input}
Output: {example_output}

Input: {input_text}
Output:"""
        return prompt
    
    def create_few_shot_prompt(
        self,
        task_description: str,
        examples: List[Tuple[str, str]],
        input_text: str
    ) -> str:
        """
        Create a few-shot prompt (multiple examples)
        
        Multiple examples help the model learn the pattern more reliably.
        GPT-3 showed this works remarkably well for many tasks.
        """
        examples_text = "\n\n".join([
            f"Input: {inp}\nOutput: {out}"
            for inp, out in examples
        ])
        
        prompt = f"""{task_description}

{examples_text}

Input: {input_text}
Output:"""
        return prompt


class GPT3Decoder:
    """
    Simplified GPT-3 Decoder for demonstration.
    
    Key architectural features:
    - Decoder-only transformer
    - Autoregressive generation
    - Learned position embeddings
    - Pre-norm architecture (layer norm before attention/FFN)
    """
    
    def __init__(self, config: GPT3Config):
        self.config = config
        
        # Embeddings
        self.token_embedding = np.random.randn(config.vocab_size, config.n_embd) * 0.02
        self.position_embedding = np.random.randn(config.n_positions, config.n_embd) * 0.01
        
        # Single layer for demonstration
        self.W_q = np.random.randn(config.n_embd, config.n_embd) * 0.02
        self.W_k = np.random.randn(config.n_embd, config.n_embd) * 0.02
        self.W_v = np.random.randn(config.n_embd, config.n_embd) * 0.02
        self.W_o = np.random.randn(config.n_embd, config.n_embd) * 0.02
        
        # Feed-forward
        self.ff_1 = np.random.randn(config.n_embd, 4 * config.n_embd) * 0.02
        self.ff_2 = np.random.randn(4 * config.n_embd, config.n_embd) * 0.02
        
        # Output head
        self.lm_head = self.token_embedding.T  # Weight tying
    
    def causal_attention(self, x: np.ndarray) -> np.ndarray:
        """Causal (masked) self-attention"""
        seq_len = x.shape[0]
        head_dim = self.config.n_embd // self.config.n_head
        
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Attention scores
        scores = np.matmul(Q, K.T) / np.sqrt(head_dim)
        
        # Causal mask (prevent attending to future tokens)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = np.where(mask == 1, -1e9, scores)
        
        weights = softmax(scores, axis=-1)
        attn_out = np.matmul(weights, V)
        
        return np.matmul(attn_out, self.W_o)
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass through GPT-3
        
        Args:
            token_ids: Input token IDs (seq_len,)
            
        Returns:
            Logits over vocabulary (seq_len, vocab_size)
        """
        seq_len = len(token_ids)
        
        # Embeddings
        x = self.token_embedding[token_ids] + self.position_embedding[:seq_len]
        
        # Pre-norm + attention + residual
        norm_x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
        x = x + self.causal_attention(norm_x)
        
        # Pre-norm + FFN + residual
        norm_x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
        hidden = np.maximum(0, np.matmul(norm_x, self.ff_1))  # GELU approximation
        x = x + np.matmul(hidden, self.ff_2)
        
        # Final layer norm
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
        
        # Project to vocabulary
        logits = np.matmul(x, self.lm_head)
        
        return logits
    
    def generate(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 40
    ) -> np.ndarray:
        """
        Autoregressive text generation
        
        Args:
            token_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top-k tokens
            
        Returns:
            Generated token IDs
        """
        generated = list(token_ids)
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            input_ids = np.array(generated[-self.config.n_positions:])
            logits = self.forward(input_ids)[-1]  # Last token
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_indices = np.argsort(logits)[-top_k:]
                mask = np.ones_like(logits) * (-1e9)
                mask[top_k_indices] = 0
                logits = logits + mask
            
            # Sample
            probs = softmax(logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            generated.append(next_token)
        
        return np.array(generated)


class GPT3ScalingAnalysis:
    """
    Analysis of GPT-3's scaling behavior.
    
    The paper showed that performance scales smoothly with:
    - Model size (parameters)
    - Dataset size
    - Compute
    """
    
    # Model sizes from the paper
    MODEL_SIZES = {
        'GPT-3 Small': 125e6,
        'GPT-3 Medium': 350e6,
        'GPT-3 Large': 760e6,
        'GPT-3 XL': 1.3e9,
        'GPT-3 2.7B': 2.7e9,
        'GPT-3 6.7B': 6.7e9,
        'GPT-3 13B': 13e9,
        'GPT-3 175B': 175e9
    }
    
    @staticmethod
    def estimate_compute_petaflop_days(
        parameters: float,
        tokens: float,
        flops_per_param_per_token: float = 6
    ) -> float:
        """
        Estimate training compute in PetaFLOP-days
        
        Approximate formula: 6 * N * D
        where N = parameters, D = tokens
        """
        flops = flops_per_param_per_token * parameters * tokens
        petaflop_days = flops / (1e15 * 24 * 3600)
        return petaflop_days
    
    @staticmethod
    def print_scaling_table():
        """Print GPT-3 model scaling table"""
        print("\n📊 GPT-3 Model Sizes:")
        print("-" * 50)
        print(f"{'Model':<20} {'Parameters':>15} {'Layers':>8}")
        print("-" * 50)
        
        layer_counts = {
            'GPT-3 Small': 12,
            'GPT-3 Medium': 24,
            'GPT-3 Large': 24,
            'GPT-3 XL': 24,
            'GPT-3 2.7B': 32,
            'GPT-3 6.7B': 32,
            'GPT-3 13B': 40,
            'GPT-3 175B': 96
        }
        
        for name, params in GPT3ScalingAnalysis.MODEL_SIZES.items():
            params_str = f"{params/1e9:.1f}B" if params >= 1e9 else f"{params/1e6:.0f}M"
            layers = layer_counts.get(name, "?")
            print(f"{name:<20} {params_str:>15} {layers:>8}")


def demo_in_context_learning():
    """Demonstrate in-context learning paradigms"""
    print("=" * 80)
    print("GPT-3: In-Context Learning Demo")
    print("=" * 80)
    
    icl = InContextLearning()
    
    # Task: Sentiment Analysis
    task = "Classify the sentiment of the following text as positive or negative."
    
    print("\n📝 Task: Sentiment Analysis")
    print("-" * 40)
    
    # Zero-shot
    print("\n1️⃣ Zero-Shot Learning:")
    prompt = icl.create_zero_shot_prompt(task, "I love this product!")
    print(f"Prompt:\n{prompt}")
    print("(Model must understand task from description alone)")
    
    # One-shot
    print("\n2️⃣ One-Shot Learning:")
    example = ("This movie was terrible.", "negative")
    prompt = icl.create_one_shot_prompt(task, example, "I love this product!")
    print(f"Prompt:\n{prompt}")
    print("(Model learns format from one example)")
    
    # Few-shot
    print("\n3️⃣ Few-Shot Learning:")
    examples = [
        ("This movie was terrible.", "negative"),
        ("Best restaurant I've ever been to!", "positive"),
        ("Waste of money, don't buy.", "negative"),
        ("Absolutely amazing experience!", "positive"),
    ]
    prompt = icl.create_few_shot_prompt(task, examples, "I love this product!")
    print(f"Prompt:\n{prompt}")
    print("(Model learns pattern from multiple examples)")
    
    # Translation example
    print("\n" + "-" * 40)
    print("\n📝 Task: Translation (English → French)")
    
    translation_examples = [
        ("Hello, how are you?", "Bonjour, comment allez-vous?"),
        ("Thank you very much.", "Merci beaucoup."),
        ("I love programming.", "J'adore la programmation."),
    ]
    
    prompt = icl.create_few_shot_prompt(
        "Translate English to French.",
        translation_examples,
        "The weather is beautiful today."
    )
    print(f"Few-shot prompt:\n{prompt}")


def demo_model_architecture():
    """Demonstrate GPT-3 architecture"""
    print("\n" + "=" * 80)
    print("GPT-3: Architecture Demo")
    print("=" * 80)
    
    # Create small model
    config = GPT3Config.gpt3_small()
    model = GPT3Decoder(config)
    
    print("\n📦 Model Configuration:")
    print(f"   Vocabulary size: {config.vocab_size}")
    print(f"   Embedding dimension: {config.n_embd}")
    print(f"   Number of heads: {config.n_head}")
    print(f"   Number of layers: {config.n_layer}")
    print(f"   Max sequence length: {config.n_positions}")
    
    # Sample input
    token_ids = np.random.randint(0, 1000, size=20)
    print(f"\n📥 Input shape: {token_ids.shape}")
    
    # Forward pass
    logits = model.forward(token_ids)
    print(f"📤 Output logits shape: {logits.shape}")
    
    # Generation
    print("\n🔄 Autoregressive Generation:")
    generated = model.generate(token_ids[:5], max_new_tokens=10)
    print(f"   Input tokens: {token_ids[:5]}")
    print(f"   Generated: {generated}")


def demo_scaling():
    """Demonstrate scaling analysis"""
    print("\n" + "=" * 80)
    print("GPT-3: Scaling Analysis")
    print("=" * 80)
    
    GPT3ScalingAnalysis.print_scaling_table()
    
    print("\n📈 Key Findings:")
    print("-" * 40)
    print("""
    1. Smooth Scaling: Performance improves predictably with model size
    
    2. Emergent Abilities: Larger models can perform tasks that
       smaller models cannot (e.g., 3-digit arithmetic)
    
    3. Few-Shot >> Zero-Shot: More examples in prompt helps a lot
    
    4. Sample Efficiency: Larger models need fewer examples to
       achieve the same performance
    
    5. Training Data: 300B tokens from filtered Common Crawl,
       books, Wikipedia, and web text
    """)
    
    # Compute estimate
    tokens_trained = 300e9
    params_175b = 175e9
    compute = GPT3ScalingAnalysis.estimate_compute_petaflop_days(params_175b, tokens_trained)
    print(f"\n💻 Estimated GPT-3 175B Training Compute:")
    print(f"   ~{compute:.0f} PetaFLOP-days")


def main():
    """Main demonstration"""
    demo_in_context_learning()
    demo_model_architecture()
    demo_scaling()
    
    print("\n" + "=" * 80)
    print("📚 GPT-3 Key Contributions:")
    print("=" * 80)
    print("""
    1. In-Context Learning: Models can learn from examples in the prompt
       without any weight updates
    
    2. Scale Matters: 175B parameters showed emergent capabilities
       not present in smaller models
    
    3. Few-Shot Prompting: A new paradigm for using language models
       that doesn't require fine-tuning
    
    4. General-Purpose: One model for many tasks, selected by prompt
    
    5. Opened the Door: Led to ChatGPT, instruction tuning, and
       the modern era of AI assistants
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
