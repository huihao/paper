#!/usr/bin/env python3
"""
DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
(Guo et al., 2025)

Implementation demonstrating the R1 approach to reasoning through RL.

Paper: https://arxiv.org/abs/2501.12948
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ReasoningPattern(Enum):
    """Patterns that emerge during R1 training"""
    SELF_VERIFICATION = "self_verification"
    REFLECTION = "reflection"
    BACKTRACKING = "backtracking"
    STEP_BY_STEP = "step_by_step"
    EXPLORATION = "exploration"


@dataclass
class ReasoningTrace:
    """A reasoning trace with self-verification"""
    question: str
    thinking: str  # The <think> block
    answer: str
    verified: bool
    patterns_used: List[ReasoningPattern]


class DeepSeekR1Concepts:
    """
    DeepSeek-R1: Key Concepts
    
    The R1 paper proved that large-scale reinforcement learning without
    supervised data can induce self-verification and structured reasoning.
    
    Key innovations:
    1. Pure RL training on reasoning tasks
    2. Emergence of self-verification
    3. Long chain-of-thought without examples
    4. Aha moments during training
    """
    
    @staticmethod
    def describe_training_process():
        return """
DeepSeek-R1 Training Process:

Phase 1: Cold Start with SFT
    - Small amount of long CoT data
    - Teaches format but not reasoning

Phase 2: Reasoning RL (GRPO)
    - Pure RL on math/reasoning tasks
    - Reward: correctness only
    - No intermediate supervision
    
Phase 3: Rejection Sampling + SFT
    - Generate many solutions
    - Keep correct ones
    - Distill back into model

Phase 4: Final RL
    - Align for helpfulness
    - Add safety constraints

The Magic:
    Without being taught HOW to reason, the model discovers:
    - Self-verification ("Let me check...")
    - Reflection ("Wait, that doesn't work...")
    - Multi-step decomposition
"""
    
    @staticmethod
    def describe_emergent_behaviors():
        return """
Emergent Behaviors in R1:

1. Self-Verification:
   Model: "Let me verify this step..."
   "Checking: 5 × 7 = 35. Yes, correct."
   
   Emerges naturally from reward signal!

2. Reflection:
   Model: "Hmm, this approach isn't working..."
   "Let me try a different method."
   
   Backtracking to find better solutions.

3. Aha Moments:
   During training, loss suddenly drops when model
   "discovers" new reasoning strategies.
   
4. Extended Thinking:
   Can generate very long thinking traces
   (thousands of tokens) when needed.

5. Structured Problem Decomposition:
   Breaks complex problems into steps
   without being explicitly taught.
"""


class GRPOSimulator:
    """
    Group Relative Policy Optimization (GRPO)
    
    A simplified RL algorithm used in DeepSeek-R1.
    Key idea: Compare multiple samples from the same prompt
    and use relative rankings for optimization.
    """
    
    def __init__(self, num_samples: int = 8):
        self.num_samples = num_samples
        self.reward_history = []
    
    def sample_responses(self, prompt: str) -> List[str]:
        """
        Sample multiple responses for a prompt.
        In practice, this is done with temperature > 0.
        """
        # Simulate different quality responses
        responses = [
            f"Response {i}: [Simulated reasoning chain]"
            for i in range(self.num_samples)
        ]
        return responses
    
    def compute_rewards(self, responses: List[str], correct_answer: str) -> np.ndarray:
        """
        Compute rewards for each response.
        In R1, reward is primarily based on correctness.
        """
        # Simulate reward computation
        # Higher reward for responses that would be correct
        rewards = np.random.rand(len(responses))
        
        # Normalize to have zero mean within group
        rewards = rewards - np.mean(rewards)
        
        return rewards
    
    def grpo_objective(
        self,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        old_log_probs: np.ndarray,
        beta: float = 0.1
    ) -> float:
        """
        GRPO objective function.
        
        Similar to PPO but uses group-relative advantages.
        """
        # Ratio of new to old policy
        ratio = np.exp(log_probs - old_log_probs)
        
        # Clipped objective
        epsilon = 0.2
        clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
        
        # Use rewards as advantages (already normalized)
        obj1 = ratio * rewards
        obj2 = clipped_ratio * rewards
        
        # KL penalty
        kl = np.mean(old_log_probs - log_probs)
        
        objective = np.mean(np.minimum(obj1, obj2)) - beta * kl
        
        return objective
    
    def train_step(self, prompt: str, correct_answer: str) -> Dict:
        """Simulate a GRPO training step"""
        # Sample responses
        responses = self.sample_responses(prompt)
        
        # Compute rewards
        rewards = self.compute_rewards(responses, correct_answer)
        
        # Simulate log probs
        log_probs = np.random.randn(len(responses)) - 1
        old_log_probs = log_probs + np.random.randn(len(responses)) * 0.1
        
        # Compute objective
        objective = self.grpo_objective(log_probs, rewards, old_log_probs)
        
        self.reward_history.append(np.max(rewards))
        
        return {
            'num_samples': len(responses),
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'objective': objective
        }


class R1ReasoningFormat:
    """
    The R1 reasoning format uses <think> tags for extended thinking.
    """
    
    @staticmethod
    def format_response(thinking: str, answer: str) -> str:
        """Format a response in R1 style"""
        return f"<think>\n{thinking}\n</think>\n\n{answer}"
    
    @staticmethod
    def parse_response(response: str) -> Tuple[str, str]:
        """Parse thinking and answer from R1 format"""
        if "<think>" in response and "</think>" in response:
            think_start = response.index("<think>") + len("<think>")
            think_end = response.index("</think>")
            thinking = response[think_start:think_end].strip()
            answer = response[think_end + len("</think>"):].strip()
            return thinking, answer
        return "", response
    
    @staticmethod
    def example_trace() -> str:
        """Show an example R1 reasoning trace"""
        return """<think>
Let me solve this step by step.

First, I need to understand what the problem is asking...
The question asks for the sum of the first 10 prime numbers.

Let me list the first 10 prime numbers:
2, 3, 5, 7, 11, 13, 17, 19, 23, 29

Wait, let me verify these are all prime:
- 2: only divisible by 1 and 2 ✓
- 3: only divisible by 1 and 3 ✓
- 5: only divisible by 1 and 5 ✓
...
- 29: not divisible by 2, 3, 5. 29/7 ≈ 4.14, 29/11 ≈ 2.6. ✓

Now I'll add them up:
2 + 3 = 5
5 + 5 = 10
10 + 7 = 17
17 + 11 = 28
28 + 13 = 41
41 + 17 = 58
58 + 19 = 77
77 + 23 = 100
100 + 29 = 129

Let me double-check: 2+3+5+7+11+13+17+19+23+29
= (2+3+5) + (7+11+13) + (17+19+23) + 29
= 10 + 31 + 59 + 29
= 41 + 88
= 129 ✓

The answer is verified.
</think>

The sum of the first 10 prime numbers is **129**."""


def simulate_aha_moment():
    """Simulate the 'aha moment' phenomenon during training"""
    print("\n📊 Simulating Training 'Aha Moment'")
    print("-" * 50)
    
    # Simulate training progress with sudden improvement
    steps = np.arange(1000)
    
    # Base learning curve
    base_reward = 0.3 + 0.2 * (1 - np.exp(-steps / 200))
    
    # Add "aha moment" - sudden jump around step 400
    aha_effect = 0.3 * (1 / (1 + np.exp(-(steps - 400) / 20)))
    
    rewards = base_reward + aha_effect + np.random.randn(1000) * 0.05
    
    print("Training Progress (reward over steps):")
    for checkpoint in [0, 200, 399, 401, 600, 800, 999]:
        r = rewards[checkpoint]
        bar = "█" * int(r * 30)
        status = "← AHA MOMENT!" if checkpoint in [399, 401] else ""
        print(f"  Step {checkpoint:4d}: {r:.3f} {bar} {status}")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("DeepSeek-R1: Incentivizing Reasoning via Reinforcement Learning")
    print("Guo et al., 2025")
    print("=" * 70)
    
    print(DeepSeekR1Concepts.describe_training_process())
    print(DeepSeekR1Concepts.describe_emergent_behaviors())
    
    # Show example reasoning
    print("\n" + "=" * 70)
    print("📝 Example R1 Reasoning Trace")
    print("=" * 70)
    print(R1ReasoningFormat.example_trace())
    
    # GRPO simulation
    print("\n" + "=" * 70)
    print("📊 GRPO Training Simulation")
    print("=" * 70)
    
    grpo = GRPOSimulator(num_samples=8)
    
    for i in range(5):
        result = grpo.train_step(
            "What is 15 × 17?",
            "255"
        )
        print(f"Step {i+1}: Max Reward = {result['max_reward']:.3f}, "
              f"Objective = {result['objective']:.4f}")
    
    # Aha moment
    simulate_aha_moment()
    
    # Summary
    print("\n" + "=" * 70)
    print("📚 Why DeepSeek-R1 Matters")
    print("=" * 70)
    print("""
    1. Pure RL Works: Proved reasoning can emerge from reward only
    
    2. Self-Verification: Model learns to check its own work
    
    3. Emergent Capabilities: Complex behaviors without explicit teaching
    
    4. Open Research: Unlike OpenAI's o1, R1's methods are documented
    
    5. Efficient: Distilled versions (R1-7B) are very capable
    
    Key Insight:
        The right reward signal + enough scale = emergent reasoning
        No need for step-by-step supervision!
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
