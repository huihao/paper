#!/usr/bin/env python3
"""
Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
Ouyang et al., 2022

Implementation of the RLHF pipeline: SFT → Reward Model → PPO.

Paper: https://arxiv.org/abs/2203.02155
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Demonstration:
    """A demonstration for supervised fine-tuning"""
    prompt: str
    response: str
    

@dataclass
class Comparison:
    """A comparison for reward model training"""
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Less preferred response


class SupervisedFineTuning:
    """
    Step 1: Supervised Fine-Tuning (SFT)
    
    Train a pretrained LLM on high-quality demonstrations
    of desired behavior.
    """
    
    def __init__(self, base_model_params: int = 1e9):
        self.base_model_params = base_model_params
        self.sft_loss_history = []
    
    def prepare_dataset(self, demonstrations: List[Demonstration]) -> Dict:
        """
        Prepare dataset for SFT training.
        Format: prompt + response pairs from human labelers.
        """
        return {
            'prompts': [d.prompt for d in demonstrations],
            'responses': [d.response for d in demonstrations],
            'num_examples': len(demonstrations)
        }
    
    def train(self, dataset: Dict, epochs: int = 3) -> Dict:
        """
        Simulate SFT training.
        In practice, this is standard language model fine-tuning.
        """
        print("📚 Step 1: Supervised Fine-Tuning (SFT)")
        print("-" * 40)
        
        for epoch in range(epochs):
            # Simulate training loss decreasing
            loss = 2.5 * np.exp(-0.5 * epoch) + np.random.uniform(0, 0.1)
            self.sft_loss_history.append(loss)
            print(f"   Epoch {epoch+1}: Loss = {loss:.4f}")
        
        return {
            'final_loss': self.sft_loss_history[-1],
            'epochs': epochs,
            'model': 'sft_model'
        }


class RewardModel:
    """
    Step 2: Reward Model Training
    
    Train a model to predict human preferences.
    Given a prompt and two responses, predict which is better.
    """
    
    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size
        # Simple reward head
        self.reward_head = np.random.randn(hidden_size) * 0.02
        self.loss_history = []
    
    def prepare_comparison_dataset(self, comparisons: List[Comparison]) -> Dict:
        """
        Prepare comparison dataset.
        Each example has a prompt with chosen and rejected responses.
        """
        return {
            'prompts': [c.prompt for c in comparisons],
            'chosen': [c.chosen for c in comparisons],
            'rejected': [c.rejected for c in comparisons],
            'num_comparisons': len(comparisons)
        }
    
    def compute_reward(self, hidden_state: np.ndarray) -> float:
        """Compute scalar reward from hidden state"""
        return np.dot(hidden_state, self.reward_head)
    
    def compute_loss(self, reward_chosen: float, reward_rejected: float) -> float:
        """
        Bradley-Terry loss for preference learning.
        
        L = -log(sigmoid(r_chosen - r_rejected))
        
        We want reward_chosen > reward_rejected.
        """
        diff = reward_chosen - reward_rejected
        loss = -np.log(1 / (1 + np.exp(-diff)) + 1e-10)
        return loss
    
    def train(self, dataset: Dict, epochs: int = 1) -> Dict:
        """
        Simulate reward model training.
        """
        print("\n📊 Step 2: Reward Model Training")
        print("-" * 40)
        
        n_comparisons = dataset['num_comparisons']
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(n_comparisons):
                # Simulate hidden states
                hidden_chosen = np.random.randn(self.hidden_size)
                hidden_rejected = np.random.randn(self.hidden_size) - 0.1
                
                r_chosen = self.compute_reward(hidden_chosen)
                r_rejected = self.compute_reward(hidden_rejected)
                
                loss = self.compute_loss(r_chosen, r_rejected)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return {
            'final_loss': self.loss_history[-1],
            'model': 'reward_model'
        }


class PPOTrainer:
    """
    Step 3: Proximal Policy Optimization (PPO)
    
    Fine-tune the SFT model using rewards from the reward model.
    """
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.1,
        value_coeff: float = 0.5
    ):
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.value_coeff = value_coeff
        self.reward_history = []
        self.kl_history = []
    
    def compute_advantages(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * last_advantage
        
        return advantages
    
    def ppo_loss(
        self,
        old_log_probs: np.ndarray,
        new_log_probs: np.ndarray,
        advantages: np.ndarray
    ) -> float:
        """
        PPO clipped objective.
        
        L = min(r * A, clip(r, 1-ε, 1+ε) * A)
        where r = exp(new_log_prob - old_log_prob)
        """
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        
        return -np.mean(np.minimum(loss1, loss2))
    
    def kl_penalty(self, old_log_probs: np.ndarray, new_log_probs: np.ndarray) -> float:
        """
        KL divergence penalty to prevent policy from deviating too far.
        """
        return np.mean(old_log_probs - new_log_probs)
    
    def train_step(
        self,
        prompts: List[str],
        reward_model: RewardModel,
        num_steps: int = 100
    ) -> Dict:
        """
        Simulate PPO training step.
        """
        print("\n🎯 Step 3: PPO Training")
        print("-" * 40)
        
        for step in range(num_steps):
            # Simulate generating responses and getting rewards
            batch_rewards = []
            batch_kl = []
            
            for prompt in prompts[:min(16, len(prompts))]:  # Mini-batch
                # Simulate reward from reward model
                hidden = np.random.randn(reward_model.hidden_size)
                reward = reward_model.compute_reward(hidden)
                batch_rewards.append(reward)
                
                # Simulate KL divergence
                kl = np.abs(np.random.randn()) * 0.1
                batch_kl.append(kl)
            
            avg_reward = np.mean(batch_rewards)
            avg_kl = np.mean(batch_kl)
            
            self.reward_history.append(avg_reward)
            self.kl_history.append(avg_kl)
            
            if (step + 1) % 20 == 0:
                print(f"   Step {step+1}: Reward = {avg_reward:.4f}, KL = {avg_kl:.4f}")
        
        return {
            'final_reward': np.mean(self.reward_history[-10:]),
            'final_kl': np.mean(self.kl_history[-10:])
        }


class InstructGPTPipeline:
    """
    Complete InstructGPT RLHF Pipeline
    
    1. Collect demonstrations → SFT
    2. Collect comparisons → Reward Model
    3. Use RM to optimize policy with PPO
    """
    
    def __init__(self):
        self.sft = SupervisedFineTuning()
        self.reward_model = RewardModel()
        self.ppo = PPOTrainer()
    
    def run_pipeline(
        self,
        demonstrations: List[Demonstration],
        comparisons: List[Comparison],
        prompts_for_ppo: List[str]
    ) -> Dict:
        """
        Run the complete RLHF pipeline.
        """
        print("=" * 60)
        print("InstructGPT RLHF Pipeline")
        print("=" * 60)
        
        # Step 1: SFT
        sft_dataset = self.sft.prepare_dataset(demonstrations)
        sft_result = self.sft.train(sft_dataset)
        
        # Step 2: Reward Model
        rm_dataset = self.reward_model.prepare_comparison_dataset(comparisons)
        rm_result = self.reward_model.train(rm_dataset)
        
        # Step 3: PPO
        ppo_result = self.ppo.train_step(prompts_for_ppo, self.reward_model)
        
        return {
            'sft': sft_result,
            'reward_model': rm_result,
            'ppo': ppo_result
        }


def create_sample_data() -> Tuple[List[Demonstration], List[Comparison], List[str]]:
    """Create sample data for demonstration"""
    
    demonstrations = [
        Demonstration(
            prompt="Explain photosynthesis to a 5-year-old.",
            response="Plants are like little food factories! They use sunlight, water, and air to make their own food. It's like magic cooking!"
        ),
        Demonstration(
            prompt="What is the capital of France?",
            response="The capital of France is Paris. It's a beautiful city known for the Eiffel Tower and great food!"
        ),
        Demonstration(
            prompt="Write a haiku about coding.",
            response="Lines of code appear\nBugs hide in the syntax deep\nDebug, test, repeat"
        ),
    ]
    
    comparisons = [
        Comparison(
            prompt="How do I make pasta?",
            chosen="Here's a simple pasta recipe: 1) Boil water with salt. 2) Add pasta and cook for 8-10 minutes. 3) Drain and add your favorite sauce. Enjoy!",
            rejected="Just boil it."
        ),
        Comparison(
            prompt="Tell me a joke.",
            chosen="Why don't scientists trust atoms? Because they make up everything! 😄",
            rejected="I don't know any jokes."
        ),
    ]
    
    prompts_for_ppo = [
        "What is machine learning?",
        "How do computers work?",
        "Explain gravity simply.",
        "What makes the sky blue?",
    ]
    
    return demonstrations, comparisons, prompts_for_ppo


def explain_rlhf():
    """Explain the RLHF process"""
    print("\n" + "=" * 60)
    print("📚 Understanding RLHF")
    print("=" * 60)
    
    print("""
The RLHF Pipeline:

1️⃣ Supervised Fine-Tuning (SFT)
   • Collect high-quality demonstrations from labelers
   • Fine-tune pretrained LLM on these demonstrations
   • Result: Model that can follow instructions
   
2️⃣ Reward Model (RM) Training
   • Collect comparison data: "Which response is better?"
   • Train a model to predict human preferences
   • Uses Bradley-Terry model: P(A > B) = σ(r_A - r_B)
   
3️⃣ Policy Optimization (PPO)
   • Generate responses from current policy
   • Score responses with reward model
   • Update policy to maximize reward
   • KL penalty prevents diverging too far from SFT model

Key Insights:
   • SFT alone isn't enough (can still produce harmful outputs)
   • Reward model captures nuanced human preferences
   • PPO balances reward optimization with staying close to SFT
   • The KL penalty is crucial for stability
""")


def demo():
    """Main demonstration"""
    # Create sample data
    demonstrations, comparisons, prompts = create_sample_data()
    
    # Run pipeline
    pipeline = InstructGPTPipeline()
    results = pipeline.run_pipeline(demonstrations, comparisons, prompts)
    
    # Explain RLHF
    explain_rlhf()
    
    print("\n" + "=" * 60)
    print("📊 Pipeline Results Summary")
    print("=" * 60)
    print(f"""
    SFT:
      Final Loss: {results['sft']['final_loss']:.4f}
      
    Reward Model:
      Final Loss: {results['reward_model']['final_loss']:.4f}
      
    PPO:
      Final Reward: {results['ppo']['final_reward']:.4f}
      Final KL: {results['ppo']['final_kl']:.4f}
""")
    
    print("=" * 60)
    print("📚 Impact of InstructGPT")
    print("=" * 60)
    print("""
    1. Created the modern alignment paradigm
    2. Showed RLHF makes models more helpful and less harmful
    3. Led directly to ChatGPT
    4. Became the blueprint for instruction-tuned models
    5. Demonstrated the importance of human feedback
    """)
    print("=" * 60)


if __name__ == "__main__":
    demo()
