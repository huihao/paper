#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) (Rafailov et al., 2023)
Implementation of DPO as a simpler alternative to PPO-based RLHF.

Paper: https://arxiv.org/abs/2305.18290
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreferencePair:
    """A preference pair for DPO training"""
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Less preferred response


def log_sigmoid(x: float) -> float:
    """Numerically stable log-sigmoid"""
    if x >= 0:
        return -np.log(1 + np.exp(-x))
    else:
        return x - np.log(1 + np.exp(x))


class DPOTrainer:
    """
    Direct Preference Optimization (DPO)
    
    Key insight: The optimal policy under the RLHF objective has a closed form!
    
    Instead of:
    1. Train reward model
    2. Use RL (PPO) to optimize
    
    DPO directly optimizes the policy using preference data:
    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    
    where:
    - y_w = winning (chosen) response
    - y_l = losing (rejected) response
    - π = policy being trained
    - π_ref = reference policy (frozen SFT model)
    - β = temperature parameter
    """
    
    def __init__(self, beta: float = 0.1):
        """
        Initialize DPO trainer.
        
        Args:
            beta: Temperature parameter controlling deviation from reference
        """
        self.beta = beta
        self.loss_history = []
        self.accuracy_history = []
    
    def compute_log_prob(
        self, 
        response: str, 
        prompt: str, 
        model_params: np.ndarray
    ) -> float:
        """
        Compute log probability of response given prompt.
        
        In practice, this sums log probabilities of each token.
        Here we simulate with a simple score.
        """
        # Simulate log probability based on model params and response
        combined = prompt + response
        score = np.sum([model_params[hash(c) % len(model_params)] for c in combined[:50]])
        return score / 10.0
    
    def dpo_loss(
        self,
        log_prob_chosen_policy: float,
        log_prob_rejected_policy: float,
        log_prob_chosen_ref: float,
        log_prob_rejected_ref: float
    ) -> Tuple[float, bool]:
        """
        Compute DPO loss for a single preference pair.
        
        L = -log σ(β * ((log π(y_w|x) - log π_ref(y_w|x)) - 
                        (log π(y_l|x) - log π_ref(y_l|x))))
        
        Returns:
            Tuple of (loss, correct_prediction)
        """
        # Compute log-ratio differences
        chosen_logratios = log_prob_chosen_policy - log_prob_chosen_ref
        rejected_logratios = log_prob_rejected_policy - log_prob_rejected_ref
        
        # The implicit reward difference
        reward_diff = self.beta * (chosen_logratios - rejected_logratios)
        
        # DPO loss
        loss = -log_sigmoid(reward_diff)
        
        # Check if policy prefers the chosen response
        correct = reward_diff > 0
        
        return loss, correct
    
    def train_step(
        self,
        preference_pairs: List[PreferencePair],
        policy_params: np.ndarray,
        ref_params: np.ndarray
    ) -> Tuple[float, float]:
        """
        Single training step on a batch of preference pairs.
        
        Args:
            preference_pairs: List of (prompt, chosen, rejected) tuples
            policy_params: Current policy parameters
            ref_params: Reference (frozen SFT) parameters
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        losses = []
        correct_count = 0
        
        for pair in preference_pairs:
            # Compute log probs under policy
            log_prob_chosen_policy = self.compute_log_prob(
                pair.chosen, pair.prompt, policy_params
            )
            log_prob_rejected_policy = self.compute_log_prob(
                pair.rejected, pair.prompt, policy_params
            )
            
            # Compute log probs under reference
            log_prob_chosen_ref = self.compute_log_prob(
                pair.chosen, pair.prompt, ref_params
            )
            log_prob_rejected_ref = self.compute_log_prob(
                pair.rejected, pair.prompt, ref_params
            )
            
            # Compute loss
            loss, correct = self.dpo_loss(
                log_prob_chosen_policy,
                log_prob_rejected_policy,
                log_prob_chosen_ref,
                log_prob_rejected_ref
            )
            
            losses.append(loss)
            if correct:
                correct_count += 1
        
        avg_loss = np.mean(losses)
        accuracy = correct_count / len(preference_pairs)
        
        return avg_loss, accuracy
    
    def train(
        self,
        preference_data: List[PreferencePair],
        num_epochs: int = 3,
        batch_size: int = 4
    ) -> dict:
        """
        Train using DPO.
        
        Args:
            preference_data: List of preference pairs
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training results dictionary
        """
        print("=" * 60)
        print("Direct Preference Optimization (DPO) Training")
        print("=" * 60)
        
        # Initialize parameters (simulated)
        param_size = 1000
        policy_params = np.random.randn(param_size) * 0.01
        ref_params = policy_params.copy()  # Reference is frozen SFT model
        
        print(f"\nβ (beta): {self.beta}")
        print(f"Preference pairs: {len(preference_data)}")
        print(f"Epochs: {num_epochs}")
        print("-" * 40)
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            # Process in batches
            for i in range(0, len(preference_data), batch_size):
                batch = preference_data[i:i+batch_size]
                loss, acc = self.train_step(batch, policy_params, ref_params)
                epoch_losses.append(loss)
                epoch_accuracies.append(acc)
                
                # Simulate gradient update (very simplified)
                policy_params += np.random.randn(param_size) * 0.001
            
            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accuracies)
            
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_acc)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.2%}")
        
        return {
            'final_loss': self.loss_history[-1],
            'final_accuracy': self.accuracy_history[-1],
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }


def compare_dpo_vs_ppo():
    """Compare DPO and PPO approaches"""
    print("\n" + "=" * 60)
    print("📊 DPO vs PPO Comparison")
    print("=" * 60)
    
    print("""
PPO-based RLHF:
    1. Train reward model on preference data
    2. Sample from policy
    3. Score samples with reward model
    4. Update policy with PPO
    5. Repeat steps 2-4 many times
    
    Challenges:
    - Reward model can be exploited (reward hacking)
    - PPO is sensitive to hyperparameters
    - Requires careful KL penalty tuning
    - Complex implementation

DPO:
    1. Use preference data directly
    2. Compute log-ratio loss
    3. Update policy
    4. Done!
    
    Benefits:
    - No reward model needed
    - No RL sampling loop
    - More stable optimization
    - Simpler to implement
    - Theoretically equivalent under certain assumptions
""")
    
    print("\n📐 The DPO Insight:")
    print("-" * 40)
    print("""
The optimal policy under the RLHF objective:
    
    π*(y|x) ∝ π_ref(y|x) * exp(r(x,y) / β)

Can be rearranged to express reward as:

    r(x,y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)
    
This means we can optimize directly for the policy without 
explicitly learning the reward function!
""")


def explain_dpo_loss():
    """Explain the DPO loss function"""
    print("\n" + "=" * 60)
    print("📐 Understanding the DPO Loss")
    print("=" * 60)
    
    print("""
DPO Loss:
    
    L_DPO(π; π_ref) = -E[(x, y_w, y_l)] [
        log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    ]
    
Breaking it down:

1. log π(y|x) / π_ref(y|x) is the log-ratio of policy to reference
   - Positive when policy assigns higher prob than reference
   - This is the "implicit reward" for response y

2. Difference between chosen and rejected log-ratios
   - We want: log-ratio(chosen) > log-ratio(rejected)
   - This means the policy should prefer the chosen response
   
3. σ (sigmoid) converts to probability
   - Loss is low when policy strongly prefers chosen
   - Loss is high when policy prefers rejected

4. β controls regularization strength
   - High β: Stay close to reference policy
   - Low β: Can deviate more from reference
""")


def create_sample_preference_data() -> List[PreferencePair]:
    """Create sample preference data for demonstration"""
    return [
        PreferencePair(
            prompt="What is the capital of France?",
            chosen="The capital of France is Paris, a beautiful city known for the Eiffel Tower and rich culture.",
            rejected="idk maybe london"
        ),
        PreferencePair(
            prompt="Explain quantum computing simply.",
            chosen="Quantum computers use quantum bits (qubits) that can be 0, 1, or both at once, allowing them to solve certain problems much faster than regular computers.",
            rejected="It's just really fast computers."
        ),
        PreferencePair(
            prompt="How do I make friends?",
            chosen="Making friends takes time and effort. Try joining clubs or activities you enjoy, be a good listener, show genuine interest in others, and be patient as relationships develop naturally.",
            rejected="Just talk to people I guess."
        ),
        PreferencePair(
            prompt="Write a short poem about the moon.",
            chosen="Silver orb in velvet night,\nGuiding sailors with your light,\nPhases waxing, ever changing,\nThrough the cosmos, freely ranging.",
            rejected="Moon is bright. The end."
        ),
        PreferencePair(
            prompt="What should I do if I'm feeling stressed?",
            chosen="When stressed, try deep breathing exercises, take a short walk, or talk to someone you trust. Regular exercise, adequate sleep, and breaking tasks into smaller steps can also help manage stress.",
            rejected="Just don't be stressed."
        ),
    ]


def demo():
    """Main demonstration"""
    # Create preference data
    preference_data = create_sample_preference_data()
    
    # Initialize trainer
    trainer = DPOTrainer(beta=0.1)
    
    # Train
    results = trainer.train(preference_data, num_epochs=5)
    
    # Explanations
    compare_dpo_vs_ppo()
    explain_dpo_loss()
    
    # Summary
    print("\n" + "=" * 60)
    print("📚 Summary")
    print("=" * 60)
    print("""
DPO Key Takeaways:

1. Eliminates the need for a separate reward model
2. No RL sampling loop - just supervised learning on preferences
3. Theoretically equivalent to RLHF under certain assumptions
4. More stable and easier to implement than PPO
5. Widely adopted: LLaMA-2-Chat, many open models

Variants and Extensions:
- IPO: Identity Preference Optimization
- KTO: Kahneman-Tversky Optimization (unpaired preferences)
- ORPO: Odds Ratio Preference Optimization
- SimPO: Simple Preference Optimization
""")
    print("=" * 60)


if __name__ == "__main__":
    demo()
