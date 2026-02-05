#!/usr/bin/env python3
"""
Training Compute-Optimal Large Language Models (Chinchilla) (Hoffmann et al., 2022)
Implementation showing the revised scaling laws and optimal compute allocation.

Paper: https://arxiv.org/abs/2203.15556
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChinchillaConstants:
    """
    Chinchilla scaling law constants.
    
    The paper revised Kaplan's findings:
    - Parameters and data should scale equally with compute
    - Most models were undertrained (too big for their data budget)
    """
    # Power law exponents (revised)
    alpha: float = 0.34  # For parameters
    beta: float = 0.28   # For data
    
    # Optimal scaling exponents
    a: float = 0.50  # N ∝ C^a (vs Kaplan's 0.73)
    b: float = 0.50  # D ∝ C^b (vs Kaplan's 0.27)
    
    # Fitting constants (from the paper)
    E: float = 1.69  # Entropy of natural text
    A: float = 406.4
    B: float = 410.7


class ChinchillaScalingLaws:
    """
    Chinchilla's revised scaling laws.
    
    Key insight: For compute-optimal training, parameters and tokens
    should scale roughly equally.
    
    The optimal ratio is approximately:
    - 20 tokens per parameter
    - N_opt ≈ D_opt / 20
    """
    
    def __init__(self, constants: Optional[ChinchillaConstants] = None):
        self.c = constants or ChinchillaConstants()
        self.tokens_per_param = 20  # The famous Chinchilla ratio
    
    def loss_from_params_and_data(self, N: float, D: float) -> float:
        """
        Predict loss from parameters and data using Chinchilla formula.
        
        L(N, D) = E + A/N^α + B/D^β
        
        Where E is irreducible entropy of natural text.
        """
        return self.c.E + self.c.A / (N ** self.c.alpha) + self.c.B / (D ** self.c.beta)
    
    def optimal_params_for_compute(self, C_flops: float) -> float:
        """
        Calculate compute-optimal number of parameters.
        
        N_opt ∝ C^0.5 (vs Kaplan's C^0.73)
        
        This means for 10x more compute:
        - Kaplan: 5.4x more parameters
        - Chinchilla: 3.2x more parameters
        """
        # C ≈ 6ND, and N = D/20 for optimal
        # So C ≈ 6 * N * 20N = 120 * N^2
        # N = sqrt(C / 120)
        return np.sqrt(C_flops / 120)
    
    def optimal_tokens_for_compute(self, C_flops: float) -> float:
        """
        Calculate compute-optimal number of training tokens.
        
        D_opt ≈ 20 * N_opt
        """
        N_opt = self.optimal_params_for_compute(C_flops)
        return self.tokens_per_param * N_opt
    
    def compute_for_training(self, N: float, D: float) -> float:
        """
        Estimate training compute in FLOPs.
        
        C ≈ 6ND (forward + backward pass)
        """
        return 6 * N * D
    
    def is_compute_optimal(self, N: float, D: float, tolerance: float = 0.5) -> bool:
        """
        Check if a model is compute-optimal (within tolerance).
        
        Optimal when D/N ≈ 20
        """
        ratio = D / N
        return abs(ratio - self.tokens_per_param) / self.tokens_per_param < tolerance


class ModelComparison:
    """
    Compare different models according to Chinchilla scaling laws.
    """
    
    # Famous models and their training configurations
    MODELS = {
        'GPT-3': {'params': 175e9, 'tokens': 300e9},
        'Gopher': {'params': 280e9, 'tokens': 300e9},
        'Chinchilla': {'params': 70e9, 'tokens': 1.4e12},
        'LLaMA-7B': {'params': 7e9, 'tokens': 1e12},
        'LLaMA-13B': {'params': 13e9, 'tokens': 1e12},
        'LLaMA-65B': {'params': 65e9, 'tokens': 1.4e12},
        'Mistral-7B': {'params': 7e9, 'tokens': 8e12},  # Estimated
    }
    
    def __init__(self):
        self.laws = ChinchillaScalingLaws()
    
    def analyze_model(self, name: str, params: float, tokens: float) -> Dict:
        """
        Analyze a model's compute efficiency.
        """
        ratio = tokens / params
        is_optimal = self.laws.is_compute_optimal(params, tokens)
        compute = self.laws.compute_for_training(params, tokens)
        predicted_loss = self.laws.loss_from_params_and_data(params, tokens)
        
        # What would be optimal for this compute?
        optimal_N = self.laws.optimal_params_for_compute(compute)
        optimal_D = self.laws.optimal_tokens_for_compute(compute)
        optimal_loss = self.laws.loss_from_params_and_data(optimal_N, optimal_D)
        
        return {
            'name': name,
            'params': params,
            'tokens': tokens,
            'ratio': ratio,
            'is_optimal': is_optimal,
            'compute_flops': compute,
            'predicted_loss': predicted_loss,
            'optimal_params': optimal_N,
            'optimal_tokens': optimal_D,
            'optimal_loss': optimal_loss,
            'loss_gap': predicted_loss - optimal_loss
        }
    
    def compare_all_models(self) -> None:
        """Print comparison of all models"""
        print("\n📊 Model Compute Efficiency Analysis")
        print("=" * 90)
        print(f"{'Model':<15} {'Params':<12} {'Tokens':<12} {'Ratio':<10} {'Optimal?':<10} {'Loss Gap':<10}")
        print("-" * 90)
        
        for name, config in self.MODELS.items():
            analysis = self.analyze_model(name, config['params'], config['tokens'])
            params_str = f"{analysis['params']/1e9:.0f}B"
            tokens_str = f"{analysis['tokens']/1e12:.1f}T"
            ratio_str = f"{analysis['ratio']:.1f}"
            optimal_str = "✓ Yes" if analysis['is_optimal'] else "✗ No"
            gap_str = f"{analysis['loss_gap']:.4f}" if analysis['loss_gap'] > 0.001 else "~0"
            
            print(f"{name:<15} {params_str:<12} {tokens_str:<12} {ratio_str:<10} {optimal_str:<10} {gap_str:<10}")
        
        print("-" * 90)
        print(f"Optimal ratio: ~{self.laws.tokens_per_param} tokens per parameter")


def print_chinchilla_insights():
    """Print key insights from the Chinchilla paper"""
    laws = ChinchillaScalingLaws()
    
    print("\n📚 Chinchilla Key Insights")
    print("=" * 60)
    
    print("\n1️⃣ The 20:1 Rule")
    print("-" * 40)
    print("""
    For compute-optimal training:
    - Train on ~20 tokens per parameter
    - GPT-3 (175B params) should have used ~3.5T tokens (not 300B)
    - Most pre-Chinchilla models were severely undertrained
    """)
    
    print("\n2️⃣ Revised Scaling Exponents")
    print("-" * 40)
    print("""
    Kaplan (2020):          Chinchilla (2022):
    - N ∝ C^0.73            - N ∝ C^0.50
    - D ∝ C^0.27            - D ∝ C^0.50
    
    Data scaling matters much more than Kaplan suggested!
    """)
    
    print("\n3️⃣ Practical Implications")
    print("-" * 40)
    
    # Example: What's optimal for 10^23 FLOPs?
    compute = 1e23
    N_opt = laws.optimal_params_for_compute(compute)
    D_opt = laws.optimal_tokens_for_compute(compute)
    
    print(f"""
    For a compute budget of {compute:.0e} FLOPs:
    - Optimal parameters: {N_opt:.2e} ({N_opt/1e9:.1f}B)
    - Optimal tokens: {D_opt:.2e} ({D_opt/1e12:.1f}T)
    - Ratio: {D_opt/N_opt:.1f} tokens/param
    """)


def compare_kaplan_vs_chinchilla():
    """Compare Kaplan and Chinchilla recommendations"""
    print("\n📊 Kaplan vs Chinchilla: Optimal Allocation")
    print("=" * 70)
    
    laws = ChinchillaScalingLaws()
    
    # Kaplan's recommendations
    kaplan_a = 0.73
    kaplan_b = 0.27
    
    print(f"{'Compute (FLOPs)':<20} {'Kaplan N':<15} {'Chinchilla N':<15} {'Ratio':<10}")
    print("-" * 70)
    
    for C in [1e20, 1e21, 1e22, 1e23, 1e24]:
        # Kaplan's recommendation
        kaplan_N = 1e9 * (C / 1e20) ** kaplan_a
        
        # Chinchilla's recommendation
        chinchilla_N = laws.optimal_params_for_compute(C)
        
        ratio = kaplan_N / chinchilla_N
        
        C_str = f"{C:.0e}"
        kaplan_str = f"{kaplan_N/1e9:.1f}B"
        chinchilla_str = f"{chinchilla_N/1e9:.1f}B"
        
        print(f"{C_str:<20} {kaplan_str:<15} {chinchilla_str:<15} {ratio:.2f}x")
    
    print("\nKaplan recommends larger models with less data")
    print("Chinchilla recommends smaller models with more data")


def demonstrate_undertrained_models():
    """Show how undertrained pre-Chinchilla models were"""
    print("\n📉 How Undertrained Were Pre-Chinchilla Models?")
    print("=" * 70)
    
    laws = ChinchillaScalingLaws()
    
    undertrained = [
        ('GPT-3', 175e9, 300e9),
        ('Gopher', 280e9, 300e9),
        ('Megatron-Turing', 530e9, 270e9),
    ]
    
    print(f"{'Model':<20} {'Actual Tokens':<15} {'Optimal Tokens':<15} {'Undertrained by':<15}")
    print("-" * 70)
    
    for name, params, actual_tokens in undertrained:
        compute = laws.compute_for_training(params, actual_tokens)
        optimal_tokens = laws.optimal_tokens_for_compute(compute)
        
        # What tokens should they have trained on given their size?
        recommended = params * laws.tokens_per_param
        
        actual_str = f"{actual_tokens/1e12:.1f}T"
        optimal_str = f"{recommended/1e12:.1f}T"
        ratio = recommended / actual_tokens
        
        print(f"{name:<20} {actual_str:<15} {optimal_str:<15} {ratio:.1f}x")


def demo():
    """Main demonstration"""
    print("=" * 80)
    print("Training Compute-Optimal Large Language Models (Chinchilla)")
    print("Hoffmann et al., 2022")
    print("=" * 80)
    
    print_chinchilla_insights()
    
    comparison = ModelComparison()
    comparison.compare_all_models()
    
    compare_kaplan_vs_chinchilla()
    demonstrate_undertrained_models()
    
    print("\n" + "=" * 80)
    print("📚 Summary: Why Chinchilla Changed Everything")
    print("=" * 80)
    print("""
    1. Data Matters More Than We Thought
       - Kaplan suggested focusing on model size
       - Chinchilla showed equal scaling of N and D is optimal
    
    2. Smaller Models Can Match Larger Ones
       - Chinchilla 70B matched Gopher 280B
       - By training on 4x more tokens
       - At 1/4 the inference cost
    
    3. The Era of "Overtrained" Models
       - Post-Chinchilla: LLaMA, Mistral trained far beyond optimal
       - Inference efficiency matters more than training efficiency
       - LLaMA 7B trained on 1T tokens (142:1 ratio, not 20:1)
    
    4. Changed Industry Practice
       - Everyone started training on more data
       - Focus shifted to data quality and quantity
       - Led to data-centric AI movement
    """)
    print("=" * 80)


if __name__ == "__main__":
    demo()
