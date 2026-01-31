#!/usr/bin/env python3
"""
Scaling Laws for Neural Language Models (Kaplan et al., 2020)
Implementation of scaling law analysis and predictions.

Paper: https://arxiv.org/abs/2001.08361
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ScalingLawConstants:
    """
    Constants from the Kaplan et al. paper
    
    The paper found power-law relationships:
    L(N) = (N_c / N)^α_N  - Loss as function of parameters
    L(D) = (D_c / D)^α_D  - Loss as function of data
    L(C) = (C_c / C)^α_C  - Loss as function of compute
    """
    # Power law exponents
    alpha_N: float = 0.076  # For parameters
    alpha_D: float = 0.095  # For data tokens
    alpha_C: float = 0.050  # For compute
    
    # Critical values (where loss = 1)
    N_c: float = 8.8e13    # Critical parameter count
    D_c: float = 5.4e13    # Critical data tokens
    C_c: float = 3.1e8     # Critical compute (PF-days)
    
    # Optimal allocation exponents
    a: float = 0.73        # N ∝ C^a (optimal parameter scaling)
    b: float = 0.27        # D ∝ C^b (optimal data scaling)


class ScalingLaws:
    """
    Implementation of Kaplan et al. scaling laws.
    
    Key findings:
    1. Loss scales as power laws in N, D, and C
    2. For fixed compute, there's an optimal N/D trade-off
    3. Larger models are more sample efficient
    """
    
    def __init__(self, constants: Optional[ScalingLawConstants] = None):
        self.c = constants or ScalingLawConstants()
    
    def loss_from_parameters(self, N: float) -> float:
        """
        Predict loss from number of parameters
        L(N) = (N_c / N)^α_N
        
        This represents the irreducible loss at infinite data
        """
        return (self.c.N_c / N) ** self.c.alpha_N
    
    def loss_from_data(self, D: float) -> float:
        """
        Predict loss from number of training tokens
        L(D) = (D_c / D)^α_D
        
        This represents the loss with infinite parameters
        """
        return (self.c.D_c / D) ** self.c.alpha_D
    
    def loss_from_compute(self, C: float) -> float:
        """
        Predict loss from compute budget (PF-days)
        L(C) = (C_c / C)^α_C
        
        This assumes optimal allocation between N and D
        """
        return (self.c.C_c / C) ** self.c.alpha_C
    
    def combined_loss(self, N: float, D: float) -> float:
        """
        Predict loss from both parameters and data
        
        L(N, D) ≈ [(N_c/N)^(α_N/α_D) + D_c/D]^α_D
        
        This combines data and parameter scaling.
        """
        term1 = (self.c.N_c / N) ** (self.c.alpha_N / self.c.alpha_D)
        term2 = self.c.D_c / D
        return (term1 + term2) ** self.c.alpha_D
    
    def optimal_parameters_for_compute(self, C: float) -> float:
        """
        Calculate optimal number of parameters for a compute budget
        N_opt ∝ C^a where a ≈ 0.73
        
        Key insight: Most compute should go to parameters, not data
        (This was later revised by Chinchilla)
        """
        # Simplified formula
        return 1.3e9 * (C ** self.c.a)
    
    def optimal_data_for_compute(self, C: float) -> float:
        """
        Calculate optimal amount of data for a compute budget
        D_opt ∝ C^b where b ≈ 0.27
        """
        return 2e10 * (C ** self.c.b)
    
    def sample_efficiency_ratio(self, N_large: float, N_small: float) -> float:
        """
        Calculate sample efficiency ratio between two model sizes.
        
        Larger models achieve the same loss with less data.
        Returns how many times more efficient the larger model is.
        """
        # Based on the paper's finding that larger models are more sample efficient
        ratio = (N_large / N_small) ** 0.4  # Approximate exponent
        return ratio
    
    def compute_for_training(
        self, 
        N: float, 
        D: float,
        flops_per_param_per_token: float = 6
    ) -> float:
        """
        Estimate compute for training in PetaFLOP-days
        
        C ≈ 6ND (approximate formula)
        """
        flops = flops_per_param_per_token * N * D
        pf_days = flops / (1e15 * 24 * 3600)
        return pf_days


class ScalingExperiment:
    """
    Simulate scaling experiments to verify power law relationships
    """
    
    def __init__(self, scaling_laws: ScalingLaws):
        self.laws = scaling_laws
    
    def run_parameter_sweep(
        self, 
        param_range: Tuple[float, float] = (1e6, 1e12),
        num_points: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Simulate experiments across parameter counts
        """
        params = np.logspace(
            np.log10(param_range[0]),
            np.log10(param_range[1]),
            num_points
        )
        
        losses = np.array([self.laws.loss_from_parameters(N) for N in params])
        
        return {
            'parameters': params,
            'loss': losses,
            'log_params': np.log10(params),
            'log_loss': np.log10(losses)
        }
    
    def run_data_sweep(
        self,
        data_range: Tuple[float, float] = (1e9, 1e13),
        num_points: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Simulate experiments across data sizes
        """
        data_sizes = np.logspace(
            np.log10(data_range[0]),
            np.log10(data_range[1]),
            num_points
        )
        
        losses = np.array([self.laws.loss_from_data(D) for D in data_sizes])
        
        return {
            'data': data_sizes,
            'loss': losses,
            'log_data': np.log10(data_sizes),
            'log_loss': np.log10(losses)
        }
    
    def run_compute_sweep(
        self,
        compute_range: Tuple[float, float] = (1e-3, 1e6),
        num_points: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Simulate experiments across compute budgets
        """
        compute = np.logspace(
            np.log10(compute_range[0]),
            np.log10(compute_range[1]),
            num_points
        )
        
        losses = np.array([self.laws.loss_from_compute(C) for C in compute])
        optimal_N = np.array([self.laws.optimal_parameters_for_compute(C) for C in compute])
        optimal_D = np.array([self.laws.optimal_data_for_compute(C) for C in compute])
        
        return {
            'compute': compute,
            'loss': losses,
            'optimal_N': optimal_N,
            'optimal_D': optimal_D
        }


def print_scaling_law_predictions():
    """Print predictions from scaling laws"""
    laws = ScalingLaws()
    
    print("\n📊 Parameter Scaling (L(N) = (N_c / N)^α_N)")
    print("-" * 50)
    print(f"{'Parameters':<15} {'Predicted Loss':<15}")
    print("-" * 50)
    
    for N in [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]:
        loss = laws.loss_from_parameters(N)
        N_str = f"{N:.0e}"
        print(f"{N_str:<15} {loss:.4f}")
    
    print("\n📊 Data Scaling (L(D) = (D_c / D)^α_D)")
    print("-" * 50)
    print(f"{'Tokens':<15} {'Predicted Loss':<15}")
    print("-" * 50)
    
    for D in [1e9, 1e10, 1e11, 1e12, 1e13]:
        loss = laws.loss_from_data(D)
        D_str = f"{D:.0e}"
        print(f"{D_str:<15} {loss:.4f}")
    
    print("\n📊 Compute Scaling (L(C) = (C_c / C)^α_C)")
    print("-" * 50)
    print(f"{'PF-days':<15} {'Predicted Loss':<15}")
    print("-" * 50)
    
    for C in [1, 10, 100, 1000, 10000, 100000]:
        loss = laws.loss_from_compute(C)
        print(f"{C:<15} {loss:.4f}")


def print_optimal_allocation():
    """Print optimal parameter/data allocation for compute budgets"""
    laws = ScalingLaws()
    
    print("\n📊 Optimal Allocation (Kaplan et al.)")
    print("-" * 70)
    print(f"{'Compute (PF-days)':<20} {'Optimal N':<20} {'Optimal D':<20}")
    print("-" * 70)
    
    for C in [1, 10, 100, 1000, 10000]:
        N_opt = laws.optimal_parameters_for_compute(C)
        D_opt = laws.optimal_data_for_compute(C)
        
        print(f"{C:<20} {N_opt:.2e}       {D_opt:.2e}")
    
    print("\n⚠️  Note: These allocations were later revised by Chinchilla (2022)")
    print("   Chinchilla showed data should be scaled more (closer to N = D)")


def compare_model_sizes():
    """Compare different model sizes"""
    laws = ScalingLaws()
    
    print("\n📊 Sample Efficiency Comparison")
    print("-" * 50)
    
    # Compare different model sizes
    models = [
        ("1B", 1e9),
        ("7B", 7e9),
        ("13B", 13e9),
        ("70B", 70e9),
        ("175B", 175e9)
    ]
    
    print(f"{'Model':<10} {'Parameters':<15} {'Loss @ 300B tokens':<20}")
    print("-" * 50)
    
    for name, N in models:
        loss = laws.combined_loss(N, 300e9)  # 300B tokens
        print(f"{name:<10} {N:.0e}       {loss:.4f}")
    
    # Sample efficiency
    print("\n📈 Sample Efficiency (relative to 1B model):")
    base_N = 1e9
    for name, N in models[1:]:
        ratio = laws.sample_efficiency_ratio(N, base_N)
        print(f"   {name} is {ratio:.1f}x more sample efficient than 1B")


def demo():
    """Main demonstration"""
    print("=" * 80)
    print("Scaling Laws for Neural Language Models (Kaplan et al., 2020)")
    print("=" * 80)
    
    print("\n📚 Key Findings:")
    print("-" * 40)
    print("""
    1. Power Law Relationships:
       - Loss scales as power laws with parameters, data, and compute
       - L(N) ∝ N^(-0.076)
       - L(D) ∝ D^(-0.095)
       - L(C) ∝ C^(-0.050)
    
    2. Optimal Compute Allocation:
       - For fixed compute budget, there's an optimal trade-off
       - Parameters should scale as C^0.73
       - Data should scale as C^0.27
       - (This means prioritize bigger models over more data)
    
    3. Sample Efficiency:
       - Larger models achieve same loss with less data
       - 10x larger model needs ~3x less data
    
    4. Smooth Scaling:
       - No sudden transitions or phase changes
       - Performance improves predictably
    """)
    
    # Print predictions
    print_scaling_law_predictions()
    print_optimal_allocation()
    compare_model_sizes()
    
    # Run experiment simulation
    print("\n" + "=" * 80)
    print("📊 Experiment Simulation")
    print("=" * 80)
    
    laws = ScalingLaws()
    experiment = ScalingExperiment(laws)
    
    # Parameter sweep
    results = experiment.run_parameter_sweep()
    print("\n📈 Parameter Sweep Results (log-log scale should be linear):")
    print(f"   Log10(N) range: [{results['log_params'].min():.1f}, {results['log_params'].max():.1f}]")
    print(f"   Log10(Loss) range: [{results['log_loss'].min():.3f}, {results['log_loss'].max():.3f}]")
    
    # Fit a line to verify power law
    slope, intercept = np.polyfit(results['log_params'], results['log_loss'], 1)
    print(f"   Fitted slope: {slope:.4f} (expected: {-laws.c.alpha_N:.4f})")
    
    print("\n" + "=" * 80)
    print("📚 Implications for Practitioners:")
    print("=" * 80)
    print("""
    1. If you have more compute, make the model bigger
    
    2. Training on more data has diminishing returns
       (unless you also scale up the model)
    
    3. You can predict final loss before training
       by extrapolating from smaller runs
    
    4. The power law is remarkably consistent across
       many orders of magnitude
    
    ⚠️  Important: Chinchilla (2022) later showed these allocations
        were suboptimal - data should be scaled more than Kaplan suggested.
    """)
    print("=" * 80)


if __name__ == "__main__":
    demo()
