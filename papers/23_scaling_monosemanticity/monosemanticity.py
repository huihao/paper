#!/usr/bin/env python3
"""
Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet
(Templeton et al., 2024)

The biggest leap in mechanistic interpretability - decomposing neural networks
into millions of interpretable features.

Paper: https://www.anthropic.com/research/scaling-monosemanticity
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Feature:
    """An interpretable feature extracted from the model"""
    feature_id: int
    label: str
    activation_examples: List[str]
    max_activation: float
    frequency: float  # How often is this feature active?


class SparseAutoencoder:
    """
    Sparse Autoencoder for Feature Extraction
    
    The key technique for monosemantic feature extraction.
    
    Architecture:
    - Encoder: Input → Large sparse hidden layer
    - Decoder: Sparse hidden → Reconstruction of input
    
    The sparse hidden layer learns interpretable features!
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_target: float = 0.01
    ):
        """
        Initialize sparse autoencoder.
        
        Args:
            input_dim: Size of model activations
            hidden_dim: Size of feature dictionary (typically much larger)
            sparsity_target: Target sparsity level
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_target = sparsity_target
        
        # Encoder weights
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.02
        self.b_enc = np.zeros(hidden_dim)
        
        # Decoder weights
        self.W_dec = np.random.randn(hidden_dim, input_dim) * 0.02
        self.b_dec = np.zeros(input_dim)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode activations to sparse feature space.
        
        Uses ReLU for sparsity.
        """
        pre_activation = np.matmul(x, self.W_enc) + self.b_enc
        return np.maximum(0, pre_activation)  # ReLU
    
    def decode(self, h: np.ndarray) -> np.ndarray:
        """Decode from feature space back to activation space"""
        return np.matmul(h, self.W_dec) + self.b_dec
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full forward pass.
        
        Returns:
            reconstruction: Reconstructed activations
            features: Sparse feature activations
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def compute_loss(
        self,
        x: np.ndarray,
        reconstruction: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute training loss.
        
        Loss = Reconstruction Loss + λ * Sparsity Loss
        """
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - reconstruction) ** 2)
        
        # Sparsity loss (L1 on features)
        sparsity_loss = np.mean(np.abs(features))
        
        # Combined loss
        total_loss = recon_loss + 0.1 * sparsity_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss,
            'active_features': np.mean(features > 0)
        }


class FeatureAnalyzer:
    """
    Analyze extracted features for interpretability.
    """
    
    @staticmethod
    def find_max_activating_examples(
        feature_activations: np.ndarray,
        texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find texts that maximally activate a feature.
        """
        indices = np.argsort(feature_activations)[-top_k:][::-1]
        return [(texts[i], feature_activations[i]) for i in indices]
    
    @staticmethod
    def compute_feature_statistics(
        features: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistics about feature activations"""
        return {
            'mean_activation': np.mean(features[features > 0]),
            'sparsity': np.mean(features == 0),
            'max_activation': np.max(features),
            'active_features_per_example': np.mean(np.sum(features > 0, axis=-1))
        }


def demonstrate_monosemanticity():
    """Demonstrate the monosemanticity concept"""
    print("=" * 70)
    print("Scaling Monosemanticity (Templeton et al., 2024)")
    print("=" * 70)
    
    print("""
The Problem: Polysemanticity

In neural networks, individual neurons often respond to many unrelated things.
This is called "polysemanticity" - one neuron, many meanings.

Example of a polysemantic neuron:
    Neuron 4792 activates for:
    - The color "blue"
    - The number "7"
    - References to sadness
    - Some French words
    
    This makes interpretation nearly impossible!

The Solution: Sparse Autoencoders

Train a sparse autoencoder on model activations.
The hidden layer learns "features" that are:
    1. Sparse (mostly inactive)
    2. Interpretable (one meaning each)
    3. Comprehensive (explain all behavior)

Example of a monosemantic feature:
    Feature 12847: "Golden Gate Bridge"
    - Activates only for Golden Gate Bridge mentions
    - Consistent across different contexts
    - Can be used to understand and steer model behavior
""")


def show_discovered_features():
    """Show examples of discovered features"""
    print("\n📊 Examples of Discovered Features")
    print("-" * 40)
    
    # Example features from the paper
    example_features = [
        Feature(
            feature_id=12847,
            label="Golden Gate Bridge",
            activation_examples=[
                "The Golden Gate Bridge spans the strait...",
                "I visited San Francisco and saw the famous red bridge...",
                "Engineering marvel of the 1930s..."
            ],
            max_activation=0.95,
            frequency=0.0001
        ),
        Feature(
            feature_id=23456,
            label="Recursive code patterns",
            activation_examples=[
                "def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)",
                "The function calls itself with smaller inputs...",
                "Base case prevents infinite recursion..."
            ],
            max_activation=0.88,
            frequency=0.0012
        ),
        Feature(
            feature_id=45678,
            label="Sycophantic agreement",
            activation_examples=[
                "You're absolutely right!",
                "That's a great point, I completely agree.",
                "I couldn't have said it better myself."
            ],
            max_activation=0.72,
            frequency=0.0034
        ),
        Feature(
            feature_id=67890,
            label="Safety/refusal concepts",
            activation_examples=[
                "I'm not able to help with that.",
                "That request goes against my values.",
                "I should decline this request."
            ],
            max_activation=0.91,
            frequency=0.0005
        ),
    ]
    
    for feature in example_features:
        print(f"\n🔍 Feature {feature.feature_id}: \"{feature.label}\"")
        print(f"   Max activation: {feature.max_activation:.2f}")
        print(f"   Frequency: {feature.frequency:.4f}")
        print(f"   Example activations:")
        for i, ex in enumerate(feature.activation_examples[:2], 1):
            print(f"      {i}. \"{ex[:50]}...\"")


def demonstrate_feature_steering():
    """Demonstrate how features can be used for steering"""
    print("\n" + "=" * 70)
    print("📊 Feature Steering Applications")
    print("=" * 70)
    
    print("""
Once you have interpretable features, you can:

1. Understanding Model Behavior:
   - See which features activate for a given input
   - Explain why the model made a decision
   - Find potential biases or issues

2. Steering Model Outputs:
   Original: "The Golden Gate Bridge is..."
   
   Suppress bridge feature → "The ... is a famous landmark"
   Amplify bridge feature → "The GOLDEN GATE BRIDGE, that magnificent..."

3. Safety Applications:
   - Monitor activation of deception features
   - Suppress harmful behavior features
   - Amplify helpfulness features

4. Red Teaming:
   - Find adversarial features
   - Understand what triggers unsafe behavior
   - Build better guardrails

Example: Sycophancy Steering
   High sycophancy feature → "You're so right! That's brilliant!"
   Suppressed feature → "I have a different perspective on this..."
""")


def show_scale_of_achievement():
    """Show the scale of what was achieved"""
    print("\n" + "=" * 70)
    print("📊 Scale of Achievement")
    print("=" * 70)
    
    print("""
The paper extracted features from Claude 3 Sonnet:

┌─────────────────────────────────────────────────┐
│ Model Size: Billions of parameters              │
│ Features Extracted: 34 million                  │
│ Interpretable Features: Millions confirmed      │
│ Feature Dictionary Size: 34M (1M to 34M tested) │
└─────────────────────────────────────────────────┘

This represents:
    - The largest-scale interpretability study ever
    - Proof that interpretability scales
    - Foundation for understanding frontier models
    
Key Finding:
    Larger sparse autoencoders find more features.
    1M features → Good coverage
    34M features → Even better, more specific features
""")


def demo():
    """Main demonstration"""
    demonstrate_monosemanticity()
    show_discovered_features()
    demonstrate_feature_steering()
    show_scale_of_achievement()
    
    # Demo sparse autoencoder
    print("\n" + "=" * 70)
    print("📊 Sparse Autoencoder Demo")
    print("-" * 40)
    
    sae = SparseAutoencoder(
        input_dim=64,
        hidden_dim=512,
        sparsity_target=0.01
    )
    
    # Sample activations
    x = np.random.randn(32, 64)  # 32 examples
    
    reconstruction, features = sae.forward(x)
    loss = sae.compute_loss(x, reconstruction, features)
    
    print(f"Input shape: {x.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Active features per example: {loss['active_features']*100:.1f}%")
    print(f"Sparsity: {100 - loss['active_features']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Breakthrough in Interpretability: Actually understand LLMs
    
    2. Scale Proof: Interpretability works on frontier models
    
    3. Safety Applications: Monitor and steer model behavior
    
    4. Scientific Understanding: What do neural networks learn?
    
    5. Future Direction: Path to fully interpretable AI
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
