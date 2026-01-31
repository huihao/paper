#!/usr/bin/env python3
"""
The Smol Training Playbook (Hugging Face, 2025)
Practical end-to-end handbook for efficiently training language models.

Reference: https://huggingface.co/spaces/HuggingFaceSmol/smol-training-playbook
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TrainingRecipe:
    """A training recipe for a specific model size"""
    model_size: str
    batch_size: int
    learning_rate: float
    warmup_steps: int
    total_tokens: str
    hardware: str
    training_time: str


class SmolTrainingPlaybook:
    """
    The Smol Training Playbook
    
    A comprehensive guide for training language models efficiently,
    from small (1B) to medium (7B) sizes.
    
    Key sections:
    1. Data preparation
    2. Model configuration
    3. Training infrastructure
    4. Hyperparameter selection
    5. Monitoring and debugging
    6. Evaluation
    """
    
    @staticmethod
    def overview():
        return """
The Smol Training Playbook - Key Principles

1. Start Small, Scale Up:
   - Begin with small experiments
   - Validate approach before scaling
   - Use scaling laws to predict behavior

2. Data Quality > Quantity:
   - Clean, deduplicated data
   - Domain-appropriate distribution
   - Quality filtering is essential

3. Infrastructure Efficiency:
   - Maximize hardware utilization
   - Use mixed precision (bf16)
   - Efficient data loading

4. Reproducibility:
   - Fixed seeds
   - Logged configurations
   - Version controlled experiments
"""

    @staticmethod
    def data_preparation_guide():
        return """
Data Preparation Checklist:

□ Collect and curate sources
  - Web text, books, code, etc.
  - License-appropriate data
  - Domain coverage

□ Clean and filter
  - Remove duplicates (exact and near)
  - Quality filtering (perplexity, length, etc.)
  - Remove toxic/harmful content
  
□ Format and tokenize
  - Consistent formatting
  - Tokenizer training or selection
  - Efficient storage format (Arrow, etc.)

□ Create mixtures
  - Domain proportions
  - Upsampling high-quality data
  - Curriculum considerations

□ Validate
  - Sample inspection
  - Distribution analysis
  - Decontamination from eval sets
"""

    @staticmethod
    def model_architecture_choices():
        return """
Model Architecture Decisions:

1. Vocabulary Size:
   - Typical: 32k-100k tokens
   - Trade-off: Efficiency vs. coverage
   - Recommendation: 32k-50k for English

2. Context Length:
   - Standard: 2048-4096
   - Extended: 8k-32k with RoPE scaling
   - Trade-off: Memory vs. capability

3. Architecture Choices:
   ┌────────────────────────────────────────────┐
   │ Component     │ Modern Default            │
   ├────────────────────────────────────────────┤
   │ Normalization │ RMSNorm (pre-norm)        │
   │ Activation    │ SwiGLU                    │
   │ Position      │ RoPE                      │
   │ Attention     │ GQA or MQA for efficiency │
   │ FFN ratio     │ 8/3 (SwiGLU)              │
   └────────────────────────────────────────────┘

4. Model Sizes:
   ┌────────────────────────────────────────────┐
   │ Size  │ Layers │ d_model │ Heads │ FFN    │
   ├────────────────────────────────────────────┤
   │ 1B    │ 24     │ 2048    │ 16    │ 5461   │
   │ 3B    │ 32     │ 2560    │ 20    │ 6827   │
   │ 7B    │ 32     │ 4096    │ 32    │ 10923  │
   └────────────────────────────────────────────┘
"""

    @staticmethod
    def training_hyperparameters():
        return """
Training Hyperparameters:

1. Learning Rate:
   - Peak LR: ~3e-4 (scales with size)
   - Warmup: ~2000 steps
   - Decay: Cosine to 0.1x peak

2. Batch Size:
   - Start: 2M tokens/batch
   - Scale: Up to 4M for larger models
   - Gradient accumulation as needed

3. Optimizer:
   - AdamW (β1=0.9, β2=0.95)
   - Weight decay: 0.1
   - Gradient clipping: 1.0

4. Precision:
   - BF16 for training
   - FP32 for optimizer states
   - Loss scaling not needed with BF16

5. Regularization:
   - Dropout: Usually 0 for pretraining
   - Weight decay: 0.1

Sample Recipe (7B model):
    ┌────────────────────────────────────┐
    │ Parameter      │ Value            │
    ├────────────────────────────────────┤
    │ Peak LR        │ 3e-4             │
    │ Warmup steps   │ 2000             │
    │ Batch size     │ 4M tokens        │
    │ Total tokens   │ 1-2T             │
    │ Optimizer      │ AdamW            │
    │ Weight decay   │ 0.1              │
    │ Gradient clip  │ 1.0              │
    └────────────────────────────────────┘
"""

    @staticmethod
    def training_infrastructure():
        return """
Training Infrastructure:

1. Hardware Options:
   - GPUs: H100, A100, RTX 4090
   - TPUs: v4, v5e
   - Cloud vs. on-prem trade-offs

2. Parallelism Strategies:
   ┌────────────────────────────────────────────┐
   │ Strategy       │ When to Use              │
   ├────────────────────────────────────────────┤
   │ Data Parallel  │ Model fits on one GPU    │
   │ FSDP/DeepSpeed │ Model too large for GPU  │
   │ Tensor Parallel│ Very large layers        │
   │ Pipeline       │ Very deep models         │
   └────────────────────────────────────────────┘

3. Libraries:
   - Transformers + Accelerate
   - DeepSpeed ZeRO
   - FSDP
   - Nanotron/Megatron for scale

4. Efficiency Tips:
   - Flash Attention 2
   - Gradient checkpointing
   - Efficient data loading
   - Profile before optimizing
"""

    @staticmethod
    def monitoring_and_debugging():
        return """
Monitoring and Debugging:

1. Key Metrics to Track:
   □ Training loss (smooth, per-step)
   □ Validation loss (regular intervals)
   □ Gradient norms
   □ Learning rate schedule
   □ Throughput (tokens/sec)

2. Common Issues:
   ┌────────────────────────────────────────────────────┐
   │ Issue              │ Solution                     │
   ├────────────────────────────────────────────────────┤
   │ Loss spikes        │ Lower LR, gradient clipping  │
   │ NaN loss           │ Check data, lower LR         │
   │ Slow convergence   │ Increase LR, check data      │
   │ Overfitting        │ More data, regularization    │
   │ OOM errors         │ Reduce batch, use FSDP       │
   └────────────────────────────────────────────────────┘

3. Debugging Tools:
   - TensorBoard / W&B
   - Gradient histograms
   - Activation statistics
   - Data sample inspection
"""

    @staticmethod
    def evaluation_guide():
        return """
Evaluation Guide:

1. During Training:
   - Validation loss on held-out set
   - Perplexity tracking
   - Quick sanity checks (generation)

2. Standard Benchmarks:
   ┌────────────────────────────────────────────┐
   │ Benchmark    │ Measures                   │
   ├────────────────────────────────────────────┤
   │ HellaSwag    │ Common sense reasoning     │
   │ ARC          │ Science knowledge          │
   │ MMLU         │ Multi-domain knowledge     │
   │ TriviaQA     │ Factual knowledge          │
   │ HumanEval    │ Code generation            │
   │ GSM8K        │ Math reasoning             │
   └────────────────────────────────────────────┘

3. Evaluation Tips:
   - Use consistent evaluation setup
   - Report few-shot settings
   - Consider multiple prompt formats
   - Watch for benchmark contamination
"""


def print_training_recipe():
    """Print sample training recipes"""
    recipes = [
        TrainingRecipe("1B", 2_000_000, 3e-4, 2000, "300B", "8x A100", "~3 days"),
        TrainingRecipe("3B", 2_000_000, 3e-4, 2000, "500B", "16x A100", "~1 week"),
        TrainingRecipe("7B", 4_000_000, 3e-4, 2000, "1T", "32x A100", "~2 weeks"),
    ]
    
    print("\n📊 Sample Training Recipes")
    print("=" * 80)
    print(f"{'Model':<8} {'Batch':<12} {'LR':<10} {'Warmup':<10} {'Tokens':<10} {'Hardware':<12} {'Time':<10}")
    print("-" * 80)
    
    for r in recipes:
        print(f"{r.model_size:<8} {r.batch_size:,}{'':<4} {r.learning_rate:<10} "
              f"{r.warmup_steps:<10} {r.total_tokens:<10} {r.hardware:<12} {r.training_time:<10}")


def demo():
    """Main demonstration"""
    playbook = SmolTrainingPlaybook()
    
    print("=" * 70)
    print("The Smol Training Playbook (Hugging Face, 2025)")
    print("=" * 70)
    
    print(playbook.overview())
    print(playbook.data_preparation_guide())
    print(playbook.model_architecture_choices())
    print(playbook.training_hyperparameters())
    print(playbook.training_infrastructure())
    print(playbook.monitoring_and_debugging())
    print(playbook.evaluation_guide())
    
    print_training_recipe()
    
    print("\n" + "=" * 70)
    print("📚 Why This Playbook Matters")
    print("=" * 70)
    print("""
    1. Practical Guide: End-to-end training knowledge
    
    2. Modern Defaults: Up-to-date best practices
    
    3. Efficiency Focus: Get more from less
    
    4. Reproducibility: Clear documentation
    
    5. Democratization: Make training accessible
    
    Key Insight:
        Training language models is now a well-understood
        engineering process. This playbook captures the
        collective knowledge of the community.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
