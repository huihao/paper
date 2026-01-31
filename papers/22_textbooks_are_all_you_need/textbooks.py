#!/usr/bin/env python3
"""
Textbooks Are All You Need (Gunasekar et al., 2023)
Demonstrates that high-quality synthetic data allows small models to outperform larger ones.

Paper: https://arxiv.org/abs/2306.11644
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment"""
    textbook_quality_score: float  # How educational/clear is the content?
    diversity_score: float  # How diverse are the topics?
    correctness_score: float  # How accurate is the content?
    complexity_progression: float  # Does it build from simple to complex?


class TextbookDataPhilosophy:
    """
    The Phi-1 / "Textbooks Are All You Need" Philosophy
    
    Key insight: Data quality > Data quantity
    
    By training on carefully curated, textbook-like data,
    a 1.3B model can outperform much larger models on coding tasks.
    """
    
    @staticmethod
    def describe_approach():
        return """
The Textbook Approach to Training:

Traditional Approach:
    - Scrape the internet
    - Train on everything
    - Hope the model learns from the noise
    - Bigger model = better results
    
Textbook Approach (Phi-1):
    - Curate high-quality, educational content
    - Generate synthetic "textbook" data
    - Train on clear, progressive explanations
    - Quality > Quantity
    
Key Principles:
    1. Educational value: Content should teach, not just contain
    2. Progressive complexity: Simple concepts first
    3. Clear explanations: No ambiguity
    4. Correctness: No errors or anti-patterns
    5. Diversity: Cover the topic space systematically
"""
    
    @staticmethod
    def show_phi_results():
        return """
Phi-1 Results (1.3B parameters):

Benchmark Comparison:
┌─────────────────────────────────────────────────────────┐
│ Model            │ Parameters │ HumanEval │ MBPP      │
├─────────────────────────────────────────────────────────┤
│ StarCoder        │ 15.5B      │ 33.6%     │ 43.6%     │
│ CodeLlama        │ 7B         │ 29.3%     │ 41.4%     │
│ Phi-1            │ 1.3B       │ 50.6%     │ 55.5%     │ ← !
│ Phi-1.5          │ 1.3B       │ 54.0%     │ 57.0%     │
└─────────────────────────────────────────────────────────┘

A 1.3B model outperforms 15B+ models!
The secret: textbook-quality training data.
"""


class SyntheticDataGenerator:
    """
    Simulated Synthetic Textbook Data Generator
    
    In the actual paper, GPT-3.5/4 was used to generate
    textbook-quality explanations and exercises.
    """
    
    def __init__(self, quality_threshold: float = 0.8):
        self.quality_threshold = quality_threshold
    
    def generate_code_textbook_entry(self, topic: str) -> Dict:
        """
        Generate a textbook-style entry for a coding topic.
        
        Structure:
        1. Clear explanation
        2. Simple example
        3. More complex example
        4. Common pitfalls
        5. Exercises
        """
        entry = {
            'topic': topic,
            'explanation': f"# {topic}\n\n"
                          f"{topic} is a fundamental concept in programming...\n\n"
                          f"Key points:\n"
                          f"1. First key point about {topic}\n"
                          f"2. Second key point about {topic}\n"
                          f"3. How to use {topic} correctly\n",
            
            'simple_example': f"# Simple example of {topic}\n"
                             f"def simple_example():\n"
                             f"    '''Demonstrates basic {topic}'''\n"
                             f"    # Implementation here\n"
                             f"    pass\n",
            
            'complex_example': f"# Complex example of {topic}\n"
                              f"def advanced_example():\n"
                              f"    '''Shows advanced usage of {topic}'''\n"
                              f"    # More sophisticated implementation\n"
                              f"    pass\n",
            
            'pitfalls': f"# Common mistakes with {topic}\n"
                       f"# 1. Don't do X because...\n"
                       f"# 2. Always remember to Y\n",
            
            'exercises': [
                f"Exercise 1: Implement a simple {topic} function",
                f"Exercise 2: Fix the bug in this {topic} code",
                f"Exercise 3: Optimize this {topic} implementation"
            ]
        }
        
        return entry
    
    def assess_quality(self, entry: Dict) -> DataQualityMetrics:
        """
        Assess the quality of a textbook entry.
        
        In practice, this might use an LLM to evaluate quality.
        """
        return DataQualityMetrics(
            textbook_quality_score=np.random.uniform(0.7, 1.0),
            diversity_score=np.random.uniform(0.6, 1.0),
            correctness_score=np.random.uniform(0.8, 1.0),
            complexity_progression=np.random.uniform(0.7, 1.0)
        )
    
    def filter_by_quality(self, entries: List[Dict]) -> List[Dict]:
        """Filter entries by quality threshold"""
        filtered = []
        for entry in entries:
            metrics = self.assess_quality(entry)
            avg_score = (metrics.textbook_quality_score + 
                        metrics.diversity_score + 
                        metrics.correctness_score + 
                        metrics.complexity_progression) / 4
            if avg_score >= self.quality_threshold:
                filtered.append(entry)
        return filtered


def compare_data_strategies():
    """Compare data strategies"""
    print("=" * 70)
    print("Textbooks Are All You Need (Gunasekar et al., 2023)")
    print("=" * 70)
    
    print(TextbookDataPhilosophy.describe_approach())
    print(TextbookDataPhilosophy.show_phi_results())
    
    print("\n📊 Data Strategy Comparison")
    print("-" * 40)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ Aspect           │ Web Scraping          │ Textbook Approach       │
├─────────────────────────────────────────────────────────────────────┤
│ Data Volume      │ Terabytes             │ Gigabytes               │
│ Quality Control  │ Minimal filtering     │ Careful curation        │
│ Noise Level      │ High                  │ Low                     │
│ Educational      │ Incidental            │ By design               │
│ Model Size Req.  │ Larger                │ Smaller                 │
│ Training Cost    │ Higher                │ Lower                   │
│ Performance      │ Scale-dependent       │ Quality-dependent       │
└─────────────────────────────────────────────────────────────────────┘
""")


def demonstrate_textbook_generation():
    """Demonstrate textbook data generation"""
    print("\n📚 Textbook Data Generation Demo")
    print("-" * 40)
    
    generator = SyntheticDataGenerator(quality_threshold=0.8)
    
    topics = ["Recursion", "Binary Search", "Dynamic Programming"]
    
    for topic in topics:
        entry = generator.generate_code_textbook_entry(topic)
        metrics = generator.assess_quality(entry)
        
        print(f"\n📖 Topic: {entry['topic']}")
        print(f"   Quality Score: {metrics.textbook_quality_score:.2f}")
        print(f"   Diversity: {metrics.diversity_score:.2f}")
        print(f"   Correctness: {metrics.correctness_score:.2f}")
        print(f"   Progression: {metrics.complexity_progression:.2f}")


def explain_phi_series():
    """Explain the Phi model series"""
    print("\n" + "=" * 70)
    print("📚 The Phi Model Series")
    print("=" * 70)
    
    print("""
Evolution of Phi Models:

Phi-1 (June 2023):
    - 1.3B parameters
    - Focused on Python coding
    - Trained on "textbook quality" data
    - Outperformed much larger code models

Phi-1.5 (September 2023):
    - 1.3B parameters
    - Extended to general language tasks
    - Synthetic data for common sense reasoning
    - Competitive with 10x larger models

Phi-2 (December 2023):
    - 2.7B parameters
    - Further improved data quality
    - State-of-the-art for size class
    - Matches 13B+ models on benchmarks

Phi-3 (April 2024):
    - 3.8B parameters
    - Matches Llama-3-8B on many tasks
    - Continued focus on data quality
    - Smallest model of its capability

The Pattern:
    Each version improves data quality
    rather than just adding parameters.
""")


def demo():
    """Main demonstration"""
    compare_data_strategies()
    demonstrate_textbook_generation()
    explain_phi_series()
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Changed the Narrative: Size isn't everything
    
    2. Data-Centric AI: Quality over quantity
    
    3. Efficiency: Smaller models with better data
    
    4. Synthetic Data: LLMs generating training data
    
    5. Practical Impact: Enabled edge deployment
    
    Key Insight:
        A 1.3B model trained on textbooks can beat
        a 15B model trained on internet noise.
        Data quality is the secret weapon.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
