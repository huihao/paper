#!/usr/bin/env python3
"""
Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)
Implementation of Chain-of-Thought prompting techniques.

Paper: https://arxiv.org/abs/2201.11903
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ReasoningExample:
    """An example with step-by-step reasoning"""
    question: str
    chain_of_thought: str
    answer: str


class ChainOfThoughtPrompting:
    """
    Chain-of-Thought (CoT) Prompting
    
    Key insight: Including step-by-step reasoning in few-shot examples
    causes the model to generate its own reasoning, leading to better
    performance on complex reasoning tasks.
    
    Works best on:
    - Math word problems
    - Multi-step reasoning
    - Commonsense reasoning
    - Symbolic manipulation
    """
    
    def __init__(self):
        self.examples: List[ReasoningExample] = []
    
    def add_example(self, question: str, chain_of_thought: str, answer: str):
        """Add a reasoning example"""
        self.examples.append(ReasoningExample(question, chain_of_thought, answer))
    
    def create_standard_prompt(self, question: str) -> str:
        """
        Standard (non-CoT) few-shot prompt.
        Just shows question → answer without reasoning.
        """
        prompt_parts = []
        
        for ex in self.examples:
            prompt_parts.append(f"Q: {ex.question}")
            prompt_parts.append(f"A: {ex.answer}")
            prompt_parts.append("")
        
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A:")
        
        return "\n".join(prompt_parts)
    
    def create_cot_prompt(self, question: str) -> str:
        """
        Chain-of-Thought few-shot prompt.
        Shows question → reasoning → answer.
        """
        prompt_parts = []
        
        for ex in self.examples:
            prompt_parts.append(f"Q: {ex.question}")
            prompt_parts.append(f"A: {ex.chain_of_thought} Therefore, the answer is {ex.answer}.")
            prompt_parts.append("")
        
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A: Let's think step by step.")
        
        return "\n".join(prompt_parts)


class ZeroShotCoT:
    """
    Zero-Shot Chain-of-Thought
    
    Key insight: Simply adding "Let's think step by step" to the prompt
    can elicit reasoning without any examples!
    
    Paper: "Large Language Models are Zero-Shot Reasoners" (Kojima et al., 2022)
    """
    
    @staticmethod
    def create_prompt(question: str) -> str:
        """Create zero-shot CoT prompt"""
        return f"Q: {question}\nA: Let's think step by step."
    
    @staticmethod
    def extract_answer(response: str) -> str:
        """
        Extract final answer from CoT response.
        Often includes "Therefore" or "The answer is" patterns.
        """
        markers = ["therefore", "the answer is", "thus", "so the answer", "final answer"]
        response_lower = response.lower()
        
        for marker in markers:
            if marker in response_lower:
                idx = response_lower.rfind(marker)
                return response[idx:].strip()
        
        # Return last line as fallback
        lines = response.strip().split('\n')
        return lines[-1] if lines else response


class SelfConsistency:
    """
    Self-Consistency with Chain-of-Thought
    
    Key insight: Sample multiple reasoning paths and take the majority vote.
    Different reasoning paths may lead to the same correct answer.
    
    Paper: "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)
    """
    
    @staticmethod
    def majority_vote(answers: List[str]) -> str:
        """
        Take majority vote over multiple sampled answers.
        """
        from collections import Counter
        
        # Normalize answers
        normalized = [a.strip().lower() for a in answers]
        
        # Count occurrences
        counter = Counter(normalized)
        
        # Return most common
        most_common = counter.most_common(1)
        if most_common:
            return most_common[0][0]
        return answers[0] if answers else ""
    
    @staticmethod
    def sample_and_vote(
        question: str,
        cot_prompt: str,
        num_samples: int = 5
    ) -> Dict:
        """
        Sample multiple reasoning paths and vote.
        
        In practice, this calls the LLM multiple times with temperature > 0.
        Here we simulate the process.
        """
        # Simulate multiple samples (in practice, call LLM with temperature)
        simulated_answers = ["42", "42", "41", "42", "43"][:num_samples]
        
        final_answer = SelfConsistency.majority_vote(simulated_answers)
        
        return {
            'question': question,
            'samples': simulated_answers,
            'final_answer': final_answer,
            'agreement': simulated_answers.count(final_answer.lower()) / len(simulated_answers)
        }


def create_math_examples() -> List[ReasoningExample]:
    """Create math word problem examples with CoT"""
    return [
        ReasoningExample(
            question="Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
            chain_of_thought="Roger started with 5 balls. He bought 2 cans of 3 balls each, so 2 × 3 = 6 new balls. Total = 5 + 6 = 11.",
            answer="11 tennis balls"
        ),
        ReasoningExample(
            question="The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
            chain_of_thought="Started with 23 apples. Used 20, leaving 23 - 20 = 3 apples. Bought 6 more: 3 + 6 = 9.",
            answer="9 apples"
        ),
        ReasoningExample(
            question="If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            chain_of_thought="Started with 3 cars. 2 more arrived. Total = 3 + 2 = 5 cars.",
            answer="5 cars"
        ),
    ]


def create_commonsense_examples() -> List[ReasoningExample]:
    """Create commonsense reasoning examples with CoT"""
    return [
        ReasoningExample(
            question="Would a linguist study microorganisms?",
            chain_of_thought="A linguist studies languages and how they work. Microorganisms are tiny living things studied by biologists. These are different fields.",
            answer="No"
        ),
        ReasoningExample(
            question="Can a dog drive a car?",
            chain_of_thought="Driving requires understanding traffic rules, operating pedals, and steering. Dogs don't have the cognitive ability or physical capability for this.",
            answer="No"
        ),
    ]


def demo_standard_vs_cot():
    """Compare standard prompting vs Chain-of-Thought"""
    print("=" * 70)
    print("Chain-of-Thought Prompting Demo")
    print("=" * 70)
    
    cot = ChainOfThoughtPrompting()
    
    # Add math examples
    for ex in create_math_examples():
        cot.add_example(ex.question, ex.chain_of_thought, ex.answer)
    
    test_question = "A baker has 24 cupcakes. She sells 15 and then makes 12 more. How many cupcakes does she have?"
    
    print("\n📊 Standard Few-Shot Prompt (no reasoning):")
    print("-" * 40)
    standard_prompt = cot.create_standard_prompt(test_question)
    print(standard_prompt)
    
    print("\n📊 Chain-of-Thought Prompt (with reasoning):")
    print("-" * 40)
    cot_prompt = cot.create_cot_prompt(test_question)
    print(cot_prompt)
    
    print("\n💡 Expected CoT Response:")
    print("-" * 40)
    print("""The baker started with 24 cupcakes. She sold 15, leaving 24 - 15 = 9 cupcakes.
Then she made 12 more: 9 + 12 = 21 cupcakes.
Therefore, the answer is 21 cupcakes.""")


def demo_zero_shot_cot():
    """Demonstrate zero-shot CoT"""
    print("\n" + "=" * 70)
    print("Zero-Shot Chain-of-Thought Demo")
    print("=" * 70)
    
    question = "If a train travels 60 miles per hour for 2.5 hours, how far does it go?"
    
    print("\n📊 Standard Zero-Shot Prompt:")
    print("-" * 40)
    print(f"Q: {question}")
    print("A:")
    
    print("\n📊 Zero-Shot CoT Prompt:")
    print("-" * 40)
    prompt = ZeroShotCoT.create_prompt(question)
    print(prompt)
    
    print("\n💡 The magic phrase 'Let's think step by step' triggers reasoning!")


def demo_self_consistency():
    """Demonstrate self-consistency"""
    print("\n" + "=" * 70)
    print("Self-Consistency Demo")
    print("=" * 70)
    
    question = "What is 17 × 24?"
    
    print(f"\n❓ Question: {question}")
    print("\n📊 Multiple Reasoning Paths (simulated):")
    print("-" * 40)
    
    # Simulate different reasoning paths
    paths = [
        "17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408",
        "17 × 24 = 17 × 25 - 17 = 425 - 17 = 408",
        "17 × 24 = 20 × 24 - 3 × 24 = 480 - 72 = 408",
        "17 × 24 = 10 × 24 + 7 × 24 = 240 + 168 = 408",
        "17 × 24 = (16 + 1) × 24 = 384 + 24 = 408"
    ]
    
    for i, path in enumerate(paths, 1):
        print(f"   Path {i}: {path}")
    
    result = SelfConsistency.sample_and_vote(question, "", num_samples=5)
    result['final_answer'] = "408"  # Override simulation
    result['samples'] = ["408"] * 5
    result['agreement'] = 1.0
    
    print(f"\n✅ Final Answer (majority vote): {result['final_answer']}")
    print(f"   Agreement: {result['agreement']:.0%}")


def explain_cot_findings():
    """Explain key findings from the CoT paper"""
    print("\n" + "=" * 70)
    print("📚 Key Findings from the Chain-of-Thought Paper")
    print("=" * 70)
    
    print("""
1. Scale Matters:
   - CoT provides little benefit for small models
   - Emergent at ~100B parameters
   - Works well with GPT-3, PaLM, etc.

2. Task Complexity:
   - More benefit on complex reasoning tasks
   - Less benefit on simple retrieval tasks
   - Best for multi-step problems

3. Prompt Design:
   - Reasoning steps should be clear and logical
   - 4-8 examples works well
   - Quality of reasoning matters more than quantity

4. Error Analysis:
   - When CoT fails, errors are often in reasoning
   - Can debug by examining the chain
   - More interpretable than direct answers

5. Related Techniques:
   - Zero-shot CoT: "Let's think step by step"
   - Self-consistency: Sample and vote
   - Least-to-most: Break into subproblems
   - Tree-of-thought: Explore multiple branches
""")


def demo():
    """Main demonstration"""
    demo_standard_vs_cot()
    demo_zero_shot_cot()
    demo_self_consistency()
    explain_cot_findings()
    
    print("\n" + "=" * 70)
    print("📚 Why Chain-of-Thought Matters")
    print("=" * 70)
    print("""
    1. Unlocked Reasoning: Showed LLMs can reason, just need the right prompt
    
    2. Interpretability: Can see how the model arrives at answers
    
    3. Foundation for R1: Led to reasoning-focused training (DeepSeek-R1, o1)
    
    4. Changed Prompting: Made "step by step" a standard technique
    
    5. Research Direction: Opened entire field of reasoning in LLMs
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
