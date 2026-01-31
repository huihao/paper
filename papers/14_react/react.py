#!/usr/bin/env python3
"""
ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
Implementation of the ReAct framework for agentic systems.

Paper: https://arxiv.org/abs/2210.03629
"""

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions an agent can take"""
    SEARCH = "search"
    LOOKUP = "lookup"
    CALCULATE = "calculate"
    FINISH = "finish"


@dataclass
class Thought:
    """A reasoning step in the ReAct trace"""
    content: str


@dataclass
class Action:
    """An action to take"""
    action_type: ActionType
    argument: str


@dataclass
class Observation:
    """Result of an action from the environment"""
    content: str


@dataclass
class ReActStep:
    """A single step in the ReAct loop"""
    thought: Thought
    action: Action
    observation: Optional[Observation] = None


class Environment:
    """
    Simulated environment for ReAct agent.
    In practice, this would connect to real tools (search APIs, databases, etc.)
    """
    
    def __init__(self):
        # Simulated knowledge base
        self.knowledge = {
            "eiffel tower": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889. It is 330 meters tall.",
            "paris": "Paris is the capital and largest city of France. It has a population of about 2.1 million people.",
            "python": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
            "transformer": "Transformers are a type of neural network architecture introduced in 2017. They use self-attention mechanisms.",
        }
    
    def search(self, query: str) -> str:
        """Simulate a search action"""
        query_lower = query.lower()
        for key, value in self.knowledge.items():
            if key in query_lower:
                return value
        return f"No results found for: {query}"
    
    def lookup(self, term: str) -> str:
        """Simulate looking up a specific term"""
        term_lower = term.lower()
        if term_lower in self.knowledge:
            return self.knowledge[term_lower]
        return f"'{term}' not found in current context."
    
    def calculate(self, expression: str) -> str:
        """Simulate a calculation"""
        try:
            # Very limited safe eval for demo
            allowed_chars = set('0123456789+-*/() .')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return str(result)
            return "Invalid expression"
        except Exception as e:
            return f"Calculation error: {e}"
    
    def execute(self, action: Action) -> Observation:
        """Execute an action and return observation"""
        if action.action_type == ActionType.SEARCH:
            result = self.search(action.argument)
        elif action.action_type == ActionType.LOOKUP:
            result = self.lookup(action.argument)
        elif action.action_type == ActionType.CALCULATE:
            result = self.calculate(action.argument)
        elif action.action_type == ActionType.FINISH:
            result = f"Final Answer: {action.argument}"
        else:
            result = "Unknown action type"
        
        return Observation(content=result)


class ReActAgent:
    """
    ReAct Agent: Reasoning + Acting
    
    The ReAct framework interleaves:
    1. Thought: Reasoning about what to do
    2. Action: Taking an action in the environment
    3. Observation: Receiving feedback from the environment
    
    This creates a trace like:
    Thought 1 → Action 1 → Observation 1 → Thought 2 → Action 2 → ...
    """
    
    def __init__(self, environment: Environment):
        self.environment = environment
        self.trace: List[ReActStep] = []
        self.max_steps = 10
    
    def reset(self):
        """Reset the agent's trace"""
        self.trace = []
    
    def think(self, question: str, observations: List[Observation]) -> Thought:
        """
        Generate a thought based on question and observations.
        
        In practice, this would call an LLM to generate reasoning.
        Here we simulate the process.
        """
        if not observations:
            # Initial thought
            if "eiffel" in question.lower():
                return Thought("I need to search for information about the Eiffel Tower.")
            elif "calculate" in question.lower() or any(c in question for c in "+-*/"):
                return Thought("This seems like a calculation problem. Let me compute it.")
            else:
                return Thought(f"I need to search for relevant information to answer: {question}")
        else:
            last_obs = observations[-1].content
            if "found" in last_obs.lower() or len(observations) > 2:
                return Thought("I have enough information to provide the answer.")
            else:
                return Thought("I found some information. Let me analyze it further.")
    
    def decide_action(self, thought: Thought, question: str) -> Action:
        """
        Decide which action to take based on current thought.
        
        In practice, an LLM would decide this. We simulate with rules.
        """
        thought_lower = thought.content.lower()
        
        if "have enough" in thought_lower or "provide the answer" in thought_lower:
            # Extract a summary as the answer
            return Action(ActionType.FINISH, "Based on my research, here is the answer.")
        
        if "search" in thought_lower:
            # Extract search query from question
            return Action(ActionType.SEARCH, question)
        
        if "calculate" in thought_lower or "compute" in thought_lower:
            # Extract expression (simplified)
            import re
            numbers = re.findall(r'[\d+\-*/\(\)\. ]+', question)
            if numbers:
                return Action(ActionType.CALCULATE, numbers[0].strip())
        
        return Action(ActionType.SEARCH, question)
    
    def step(self, question: str) -> ReActStep:
        """Execute a single ReAct step"""
        observations = [step.observation for step in self.trace if step.observation]
        
        # Think
        thought = self.think(question, observations)
        
        # Act
        action = self.decide_action(thought, question)
        
        # Observe
        observation = self.environment.execute(action)
        
        step = ReActStep(thought=thought, action=action, observation=observation)
        self.trace.append(step)
        
        return step
    
    def run(self, question: str) -> Dict:
        """
        Run the full ReAct loop until completion.
        """
        self.reset()
        
        for i in range(self.max_steps):
            step = self.step(question)
            
            # Check if we're done
            if step.action.action_type == ActionType.FINISH:
                break
        
        return {
            'question': question,
            'trace': self.trace,
            'answer': self.trace[-1].observation.content if self.trace else None,
            'num_steps': len(self.trace)
        }
    
    def format_trace(self) -> str:
        """Format the trace for display"""
        lines = []
        for i, step in enumerate(self.trace, 1):
            lines.append(f"Thought {i}: {step.thought.content}")
            lines.append(f"Action {i}: [{step.action.action_type.value}] {step.action.argument}")
            if step.observation:
                lines.append(f"Observation {i}: {step.observation.content}")
            lines.append("")
        return "\n".join(lines)


def create_react_prompt_example() -> str:
    """Show what a ReAct prompt looks like"""
    return """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Thought 1: I need to search for information about the Colorado orogeny, specifically its eastern sector.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.

Thought 2: It doesn't mention the eastern sector. I should search for more specific information.
Action 2: Search[eastern sector Colorado orogeny]
Observation 2: The eastern sector extends into the High Plains.

Thought 3: Now I need to find the elevation range of the High Plains.
Action 3: Search[High Plains elevation range]
Observation 3: The High Plains rise in elevation from around 1,800 to 7,000 feet.

Thought 4: I have found the elevation range.
Action 4: Finish[1,800 to 7,000 feet]"""


def demo_react_agent():
    """Demonstrate the ReAct agent"""
    print("=" * 70)
    print("ReAct: Reasoning and Acting Demo")
    print("=" * 70)
    
    env = Environment()
    agent = ReActAgent(env)
    
    questions = [
        "What is the Eiffel Tower and how tall is it?",
        "Calculate 125 * 8",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        result = agent.run(question)
        print(agent.format_trace())
        print(f"✅ Final Answer: {result['answer']}")
        print(f"   Steps taken: {result['num_steps']}")
        print()


def explain_react():
    """Explain the ReAct framework"""
    print("\n" + "=" * 70)
    print("📚 Understanding ReAct")
    print("=" * 70)
    
    print("""
ReAct = Reasoning + Acting

The key insight: Interleave reasoning (thinking) with acting (tool use).

Traditional approaches:
    1. Chain-of-Thought: Reasoning only, no external actions
    2. Action-only: Execute tools without explicit reasoning
    
ReAct combines both:
    Thought → Action → Observation → Thought → Action → ...

The Trace Format:
    Thought 1: [Internal reasoning about what to do]
    Action 1: [search/lookup/calculate/finish][argument]
    Observation 1: [Result from environment]
    
    Thought 2: [Reason about the observation]
    Action 2: [Next action based on reasoning]
    ...

Why it works:
    1. Reasoning grounds action decisions
    2. Observations inform future reasoning
    3. Synergy between thinking and acting
    4. More interpretable than action-only
    5. More grounded than reasoning-only
""")


def show_react_vs_cot():
    """Compare ReAct with Chain-of-Thought"""
    print("\n" + "=" * 70)
    print("📊 ReAct vs Chain-of-Thought")
    print("=" * 70)
    
    print("""
Chain-of-Thought (CoT):
    - Pure reasoning, no external tools
    - Good for self-contained problems
    - Limited by model's knowledge
    - Can hallucinate facts
    
    Example:
    Q: What's the capital of France?
    A: Let me think... France is a country in Europe. 
       Its capital is Paris.

ReAct:
    - Reasoning + external tool use
    - Can access real-time information
    - Grounded in actual data
    - More reliable for factual questions
    
    Example:
    Q: What's the capital of France?
    Thought 1: I should search for this information.
    Action 1: Search[capital of France]
    Observation 1: Paris is the capital of France...
    Thought 2: I found the answer.
    Action 2: Finish[Paris]

When to use which:
    - CoT: Math problems, logic puzzles, self-contained reasoning
    - ReAct: Factual questions, multi-step research, tool use
""")


def demo():
    """Main demonstration"""
    print("\n📝 ReAct Prompt Format Example:")
    print("-" * 40)
    print(create_react_prompt_example())
    
    demo_react_agent()
    explain_react()
    show_react_vs_cot()
    
    print("\n" + "=" * 70)
    print("📚 Why ReAct Matters")
    print("=" * 70)
    print("""
    1. Foundation of Agentic AI: Most agents today use ReAct-style loops
    
    2. Tool Use: Pioneered structured tool use in LLMs
    
    3. Interpretability: Thought traces show reasoning process
    
    4. Grounded Responses: Actions connect to real data
    
    5. Influenced: AutoGPT, LangChain, OpenAI Assistants, and more
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
