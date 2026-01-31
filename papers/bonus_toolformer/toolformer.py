#!/usr/bin/env python3
"""
Toolformer: Language Models Can Teach Themselves to Use Tools
(Schick et al., 2023)

Teaching LLMs to use external tools through self-supervised learning.

Paper: https://arxiv.org/abs/2302.04761
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A call to an external tool"""
    tool_name: str
    arguments: str
    result: str


class Toolformer:
    """
    Toolformer: Self-Supervised Tool Learning
    
    Key insight: LLMs can learn when and how to use tools
    by generating their own training data.
    
    Process:
    1. Sample potential tool calls from LLM
    2. Execute tools to get results
    3. Filter: Keep calls that reduce perplexity
    4. Fine-tune on successful tool uses
    """
    
    def __init__(self):
        self.available_tools = {
            'Calculator': self.calculator,
            'Search': self.search,
            'Calendar': self.calendar,
            'QA': self.qa_system,
            'Translator': self.translator,
        }
    
    @staticmethod
    def describe_approach():
        return """
Toolformer Approach:

Traditional Tool Use:
    - Hand-craft prompts for each tool
    - Manual examples of tool usage
    - Separate systems for different tools

Toolformer (Self-Supervised):
    1. Start with unannotated text
    2. Have LLM sample potential tool calls
    3. Execute tools, get results
    4. Filter: Keep if result helps prediction
    5. Fine-tune on successful examples

Example Transformation:
    Original: "The population of Toronto is 2,930,000"
    
    With tool: "The population of Toronto is 
               [Calculator(2,930,000 * 1.2)] = 3,516,000"
    
    Tool call added where it helps!

Why It Works:
    - LLM knows WHAT it doesn't know
    - Tools fill knowledge gaps
    - Self-filtering ensures quality
"""

    @staticmethod
    def describe_tools():
        return """
Available Tools in Toolformer:

1. Calculator:
   Input: Mathematical expression
   Output: Computed result
   Example: [Calculator(123 * 456)] → 56088

2. Q&A System:
   Input: Factual question
   Output: Answer from knowledge base
   Example: [QA(Who invented the telephone?)] → Alexander Graham Bell

3. Wikipedia Search:
   Input: Search query
   Output: Relevant snippet
   Example: [Search(Albert Einstein)] → Albert Einstein was a physicist...

4. Machine Translator:
   Input: text, source_lang, target_lang
   Output: Translated text
   Example: [MT(hello, en, de)] → hallo

5. Calendar:
   Input: Date query
   Output: Date/time information
   Example: [Calendar(today)] → 2025-01-31
"""

    def calculator(self, expression: str) -> str:
        """Simple calculator tool using safe evaluation"""
        import ast
        import operator
        
        # Safe operators for basic math
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](safe_eval(node.left), safe_eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](safe_eval(node.operand))
            else:
                raise ValueError(f"Unsupported operation")
        
        try:
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree.body)
            return str(result)
        except Exception:
            return "Error"
    
    def search(self, query: str) -> str:
        """Mock search tool"""
        # In practice, this would call a search API
        return f"[Search result for: {query}]"
    
    def calendar(self, query: str) -> str:
        """Mock calendar tool"""
        from datetime import datetime
        if query == "today":
            return datetime.now().strftime("%Y-%m-%d")
        return "Unknown date query"
    
    def qa_system(self, question: str) -> str:
        """Mock QA tool"""
        return f"[Answer to: {question}]"
    
    def translator(self, text: str) -> str:
        """Mock translator tool"""
        return f"[Translation of: {text}]"
    
    def parse_tool_call(self, text: str) -> Optional[ToolCall]:
        """Parse a tool call from text"""
        # Format: [ToolName(arguments)]
        import re
        match = re.search(r'\[(\w+)\((.*?)\)\]', text)
        if match:
            return ToolCall(
                tool_name=match.group(1),
                arguments=match.group(2),
                result=""
            )
        return None
    
    def execute_tool_call(self, call: ToolCall) -> str:
        """Execute a parsed tool call"""
        if call.tool_name in self.available_tools:
            return self.available_tools[call.tool_name](call.arguments)
        return f"Unknown tool: {call.tool_name}"


def demonstrate_self_supervision():
    """Demonstrate the self-supervised learning process"""
    print("\n📊 Self-Supervised Tool Learning")
    print("=" * 70)
    
    print("""
Step 1: Sample Tool Calls
    Given: "The Colosseum was built around 70-80 AD, making it about X years old"
    
    LLM samples possible tool insertions:
    a) "...built around [Calculator(2024-75)] = 1949 years old"
    b) "...built around [Search(Colosseum age)] years old"
    c) "...making it about [QA(How old is Colosseum)] years old"

Step 2: Execute Tools
    a) Calculator(2024-75) → 1949
    b) Search(Colosseum age) → "nearly 2000 years old"
    c) QA(How old is Colosseum) → "approximately 1950 years"

Step 3: Filter by Perplexity Reduction
    Compare perplexity of:
    - Original text (no tool)
    - Text with tool call and result
    
    Keep tool call if it REDUCES perplexity!
    
    a) Calculator: Keeps (gives exact number)
    b) Search: Maybe keeps (relevant info)
    c) QA: Keeps (accurate answer)

Step 4: Fine-tune on Successful Examples
    Model learns:
    - When to use tools
    - Which tool to use
    - How to format calls
    - How to integrate results
""")


def demo():
    """Main demonstration"""
    print("=" * 70)
    print("Toolformer: Language Models Can Teach Themselves to Use Tools")
    print("(Schick et al., 2023)")
    print("=" * 70)
    
    toolformer = Toolformer()
    
    print(toolformer.describe_approach())
    print(toolformer.describe_tools())
    
    demonstrate_self_supervision()
    
    # Demo tool execution
    print("\n📊 Tool Execution Demo")
    print("-" * 40)
    
    tool_calls = [
        ("[Calculator(123 * 456)]", "Calculator", "123 * 456"),
        ("[Calendar(today)]", "Calendar", "today"),
    ]
    
    for text, tool_name, args in tool_calls:
        call = ToolCall(tool_name, args, "")
        result = toolformer.execute_tool_call(call)
        print(f"  {text} → {result}")
    
    print("\n" + "=" * 70)
    print("📚 Why This Paper Matters")
    print("=" * 70)
    print("""
    1. Self-Supervised: No manual annotation needed
    
    2. Multi-Tool: One model, many tools
    
    3. Selective: Model learns WHEN to use tools
    
    4. Foundation: Influenced later agent systems
    
    5. Practical: Improves factuality and computation
    
    Key Insight:
        LLMs can teach themselves to use tools
        by filtering for calls that help prediction.
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo()
