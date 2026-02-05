# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)

## Paper Summary

Chain-of-Thought (CoT) prompting demonstrated that reasoning can be elicited through prompting alone, laying the groundwork for later reasoning-focused training.

**Paper Link:** https://arxiv.org/abs/2201.11903

## The Key Insight

Including step-by-step reasoning in few-shot examples causes the model to generate its own reasoning:

### Standard Prompt
```
Q: Roger has 5 balls. He buys 2 cans of 3 balls each. How many balls?
A: 11

Q: [New question]
A:
```

### Chain-of-Thought Prompt
```
Q: Roger has 5 balls. He buys 2 cans of 3 balls each. How many balls?
A: Roger started with 5 balls. He bought 2 × 3 = 6 balls. Total = 5 + 6 = 11.

Q: [New question]
A: Let's think step by step.
```

## Implementation

```bash
python chain_of_thought.py
```

This implementation demonstrates:
- Standard vs CoT prompt construction
- Zero-shot CoT ("Let's think step by step")
- Self-consistency (sample and vote)
- Different reasoning domains

## Key Findings

### 1. Scale Matters
- Little benefit at small scale (<10B)
- Emergent at ~100B parameters
- Works best with frontier models

### 2. Task Complexity
- More benefit on complex, multi-step problems
- Best for: math, logic, multi-hop reasoning
- Less benefit for: simple retrieval

### 3. Prompt Design
- Clear, logical reasoning steps
- 4-8 examples typically sufficient
- Quality > quantity

## Zero-Shot CoT

Simply adding "Let's think step by step" works!

```
Q: If a train travels 60 mph for 2.5 hours, how far does it go?
A: Let's think step by step.
```

This alone can improve reasoning without any examples.

## Self-Consistency

Sample multiple reasoning paths and vote:

1. Generate multiple CoT responses (temperature > 0)
2. Extract answer from each
3. Return majority vote

This improves accuracy by ~5-10% on many benchmarks.

## Related Techniques

| Technique | Description |
|-----------|-------------|
| Zero-shot CoT | "Let's think step by step" |
| Self-consistency | Sample and vote |
| Least-to-most | Break into subproblems |
| Tree-of-thought | Explore multiple branches |
| ReAct | Reason + Act |

## Why This Paper Matters

CoT was a watershed moment:
1. Showed LLMs can reason with proper prompting
2. Made reasoning interpretable (can see the chain)
3. Opened entire field of reasoning in LLMs
4. Led to o1, DeepSeek-R1, and reasoning-focused models
5. Made "step by step" a standard prompting technique
