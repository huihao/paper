# ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)

## Paper Summary

ReAct combines reasoning traces with tool use and environment interaction, forming the foundation of modern agentic AI systems.

**Paper Link:** https://arxiv.org/abs/2210.03629

## The ReAct Framework

```
Thought → Action → Observation → Thought → Action → ...
```

Interleaving reasoning (Thought) with acting (Action) and receiving feedback (Observation).

## Example Trace

```
Question: What is the elevation range of the High Plains?

Thought 1: I need to search for information about the High Plains.
Action 1: Search[High Plains]
Observation 1: The High Plains are a subregion of the Great Plains...

Thought 2: I found general info. Let me search specifically for elevation.
Action 2: Search[High Plains elevation]
Observation 2: The High Plains rise from around 1,800 to 7,000 feet.

Thought 3: I have the answer.
Action 3: Finish[1,800 to 7,000 feet]
```

## Implementation

```bash
python react.py
```

This implementation demonstrates:
- ReAct agent loop
- Tool execution (search, lookup, calculate)
- Trace formatting
- Comparison with Chain-of-Thought

## Available Actions

| Action | Description |
|--------|-------------|
| Search[query] | Search for information |
| Lookup[term] | Look up a specific term |
| Calculate[expr] | Perform a calculation |
| Finish[answer] | Return final answer |

## ReAct vs Chain-of-Thought

| Aspect | CoT | ReAct |
|--------|-----|-------|
| External tools | No | Yes |
| Real-time data | No | Yes |
| Grounded facts | Limited | Yes |
| Use case | Self-contained problems | Research, tool use |

## Key Insights

1. **Synergy**: Reasoning makes actions more targeted; observations improve reasoning
2. **Interpretability**: Can see why agent took each action
3. **Grounding**: Answers based on actual data, not just model knowledge
4. **Error Recovery**: Can adjust based on observations

## Influence on Modern AI

ReAct pioneered patterns now standard in:
- LangChain agents
- OpenAI Assistants
- AutoGPT and similar projects
- Any LLM-based agent system

## Why This Paper Matters

ReAct is the foundation of agentic AI:
1. Defined the thought-action-observation loop
2. Showed LLMs can effectively use tools
3. Created interpretable agent traces
4. Influenced all subsequent agent frameworks
