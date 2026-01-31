# Qwen3 Technical Report (Yang et al., 2025)

## Paper Summary

Qwen3 introduces a modern architecture with unified MoE and the ability to dynamically trade off cost and reasoning depth through Thinking and Non-Thinking modes.

**Paper Link:** https://arxiv.org/abs/2505.09388

## Key Innovation: Thinking Modes

The same model can operate in two modes:

### Thinking Mode
```
<think>
Let me solve this step by step...
First, I need to...
Then, I can...
</think>

The answer is X.
```
- Extended reasoning chain
- Higher quality for complex tasks
- More tokens, higher latency

### Non-Thinking Mode
```
The answer is X.
```
- Direct response
- Fast and efficient
- Fewer tokens, lower latency

## Implementation

```bash
python qwen3.py
```

This implementation demonstrates:
- Thinking mode controller
- MoE layer with fine-grained routing
- Grouped Query Attention (GQA)
- Cost estimation for modes

## Model Family

| Model | Parameters | Experts | Active |
|-------|-----------|---------|--------|
| Qwen3-0.6B | 0.6B | Dense | - |
| Qwen3-1.7B | 1.7B | Dense | - |
| Qwen3-4B | 4B | Dense | - |
| Qwen3-8B | 8B | Dense | - |
| Qwen3-14B | 14B | Dense | - |
| Qwen3-32B | 32B | Dense | - |
| Qwen3-30B-A3B | 30B | 128 | 8 |
| Qwen3-235B-A22B | 235B | 128 | 8 |

## Architectural Features

### 1. Grouped Query Attention (GQA)
- Multiple Q heads share K/V heads
- Reduces memory for KV cache
- Faster inference

### 2. Fine-Grained MoE
- 64-128 small experts
- Top-k routing (k=8)
- Better specialization

### 3. Extended Context
- RoPE with θ=1,000,000
- Base 32k, extended 128k with YaRN

## When to Use Each Mode

| Scenario | Mode | Reason |
|----------|------|--------|
| Math problem | Thinking | Needs reasoning |
| Simple Q&A | Non-thinking | Speed matters |
| Code generation | Thinking | Complex logic |
| Chitchat | Non-thinking | Fast response |
| Analysis | Thinking | Quality matters |

## Why This Paper Matters

1. **Practical Design**: Built for real deployment constraints
2. **Dynamic Trade-off**: Choose cost vs quality at runtime
3. **Unified Family**: Dense and MoE from same approach
4. **Open Weights**: Available for community use
5. **Modern Architecture**: All the latest improvements
