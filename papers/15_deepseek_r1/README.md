# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## Paper Summary

DeepSeek-R1 proved that large-scale reinforcement learning without supervised data can induce self-verification and structured reasoning behavior.

**Paper Link:** https://arxiv.org/abs/2501.12948

## The Key Insight

Using only correctness as a reward signal, the model learns to:
- Verify its own work
- Reflect and backtrack on errors
- Decompose complex problems
- Generate long, structured reasoning

No step-by-step supervision needed!

## Training Process

```
Cold Start → Reasoning RL → Rejection Sampling → Final RL
   (SFT)      (GRPO)          (Distillation)    (Alignment)
```

### Phase 1: Cold Start
- Small amount of long CoT examples
- Teaches format, not reasoning

### Phase 2: Reasoning RL (GRPO)
- Pure RL on math/reasoning tasks
- Reward = correctness only
- This is where magic happens

### Phase 3: Rejection Sampling
- Generate many solutions per problem
- Keep only correct ones
- Distill back into model

### Phase 4: Final RL
- Add helpfulness and safety

## Emergent Behaviors

1. **Self-Verification**: "Let me check: 5 × 7 = 35. ✓"
2. **Reflection**: "Wait, that doesn't work. Let me try..."
3. **Aha Moments**: Sudden capability jumps during training
4. **Extended Thinking**: Thousands of tokens when needed

## Implementation

```bash
python deepseek_r1.py
```

This implementation demonstrates:
- GRPO algorithm simulation
- R1 reasoning format with `<think>` tags
- Emergent behavior descriptions
- Training dynamics visualization

## The R1 Format

```
<think>
[Extended reasoning here]
Step 1: ...
Step 2: ...
Let me verify...
</think>

[Final answer here]
```

## Why This Paper Matters

1. **Opened the Black Box**: Unlike OpenAI's o1, R1's methods are documented
2. **Pure RL Works**: Proves complex reasoning can emerge from simple reward
3. **Efficient Distillation**: R1-7B retains much of R1's capability
4. **Research Direction**: Shows path to emergent reasoning
5. **Open Weights**: Community can study and build on it
