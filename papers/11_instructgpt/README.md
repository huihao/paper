# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

## Paper Summary

InstructGPT introduced the modern RLHF pipeline that became the blueprint for aligning language models with human preferences.

**Paper Link:** https://arxiv.org/abs/2203.02155

## The RLHF Pipeline

```
Pretrained LLM → SFT Model → RL Policy
                    ↓              ↑
              Reward Model ────────┘
```

### Step 1: Supervised Fine-Tuning (SFT)
- Collect demonstrations of desired behavior
- Human labelers write high-quality responses
- Fine-tune pretrained model on these examples

### Step 2: Reward Model Training
- Collect comparison data: "Which response is better?"
- Train a model to predict human preferences
- Uses Bradley-Terry model: P(A > B) = σ(r_A - r_B)

### Step 3: Policy Optimization (PPO)
- Generate responses from current policy
- Score with reward model
- Update policy using PPO with KL penalty

## Key Equations

### Reward Model Loss
```
L_RM = -log(σ(r_chosen - r_rejected))
```

### PPO Objective with KL Penalty
```
maximize E[r_θ(s,a) * A(s,a)] - β * KL(π_θ || π_SFT)
```

## Implementation

```bash
python instructgpt.py
```

This implementation demonstrates:
- SFT training simulation
- Reward model with Bradley-Terry loss
- PPO training with KL penalty
- Complete RLHF pipeline

## Why RLHF Works

1. **SFT teaches format**: Learn to respond helpfully
2. **RM captures preferences**: What makes a response "good"
3. **PPO optimizes**: Generate responses that score highly
4. **KL prevents collapse**: Stay close to SFT, avoid reward hacking

## Data Collection

InstructGPT used:
- 13k demonstrations for SFT
- 33k comparisons for RM
- Human labelers spent significant time on quality

## Impact

InstructGPT created the modern alignment paradigm:
- Led directly to ChatGPT
- All major labs adopted RLHF
- Showed alignment is tractable
- Human feedback is key to helpful models

## Limitations Addressed

| Without RLHF | With RLHF |
|--------------|-----------|
| Follows pretraining distribution | Follows human preferences |
| May produce harmful content | Actively avoids harm |
| Verbose, meandering | Concise, helpful |
| Ignores instructions | Follows instructions |

## Why This Paper Matters

InstructGPT wasn't the first RLHF paper, but it:
1. Showed RLHF works at scale (175B parameters)
2. Demonstrated clear improvements in helpfulness
3. Created the ChatGPT lineage
4. Made RLHF the industry standard
