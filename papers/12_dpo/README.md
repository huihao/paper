# Direct Preference Optimization (DPO) (Rafailov et al., 2023)

## Paper Summary

DPO provides a simpler and more stable alternative to PPO-based RLHF by directly optimizing the policy using preference data.

**Paper Link:** https://arxiv.org/abs/2305.18290

## The Key Insight

The optimal policy under the RLHF objective has a closed form:

```
π*(y|x) ∝ π_ref(y|x) * exp(r(x,y) / β)
```

This can be rearranged to express reward as:

```
r(x,y) = β * log(π*(y|x) / π_ref(y|x)) + C
```

So we can optimize the policy directly without learning a reward model!

## The DPO Loss

```
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

Where:
- y_w = winning (chosen) response
- y_l = losing (rejected) response
- π = policy being trained
- π_ref = reference policy (frozen SFT model)
- β = temperature parameter

## DPO vs PPO

| Aspect | PPO | DPO |
|--------|-----|-----|
| Reward Model | Required | Not needed |
| Sampling | Required (online) | Not needed (offline) |
| Complexity | High | Low |
| Stability | Sensitive to hyperparams | More stable |
| Implementation | Complex | Simple |

## Implementation

```bash
python dpo.py
```

This implementation demonstrates:
- DPO loss computation
- Training loop on preference data
- Comparison with PPO approach
- Beta (β) parameter effects

## How It Works

1. **Reference Policy**: Freeze a copy of the SFT model
2. **Policy Update**: For each preference pair:
   - Compute log-ratios for chosen and rejected
   - Apply DPO loss to increase chosen and decrease rejected
3. **β Control**: Higher β = stay closer to reference

## Why DPO Works

The implicit reward is:
```
r(x,y) = β * log(π(y|x) / π_ref(y|x))
```

By optimizing the DPO loss, we're effectively:
- Increasing log-ratio for preferred responses
- Decreasing log-ratio for rejected responses
- This matches what a reward model would learn!

## Variants

- **IPO**: Fixes theoretical issues with DPO
- **KTO**: Works with unpaired preferences (just good/bad labels)
- **ORPO**: Combines SFT and preference learning
- **SimPO**: Simplified version without reference model

## Why This Paper Matters

DPO democratized preference learning:
- Simpler to implement than PPO
- More stable training
- Widely adopted by open-source community
- Powers many open instruction-tuned models
