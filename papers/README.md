# LLM Foundations: Paper Implementations

A comprehensive collection of foundational LLM papers with Python implementations. Each paper is implemented in its own folder with code demonstrations and documentation.

## 📚 Recommended Reading Order

### Core Transformer Foundations

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 1 | [Attention Is All You Need](papers/01_attention_is_all_you_need) | 2017 | `01_attention_is_all_you_need` | Self-attention, multi-head attention, encoder-decoder |
| 2 | [The Illustrated Transformer](papers/02_illustrated_transformer) | 2018 | `02_illustrated_transformer` | Visual intuition for attention and tensor flow |
| 3 | [BERT](papers/03_bert) | 2018 | `03_bert` | Masked language modeling, bidirectional context |
| 4 | [GPT-3](papers/04_gpt3) | 2020 | `04_gpt3` | In-context learning, few-shot prompting |

### Scaling Laws

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 5 | [Scaling Laws](papers/05_scaling_laws) | 2020 | `05_scaling_laws` | Parameter/data/compute scaling |
| 6 | [Chinchilla](papers/06_chinchilla) | 2022 | `06_chinchilla` | Compute-optimal training |

### Modern Architecture

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 7 | [LLaMA](papers/07_llama) | 2023 | `07_llama` | RMSNorm, SwiGLU, RoPE |
| 8 | [RoFormer](papers/08_roformer) | 2021 | `08_roformer` | Rotary Position Embedding |
| 9 | [FlashAttention](papers/09_flash_attention) | 2022 | `09_flash_attention` | Memory-efficient attention |

### Retrieval and Alignment

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 10 | [RAG](papers/10_rag) | 2020 | `10_rag` | Retrieval-augmented generation |
| 11 | [InstructGPT](papers/11_instructgpt) | 2022 | `11_instructgpt` | RLHF, instruction tuning |
| 12 | [DPO](papers/12_dpo) | 2023 | `12_dpo` | Direct preference optimization |

### Reasoning and Agents

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 13 | [Chain-of-Thought](papers/13_chain_of_thought) | 2022 | `13_chain_of_thought` | Reasoning through prompting |
| 14 | [ReAct](papers/14_react) | 2022 | `14_react` | Reasoning + acting framework |
| 15 | [DeepSeek-R1](papers/15_deepseek_r1) | 2025 | `15_deepseek_r1` | RL for reasoning |
| 16 | [Qwen3](papers/16_qwen3) | 2025 | `16_qwen3` | Unified MoE with thinking modes |

### Mixture of Experts

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 17 | [Sparse MoE](papers/17_sparse_moe) | 2017 | `17_sparse_moe` | Conditional computation |
| 18 | [Switch Transformers](papers/18_switch_transformer) | 2021 | `18_switch_transformer` | Single-expert routing |
| 19 | [Mixtral](papers/19_mixtral) | 2024 | `19_mixtral` | Open-weight MoE |
| 20 | [Sparse Upcycling](papers/20_sparse_upcycling) | 2022 | `20_sparse_upcycling` | Dense to MoE conversion |

### Understanding and Interpretability

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 21 | [Platonic Representation](papers/21_platonic_representation) | 2024 | `21_platonic_representation` | Shared representations across modalities |
| 22 | [Textbooks Are All You Need](papers/22_textbooks_are_all_you_need) | 2023 | `22_textbooks_are_all_you_need` | Synthetic data quality |
| 23 | [Scaling Monosemanticity](papers/23_scaling_monosemanticity) | 2024 | `23_scaling_monosemanticity` | Interpretable features |

### Large-Scale Training

| # | Paper | Year | Folder | Key Concepts |
|---|-------|------|--------|--------------|
| 24 | [PaLM](papers/24_palm) | 2022 | `24_palm` | Pathways system |
| 25 | [GLaM](papers/25_glam) | 2022 | `25_glam` | MoE economics |
| 26 | [Smol Training Playbook](papers/26_smol_training_playbook) | 2025 | `26_smol_training_playbook` | Practical training guide |

### Bonus Papers

| Paper | Folder | Key Concepts |
|-------|--------|--------------|
| [T5](papers/bonus_t5) | `bonus_t5` | Text-to-text framework |
| [Toolformer](papers/bonus_toolformer) | `bonus_toolformer` | Self-supervised tool learning |
| [GShard](papers/bonus_gshard) | `bonus_gshard` | Automatic sharding |
| [Adaptive MoE](papers/bonus_adaptive_moe) | `bonus_adaptive_moe` | Original MoE (1991) |
| [Hierarchical MoE](papers/bonus_hierarchical_moe) | `bonus_hierarchical_moe` | Tree-structured gating |

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy torch transformers
```

### Running an Implementation

```bash
cd papers/01_attention_is_all_you_need
python transformer.py
```

Each folder contains:
- **README.md**: Paper summary and key concepts
- **Python implementation**: Runnable code demonstrating the paper's concepts

## 📖 Learning Path

1. **Start with fundamentals**: Papers 1-4 (Transformer, BERT, GPT-3)
2. **Understand scaling**: Papers 5-6 (Scaling Laws, Chinchilla)
3. **Modern architecture**: Papers 7-9 (LLaMA, RoPE, FlashAttention)
4. **Alignment and tuning**: Papers 10-12 (RAG, RLHF, DPO)
5. **Reasoning and agents**: Papers 13-16 (CoT, ReAct, R1)
6. **MoE deep dive**: Papers 17-20 (MoE family)
7. **Advanced topics**: Papers 21-26 (Interpretability, training)

## 🎯 Key Takeaways

If you deeply understand these fundamentals:
- **Transformer core**: Self-attention, multi-head attention
- **Scaling laws**: Parameter, data, and compute relationships
- **FlashAttention**: Memory-efficient attention
- **Instruction tuning**: RLHF and DPO
- **R1-style reasoning**: RL for reasoning
- **MoE upcycling**: Dense to sparse conversion

...you understand LLMs better than most!

## 📝 License

Educational implementations for learning purposes.

---

*Time to lock-in, good luck! 🚀*
