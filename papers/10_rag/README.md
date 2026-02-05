# Retrieval-Augmented Generation (RAG) (Lewis et al., 2020)

## Paper Summary

RAG combines parametric models (LLMs) with non-parametric knowledge (retrieved documents), enabling more factual and up-to-date responses.

**Paper Link:** https://arxiv.org/abs/2005.11401

## How RAG Works

```
Question → Retriever → Relevant Documents → LLM → Answer
                ↓                             ↓
          Vector Store                   Augmented Prompt
```

### The RAG Formula

```
P(y|x) = Σ_z P(z|x) P(y|x,z)
```

Where:
- x = input query
- z = retrieved document
- y = generated output
- P(z|x) = retrieval probability
- P(y|x,z) = generation probability given document

## Key Components

### 1. Retriever
- Encodes query and documents into embeddings
- Finds most similar documents via vector similarity
- Common: DPR (Dense Passage Retrieval), Contriever

### 2. Document Store
- Vector database with document embeddings
- Fast approximate nearest neighbor search
- Common: FAISS, Pinecone, Weaviate, Chroma

### 3. Generator
- LLM that generates answers given retrieved context
- Prompt includes query + retrieved documents
- Common: GPT-4, Claude, LLaMA

## Implementation

```bash
python rag.py
```

This implementation demonstrates:
- Document embedding and storage
- Semantic retrieval
- Context-augmented generation
- RAG-Sequence and RAG-Token variants

## RAG Variants

### RAG-Sequence
- Retrieve once for entire response
- Simpler and more efficient
- Most common in practice

### RAG-Token
- Can use different docs per token
- More flexible but expensive
- Better for complex reasoning

## Benefits of RAG

| Challenge | Without RAG | With RAG |
|-----------|-------------|----------|
| Hallucination | Common | Reduced (grounded in sources) |
| Outdated info | Retrain model | Update documents |
| Domain knowledge | Fine-tune | Add domain docs |
| Transparency | Black box | Show sources |

## Modern RAG Techniques

1. **HyDE**: Generate hypothetical answer, then retrieve
2. **Self-RAG**: Model decides when to retrieve
3. **CRAG**: Correct and refine retrieved passages
4. **Multi-hop RAG**: Iterative retrieval for complex queries

## Why This Paper Matters

RAG became foundational for enterprise AI:
- ChatGPT with browsing uses RAG
- Enterprise chatbots rely on RAG
- Almost all knowledge-grounded systems use RAG principles
- Bridges the gap between parametric and non-parametric knowledge
