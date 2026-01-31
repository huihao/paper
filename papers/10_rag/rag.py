#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) (Lewis et al., 2020)
Implementation of the RAG framework combining retrieval with generation.

Paper: https://arxiv.org/abs/2005.11401
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


@dataclass
class Document:
    """A document in the knowledge base"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class SimpleEmbedder:
    """
    Simple embedder for demonstration.
    In practice, use models like sentence-transformers or OpenAI embeddings.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.cache = {}
    
    def embed(self, text: str) -> np.ndarray:
        """Create a deterministic embedding from text"""
        if text in self.cache:
            return self.cache[text]
        
        # Create deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = np.frombuffer(hash_bytes[:self.dim], dtype=np.uint8).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Add some semantic structure (very simplified)
        words = text.lower().split()
        for i, word in enumerate(words[:10]):
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            idx = word_hash % self.dim
            embedding[idx] += 0.1 * (1.0 / (i + 1))
        
        embedding = embedding / np.linalg.norm(embedding)
        self.cache[text] = embedding
        
        return embedding


class VectorStore:
    """
    Simple vector store for document retrieval.
    In practice, use FAISS, Pinecone, Weaviate, etc.
    """
    
    def __init__(self, embedder: SimpleEmbedder):
        self.embedder = embedder
        self.documents: List[Document] = []
    
    def add_document(self, content: str, doc_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add a document to the store"""
        doc_id = doc_id or f"doc_{len(self.documents)}"
        embedding = self.embedder.embed(content)
        doc = Document(id=doc_id, content=content, embedding=embedding, metadata=metadata)
        self.documents.append(doc)
        return doc
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Retrieve top-k most similar documents"""
        query_embedding = self.embedder.embed(query)
        
        similarities = []
        for doc in self.documents:
            sim = cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class RAGGenerator:
    """
    Simple generator that combines retrieved context with query.
    In practice, this would be an LLM like GPT-4 or LLaMA.
    """
    
    def __init__(self):
        self.prompt_template = """Answer the question based on the context provided.

Context:
{context}

Question: {question}

Answer:"""
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate answer given question and context.
        This is a simplified simulation - real RAG uses an LLM.
        """
        # Simulate generation by extracting relevant sentences
        prompt = self.prompt_template.format(context=context, question=question)
        
        # In reality, this would call an LLM
        # Here we just return a simulated response
        return f"[Generated based on {len(context.split())} words of context]"
    
    def create_prompt(self, question: str, documents: List[Document]) -> str:
        """Create prompt with retrieved documents"""
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.content}"
            for i, doc in enumerate(documents)
        ])
        
        return self.prompt_template.format(context=context, question=question)


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    The RAG approach:
    1. Retrieve relevant documents based on query
    2. Augment the prompt with retrieved context
    3. Generate answer using the augmented prompt
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedder = SimpleEmbedder(dim=embedding_dim)
        self.vector_store = VectorStore(self.embedder)
        self.generator = RAGGenerator()
    
    def add_knowledge(self, documents: List[str]):
        """Add documents to the knowledge base"""
        for doc in documents:
            self.vector_store.add_document(doc)
    
    def query(
        self, 
        question: str, 
        top_k: int = 3,
        show_sources: bool = True
    ) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            show_sources: Whether to include source documents in output
            
        Returns:
            Dictionary with answer and optionally sources
        """
        # Step 1: Retrieve relevant documents
        retrieved = self.vector_store.retrieve(question, top_k=top_k)
        documents = [doc for doc, _ in retrieved]
        
        # Step 2: Create augmented prompt
        prompt = self.generator.create_prompt(question, documents)
        
        # Step 3: Generate answer
        context = "\n".join([doc.content for doc in documents])
        answer = self.generator.generate(question, context)
        
        result = {
            'question': question,
            'answer': answer,
            'prompt': prompt
        }
        
        if show_sources:
            result['sources'] = [
                {'content': doc.content, 'score': score}
                for doc, score in retrieved
            ]
        
        return result


class RAGSequence:
    """
    RAG-Sequence model variant.
    Uses the same retrieved documents for generating the entire sequence.
    
    P(y|x) = Σ_z P(z|x) P(y|x,z)
    where z is a retrieved document
    """
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
    
    def generate(self, question: str, num_docs: int = 3) -> Dict:
        """Generate using RAG-Sequence approach"""
        # Retrieve once, generate with all docs
        return self.pipeline.query(question, top_k=num_docs)


class RAGToken:
    """
    RAG-Token model variant.
    Can use different documents for generating different tokens.
    
    P(y_i|x, y_{<i}) = Σ_z P(z|x) P(y_i|x, y_{<i}, z)
    
    More flexible but more expensive.
    """
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
    
    def generate(self, question: str, num_docs: int = 3) -> Dict:
        """
        Generate using RAG-Token approach.
        
        In the real implementation, this would re-retrieve
        for each token generation step.
        """
        # Simplified: retrieve multiple times with different aspects
        result = self.pipeline.query(question, top_k=num_docs)
        result['variant'] = 'RAG-Token'
        result['note'] = 'Could use different docs per token'
        return result


def demo_rag_pipeline():
    """Demonstrate the RAG pipeline"""
    print("=" * 80)
    print("Retrieval-Augmented Generation (RAG) Demo")
    print("=" * 80)
    
    # Create pipeline
    rag = RAGPipeline()
    
    # Add knowledge base
    knowledge_base = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
        
        "The Great Wall of China is a series of fortifications made of stone, brick, "
        "and other materials. It was built to protect Chinese states against raids.",
        
        "The Colosseum is an oval amphitheatre in Rome, Italy. It was built between "
        "72 AD and 80 AD and could hold 50,000 to 80,000 spectators.",
        
        "Mount Everest is Earth's highest mountain above sea level. It is located in "
        "the Mahalangur Himal sub-range of the Himalayas, at 8,848.86 meters.",
        
        "The Amazon River is the largest river by discharge volume of water in the world. "
        "It flows through South America, primarily through Brazil.",
        
        "Python is a high-level programming language created by Guido van Rossum. "
        "It emphasizes code readability and supports multiple programming paradigms.",
        
        "Transformers are a type of neural network architecture introduced in 2017. "
        "They use self-attention mechanisms and have revolutionized NLP.",
        
        "Large Language Models (LLMs) are AI models trained on vast amounts of text data. "
        "Examples include GPT-4, Claude, and LLaMA.",
    ]
    
    print("\n📚 Adding documents to knowledge base...")
    rag.add_knowledge(knowledge_base)
    print(f"   Added {len(knowledge_base)} documents")
    
    # Query examples
    queries = [
        "Where is the Eiffel Tower located?",
        "What is the height of Mount Everest?",
        "What are transformers in machine learning?",
    ]
    
    print("\n🔍 Running queries...")
    print("-" * 40)
    
    for query in queries:
        print(f"\n❓ Question: {query}")
        result = rag.query(query, top_k=2)
        
        print(f"\n📄 Retrieved Sources:")
        for i, source in enumerate(result['sources']):
            print(f"   {i+1}. (score: {source['score']:.3f})")
            print(f"      {source['content'][:100]}...")
        
        print(f"\n🤖 Answer: {result['answer']}")
        print("-" * 40)


def explain_rag_benefits():
    """Explain the benefits of RAG"""
    print("\n📚 Benefits of RAG")
    print("=" * 60)
    print("""
    1. Reduces Hallucinations:
       - LLM answers are grounded in retrieved facts
       - Can cite sources for verification
    
    2. Up-to-date Knowledge:
       - No need to retrain for new information
       - Just update the document store
    
    3. Domain Adaptation:
       - Add domain-specific documents
       - No fine-tuning required
    
    4. Transparency:
       - Can show which documents were used
       - Easier to debug and audit
    
    5. Cost Effective:
       - Smaller LLM + retrieval can match larger LLM
       - Documents are cheaper to update than models
    """)


def explain_rag_variants():
    """Explain RAG-Sequence vs RAG-Token"""
    print("\n📚 RAG Variants")
    print("=" * 60)
    print("""
    RAG-Sequence:
    - Retrieve documents once for the entire response
    - Marginalize over documents at sequence level
    - P(y|x) = Σ_z P(z|x) P(y|x,z)
    - More efficient, simpler
    
    RAG-Token:
    - Can use different documents for each token
    - Marginalize at token level
    - P(y_i|x, y_{<i}) = Σ_z P(z|x) P(y_i|x, y_{<i}, z)
    - More flexible, more expensive
    
    Modern Practice:
    - Most systems use RAG-Sequence style
    - Multi-hop RAG: retrieve → generate → retrieve → generate
    - Iterative refinement based on generated content
    """)


def demo():
    """Main demonstration"""
    demo_rag_pipeline()
    explain_rag_benefits()
    explain_rag_variants()
    
    print("\n" + "=" * 80)
    print("📚 RAG in Practice")
    print("=" * 80)
    print("""
    Components of a Production RAG System:
    
    1. Document Processing:
       - Chunking (split documents into passages)
       - Cleaning and normalization
       - Metadata extraction
    
    2. Embedding:
       - Dense embeddings (sentence-transformers, OpenAI)
       - Sparse embeddings (BM25, SPLADE)
       - Hybrid approaches
    
    3. Vector Store:
       - FAISS, Pinecone, Weaviate, Chroma
       - Approximate nearest neighbor search
    
    4. Retrieval:
       - Semantic search
       - Re-ranking
       - Filtering by metadata
    
    5. Generation:
       - Context injection
       - Prompt engineering
       - Citation generation
    
    Advanced Techniques:
    - HyDE: Generate hypothetical document, then retrieve
    - Self-RAG: Model decides when to retrieve
    - CRAG: Correct and refine retrieved passages
    """)
    print("=" * 80)


if __name__ == "__main__":
    demo()
