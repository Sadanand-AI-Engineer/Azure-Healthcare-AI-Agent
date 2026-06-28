"""
Semantic Cache — Layer 2 of the pipeline
Uses Redis + embeddings to cache similar questions.
"What is a deductible?" and "Explain deductible to me"
are semantically similar — same cached answer returned.
Cost saving: 70-80% reduction in LLM calls at scale.
"""

import os
import json
import hashlib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Similarity threshold — how similar must two questions be
# to return the same cached answer
# 0.95 = very similar (safe for factual Q&A)
# 0.90 = somewhat similar (more aggressive caching)
SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))


def get_embedding(text: str) -> list[float]:
    """Convert text to vector embedding for similarity comparison."""
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns value between -1 and 1. Higher = more similar.
    1.0 = identical meaning
    0.0 = unrelated
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class SemanticCache:
    """
    In-memory semantic cache for development.
    In production: swap self.cache dict with Redis calls.
    Same interface — zero code changes in the agent.
    """

    def __init__(self):
        # In production this is Redis
        # For now: in-memory dict so we can test without Redis
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        print("SemanticCache initialized (in-memory mode)")

    def _make_key(self, embedding: list) -> str:
        """Create a hashable key from embedding for storage."""
        return hashlib.md5(
            json.dumps(embedding[:10]).encode()
        ).hexdigest()

    def get(self, question: str) -> dict | None:
        """
        Look up a question in cache.
        Returns cached answer if similar question found.
        Returns None if cache miss.
        """
        if not self.cache:
            self.miss_count += 1
            return None

        # Get embedding for the new question
        question_embedding = get_embedding(question)

        # Compare against all cached embeddings
        best_similarity = 0
        best_entry = None

        for key, entry in self.cache.items():
            similarity = cosine_similarity(
                question_embedding,
                entry["embedding"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        # Return cached answer if similarity above threshold
        if best_similarity >= SIMILARITY_THRESHOLD and best_entry:
            self.hit_count += 1
            print(f"Cache HIT (similarity: {best_similarity:.3f})")
            return {
                "answer": best_entry["answer"],
                "from_cache": True,
                "similarity": best_similarity,
                "cost_saved": True
            }

        self.miss_count += 1
        print(f"Cache MISS (best similarity: {best_similarity:.3f})")
        return None

    def set(self, question: str, answer: str) -> None:
        """Store a question-answer pair in cache."""
        embedding = get_embedding(question)
        key = self._make_key(embedding)
        self.cache[key] = {
            "question": question,
            "answer": answer,
            "embedding": embedding
        }
        print(f"Cached answer for: {question[:50]}")

    def get_stats(self) -> dict:
        """Return cache performance statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "total_requests": total,
            "cache_hits": self.hit_count,
            "cache_misses": self.miss_count,
            "hit_rate_percent": round(hit_rate, 1),
            "cached_entries": len(self.cache),
            "estimated_cost_saved": f"${self.hit_count * 0.002:.4f}"
        }


# Global cache instance — shared across all requests
cache = SemanticCache()


if __name__ == "__main__":
    print("=== SEMANTIC CACHE TEST ===\n")

    # First question — cache miss, hits LLM
    print("Test 1: First question (cache miss expected)")
    result = cache.get("What is a deductible?")
    print(f"Result: {result}\n")

    # Simulate storing an answer
    cache.set(
        "What is a deductible?",
        "A deductible is the amount you pay for healthcare services "
        "before your insurance begins to pay."
    )

    # Same question — cache hit
    print("Test 2: Same question (cache hit expected)")
    result = cache.get("What is a deductible?")
    print(f"From cache: {result['from_cache'] if result else False}")
    print(f"Answer: {result['answer'][:60] if result else 'None'}\n")

    # Similar question — should also hit cache
    print("Test 3: Similar question (cache hit expected)")
    result = cache.get("Can you explain what a deductible means?")
    print(f"From cache: {result['from_cache'] if result else False}")
    print(f"Similarity: {result['similarity'] if result else 'N/A'}\n")

    # Different question — cache miss
    print("Test 4: Different question (cache miss expected)")
    result = cache.get("How do I file a claim?")
    print(f"From cache: {result['from_cache'] if result else False}\n")

    print("Cache stats:", cache.get_stats())