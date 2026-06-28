"""
Query Router — Layer 3 of the pipeline
Decides which model to use for each question.
Simple → gpt-4o-mini (fast, cheap)
Complex → gpt-4o (powerful, accurate)

Two-tier approach:
Tier 1: Embedding similarity (instant, free)
Tier 2: LLM judge (only when uncertain)

This is Gokul's semantic router pattern.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from agent.cache import get_embedding, cosine_similarity

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SIMPLE_MODEL = os.getenv("SIMPLE_MODEL", "gpt-4o-mini")
COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", "gpt-4o")
CONFIDENCE_THRESHOLD = 0.15

# --- Category definitions ---
# These are example questions for each category.
# Router embeds incoming question and compares against these.
# The category with highest average similarity wins.

SIMPLE_EXAMPLES = [
    "What is a deductible?",
    "What does copay mean?",
    "Is physical therapy covered?",
    "What is a premium?",
    "What is an EOB?",
    "Does my plan cover dental?",
    "What is out of pocket maximum?",
    "How do I get a referral?",
    "What is coinsurance?",
    "Is my doctor in network?",
]

COMPLEX_EXAMPLES = [
    "Analyze this prior authorization denial and suggest appeal strategy",
    "What are the drug interactions between metformin and lisinopril?",
    "Explain why my claim was denied based on diagnosis code M17.11",
    "What is the medical necessity criteria for total knee replacement?",
    "Compare coverage differences between PPO and HMO for cancer treatment",
    "How do I appeal a denied claim for experimental treatment?",
    "What documentation do I need for a prior auth for knee surgery?",
    "Explain the step therapy requirements for rheumatoid arthritis medication",
]


class QueryRouter:
    """
    Routes questions to the appropriate model.

    Attributes:
        simple_embeddings: pre-computed embeddings for simple examples
        complex_embeddings: pre-computed embeddings for complex examples

    The embeddings are computed ONCE at startup and reused.
    This means Tier 1 routing costs zero API calls after init.
    """

    def __init__(self):
        print("Initializing QueryRouter — pre-computing category embeddings...")
        # Pre-compute embeddings for all example questions
        # Done once at startup — not on every request
        self.simple_embeddings = [
            get_embedding(q) for q in SIMPLE_EXAMPLES
        ]
        self.complex_embeddings = [
            get_embedding(q) for q in COMPLEX_EXAMPLES
        ]
        print(f"Router ready. "
              f"Simple examples: {len(self.simple_embeddings)}, "
              f"Complex examples: {len(self.complex_embeddings)}")

    def _get_category_score(
        self, question_embedding: list, category_embeddings: list
    ) -> float:
        """
        Calculate how similar a question is to a category.
        Takes the AVERAGE similarity across all category examples.
        Returns score between 0 and 1.
        """
        similarities = [
            cosine_similarity(question_embedding, cat_emb)
            for cat_emb in category_embeddings
        ]
        return sum(similarities) / len(similarities)

    def _tier1_route(self, question: str) -> dict:
        """
        Tier 1: Embedding-based routing.
        Fast — no LLM call.
        Returns routing decision with confidence score.
        """
        question_embedding = get_embedding(question)

        simple_score = self._get_category_score(
            question_embedding, self.simple_embeddings
        )
        complex_score = self._get_category_score(
            question_embedding, self.complex_embeddings
        )

        # Determine which category scored higher
        if simple_score > complex_score:
            confidence = simple_score - complex_score
            category = "simple"
            model = SIMPLE_MODEL
        else:
            confidence = complex_score - simple_score
            category = "complex"
            model = COMPLEX_MODEL

        return {
            "category": category,
            "model": model,
            "confidence": confidence,
            "simple_score": round(simple_score, 3),
            "complex_score": round(complex_score, 3),
            "tier": 1
        }

    def _tier2_route(self, question: str) -> dict:
        """
        Tier 2: LLM-as-judge routing.
        Used only when Tier 1 confidence is low.
        Costs one gpt-4o-mini call (~$0.0001).
        """
        print("Low confidence — using Tier 2 LLM judge...")
        response = client.chat.completions.create(
            model=SIMPLE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a query classifier for a healthcare Q&A system.
Classify the user question as either 'simple' or 'complex'.

simple: Basic definitions, yes/no coverage questions, 
        straightforward policy lookups
complex: Multi-step reasoning, clinical analysis, appeal strategies,
         drug interactions, diagnosis-specific questions

Respond with ONLY one word: simple or complex"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=10,
            temperature=0
        )

        category = response.choices[0].message.content.strip().lower()
        if category not in ["simple", "complex"]:
            category = "simple"  # safe default

        model = SIMPLE_MODEL if category == "simple" else COMPLEX_MODEL

        return {
            "category": category,
            "model": model,
            "confidence": 1.0,
            "tier": 2
        }

    def route(self, question: str) -> dict:
        """
        MAIN FUNCTION — routes a question to the right model.

        Flow:
        1. Try Tier 1 (embedding similarity)
        2. If confidence < threshold → try Tier 2 (LLM judge)
        3. Return model name + routing metadata

        The metadata goes to Azure Monitor for cost analysis.
        """
        tier1_result = self._tier1_route(question)

        if tier1_result["confidence"] >= CONFIDENCE_THRESHOLD:
            print(f"Tier 1 route: {tier1_result['category']} "
                  f"(confidence: {tier1_result['confidence']:.3f})")
            return tier1_result
        else:
            print(f"Tier 1 uncertain "
                  f"(confidence: {tier1_result['confidence']:.3f})")
            return self._tier2_route(question)


# Global router instance
router = QueryRouter()


if __name__ == "__main__":
    print("\n=== QUERY ROUTER TEST ===\n")

    test_questions = [
        "What is a deductible?",
        "Is physical therapy covered under my plan?",
        "Analyze my prior auth denial for knee replacement CPT 27447",
        "What are drug interactions between metformin and lisinopril?",
        "What does EOB mean?",
        "How do I appeal a denied claim for experimental cancer treatment?",
    ]

    for question in test_questions:
        print(f"Question: {question[:60]}")
        result = router.route(question)
        print(f"  → Model: {result['model']}")
        print(f"  → Category: {result['category']}")
        print(f"  → Tier used: {result['tier']}")
        print(f"  → Confidence: {result['confidence']:.3f}")
        print()