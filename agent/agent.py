"""
Main Agent Loop — ties all 5 layers together.

Pipeline for every incoming question:
0. Intent detection    — off-topic? redirect kindly
1. guardrails.py       — PHI redaction + harmful content
2. cache.py            — semantic cache lookup
3. router.py           — simple vs complex model
4. tools.py            — GPT-4o calls tools
5. memory.py           — load history, save turn
Returns answer + full pipeline metrics
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

from agent.guardrails import process_input, detect_intent
from agent.cache import cache
from agent.router import router
from agent.tools import TOOL_DEFINITIONS, execute_tool
from agent.memory import memory

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a Healthcare AI Assistant with access to a DEMO insurance policy database.

IMPORTANT — BE HONEST ABOUT YOUR DATA:
You have access to ONE synthetic demo policy:
- BlueCross Shield Gold PPO 2024 (synthetic/fictional — for demonstration only)

You do NOT have access to:
- Any specific user's real insurance plan
- Real patient records or personal health data
- Real insurer databases

When a user says "my plan" or "my insurance":
- Clarify you are using a demo BlueCross Shield Gold PPO policy
- Say: "I'm using a demo BlueCross Shield Gold PPO policy for this demonstration.
  In a production deployment, this would be replaced with your actual insurer's policy documents."
- Then answer the question based on the demo policy

You CAN answer accurately about:
- The demo BlueCross Shield Gold PPO policy (copays, deductibles, prior auth)
- Real FDA drug interactions (via live FDA API)
- Real licensed doctors (via live NPI Registry)
- Real Medicare ACO data (via live CMS API)

Always cite your source. Never pretend to know a user's personal plan."""


class HealthcareAgent:
    """
    Main agent class connecting all pipeline layers.
    One global instance per application.
    """

    def __init__(self):
        print("HealthcareAgent initialized")
        print("  Layer 0: Intent detection — active")
        print("  Layer 1: Guardrails — active")
        print("  Layer 2: Semantic cache — active")
        print("  Layer 3: Query router — active")
        print("  Layer 4: Tool calls — 5 tools loaded")
        print("  Layer 5: Cosmos DB memory — active")

    def _build_messages(
        self,
        user_message: str,
        session_id: str
    ) -> list[dict]:
        """Build messages array with history for LLM call."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = memory.get_conversation_history(
            session_id=session_id,
            last_n=10
        )
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _run_tool_loop(
        self,
        messages: list[dict],
        model: str
    ) -> tuple[str, list[str], list]:
        """
        Agent reasoning loop.
        GPT-4o decides which tools to call.
        We execute and return results back.
        Returns: (answer, tools_called, tool_results)
        """
        tools_called = []
        tool_results = []

        while True:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=int(os.getenv("MAX_TOKENS", "1000"))
            )

            msg = response.choices[0].message
            messages.append(msg)

            # No more tool calls — final answer ready
            if not msg.tool_calls:
                return msg.content, tools_called, tool_results

            # Execute each tool GPT-4o requested
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  → Tool: {tool_name}({tool_args})")
                tool_result = execute_tool(tool_name, tool_args)
                tools_called.append(tool_name)

                # Store full result dict for metrics
                if isinstance(tool_result, dict):
                    tool_results.append(tool_result)
                    content = tool_result.get(
                        "content", str(tool_result)
                    )
                else:
                    content = str(tool_result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

    def ask(self, question: str, session_id: str) -> dict:
        """
        MAIN ENTRY POINT — process one question through full pipeline.

        Returns:
            answer: the agent's response
            from_cache: True if answered from semantic cache
            model_used: which LLM was used
            tools_called: which tools were invoked
            phi_redacted: True if PHI was found and removed
            blocked: True if question was blocked
            intent: detected intent category
            metrics: full pipeline metrics for display
        """

        print(f"\n{'='*60}")
        print(f"Session: {session_id}")
        print(f"Question: {question[:80]}")
        print(f"{'='*60}")

        start_time = time.time()

        # Initialize metrics — safe to show to users
        metrics = {
            "pipeline_steps": [],
            "cache_hit": False,
            "cache_similarity": 0.0,
            "router_category": None,
            "router_confidence": 0.0,
            "router_tier": None,
            "rag_top_score": 0.0,
            "rag_scores": [],
            "tools_called": [],
            "tool_sources": [],
            "processing_time": 0.0,
            "cost_saved_usd": 0.0,
            "cost_saved_percent": 0
        }

        # ─────────────────────────────────
        # LAYER 0: Intent Detection
        # ─────────────────────────────────
        intent_result = detect_intent(question)
        intent = intent_result["intent"]
        print(f"Intent: {intent} "
              f"(confidence: {intent_result.get('confidence', 0):.2f})")

        metrics["pipeline_steps"].append({
            "step": "Intent Detection",
            "result": intent,
            "confidence": round(
                intent_result.get("confidence", 0) * 100
            )
        })

        # Off-topic — redirect kindly without calling LLM
        if intent == "off_topic":
            metrics["processing_time"] = round(
                time.time() - start_time, 3
            )
            return {
                "answer": intent_result["redirect_message"],
                "from_cache": False,
                "model_used": None,
                "tools_called": [],
                "phi_redacted": False,
                "blocked": False,
                "block_reason": None,
                "intent": intent,
                "metrics": metrics
            }

        # ─────────────────────────────────
        # LAYER 1: PHI Guardrails
        # ─────────────────────────────────
        guardrail_result = process_input(question)

        phi_types = list(
            guardrail_result.get("phi_detected", {}).keys()
        )
        metrics["pipeline_steps"].append({
            "step": "PHI Guardrails",
            "result": (
                "blocked" if guardrail_result["blocked"]
                else "phi_redacted" if guardrail_result["phi_redacted"]
                else "passed"
            ),
            "phi_types": phi_types
        })

        if guardrail_result["blocked"]:
            print(f"BLOCKED: {guardrail_result['block_reason']}")
            block_msg = (
                f"I'm unable to process this request: "
                f"{guardrail_result['block_reason']}"
            )
            memory.save_message(
                session_id=session_id,
                role="user",
                content=question
            )
            memory.save_message(
                session_id=session_id,
                role="assistant",
                content=block_msg
            )
            metrics["processing_time"] = round(
                time.time() - start_time, 3
            )
            return {
                "answer": block_msg,
                "from_cache": False,
                "model_used": None,
                "tools_called": [],
                "phi_redacted": False,
                "blocked": True,
                "block_reason": guardrail_result["block_reason"],
                "intent": intent,
                "metrics": metrics
            }

        safe_question = guardrail_result["safe_message"]
        phi_redacted = guardrail_result["phi_redacted"]

        if phi_redacted:
            print(f"PHI redacted: {phi_types}")

        # ─────────────────────────────────
        # LAYER 2: Semantic Cache
        # ─────────────────────────────────
        cached = cache.get(safe_question)

        if cached:
            similarity = cached["similarity"]
            print(f"Cache HIT — similarity: {similarity:.3f}")
            metrics["cache_hit"] = True
            metrics["cache_similarity"] = round(similarity, 3)
            metrics["cost_saved_usd"] = 0.002
            metrics["cost_saved_percent"] = 100
            metrics["pipeline_steps"].append({
                "step": "Semantic Cache",
                "result": "HIT",
                "similarity": round(similarity * 100)
            })
            metrics["processing_time"] = round(
                time.time() - start_time, 3
            )

            memory.save_message(
                session_id=session_id,
                role="user",
                content=safe_question,
                phi_redacted=phi_redacted
            )
            memory.save_message(
                session_id=session_id,
                role="assistant",
                content=cached["answer"],
                from_cache=True,
                model_used="cache"
            )

            return {
                "answer": cached["answer"],
                "from_cache": True,
                "similarity": similarity,
                "model_used": "cache",
                "tools_called": [],
                "phi_redacted": phi_redacted,
                "blocked": False,
                "block_reason": None,
                "intent": intent,
                "metrics": metrics
            }

        metrics["pipeline_steps"].append({
            "step": "Semantic Cache",
            "result": "MISS",
            "similarity": 0
        })

        # ─────────────────────────────────
        # LAYER 3: Query Router
        # ─────────────────────────────────
        routing = router.route(safe_question)
        model = routing["model"]
        confidence = routing["confidence"]

        metrics["router_category"] = routing["category"]
        metrics["router_confidence"] = round(confidence * 100)
        metrics["router_tier"] = routing["tier"]
        metrics["pipeline_steps"].append({
            "step": "Query Router",
            "result": routing["category"],
            "model": model,
            "confidence": round(confidence * 100),
            "tier": routing["tier"]
        })

        print(f"Router: {routing['category']} → {model} "
              f"(confidence: {confidence:.2f})")

        # Cost saving vs always using gpt-4o
        if model == "gpt-4o-mini":
            metrics["cost_saved_usd"] = round(0.005 - 0.0006, 4)
            metrics["cost_saved_percent"] = 88

        # ─────────────────────────────────
        # LAYER 4: Agent Tool Loop
        # ─────────────────────────────────
        messages = self._build_messages(safe_question, session_id)
        answer, tools_called, tool_results = self._run_tool_loop(
            messages, model
        )

        # Extract RAG scores from tool results
        for result in tool_results:
            if isinstance(result, dict):
                if "rag_scores" in result:
                    metrics["rag_scores"].extend(result["rag_scores"])
                if "top_score" in result:
                    if result["top_score"] > metrics["rag_top_score"]:
                        metrics["rag_top_score"] = result["top_score"]
                if "source" in result:
                    metrics["tool_sources"].append(result["source"])

        metrics["tools_called"] = tools_called
        metrics["pipeline_steps"].append({
            "step": "Agent Tool Calls",
            "tools": tools_called,
            "sources": metrics["tool_sources"],
            "rag_top_score": round(metrics["rag_top_score"] * 100)
        })

        # ─────────────────────────────────
        # Update Cache
        # ─────────────────────────────────
        cache.set(safe_question, answer)

        # ─────────────────────────────────
        # LAYER 5: Save to Memory
        # ─────────────────────────────────
        memory.save_message(
            session_id=session_id,
            role="user",
            content=safe_question,
            phi_redacted=phi_redacted
        )
        memory.save_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            model_used=model,
            from_cache=False,
            tool_calls=tools_called,
            tokens_used=len(answer.split()) * 2
        )

        metrics["processing_time"] = round(time.time() - start_time, 3)

        return {
            "answer": answer,
            "from_cache": False,
            "model_used": model,
            "tools_called": tools_called,
            "phi_redacted": phi_redacted,
            "blocked": False,
            "block_reason": None,
            "intent": intent,
            "routing": routing,
            "metrics": metrics
        }


# Global agent instance
agent = HealthcareAgent()


if __name__ == "__main__":
    print("=== HEALTHCARE AGENT FULL TEST ===\n")

    session = "test_full_001"

    tests = [
        # Intent tests
        ("Can I order pizza?", "off_topic"),
        ("What is my copay for a specialist?", "insurance"),
        ("I have fever and headache for 2 days", "clinical"),
        # PHI test
        ("My SSN is 123-45-6789, am I covered for PT?", "phi"),
        # Drug interaction
        ("What are interactions between metformin and lisinopril?",
         "drug"),
        # Prior auth
        ("What do I need for knee replacement prior auth?", "prior_auth"),
        # Doctor search
        ("I have chest pain, which doctor should I see?", "doctor"),
    ]

    for question, test_type in tests:
        print(f"\n{'─'*50}")
        print(f"TEST [{test_type.upper()}]: {question[:60]}")
        print(f"{'─'*50}")

        result = agent.ask(question=question, session_id=session)

        print(f"Intent:    {result.get('intent')}")
        print(f"Blocked:   {result.get('blocked')}")
        print(f"PHI:       {result.get('phi_redacted')}")
        print(f"Cache:     {result.get('from_cache')}")
        print(f"Model:     {result.get('model_used')}")
        print(f"Tools:     {result.get('tools_called')}")
        print(f"Answer:    {result['answer'][:150]}...")

        metrics = result.get("metrics", {})
        print(f"Time:      {metrics.get('processing_time')}s")
        print(f"RAG score: {metrics.get('rag_top_score')}")
        print(f"Cost saved:{metrics.get('cost_saved_percent')}%")

    memory.delete_session(session)
    print("\n✅ All tests complete")