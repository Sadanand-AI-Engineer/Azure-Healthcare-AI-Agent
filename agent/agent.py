"""
Main Agent Loop — ties all 5 layers together.

Flow for every incoming question:
1. guardrails.py  → validate, block harmful, redact PHI
2. cache.py       → check if similar question answered before
3. router.py      → simple (gpt-4o-mini) or complex (gpt-4o)?
4. tools.py       → GPT-4o calls tools to look up information
5. memory.py      → load history, save turn to Cosmos DB
6. Return answer with source citations

This is what makes it an AGENT:
- It reasons about WHICH tool to call
- It can call MULTIPLE tools in one turn
- It remembers PREVIOUS turns in the conversation
- It ROUTES to the cheapest model that can answer correctly
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from agent.guardrails import process_input
from agent.cache import cache
from agent.router import router
from agent.tools import TOOL_DEFINITIONS, execute_tool
from agent.memory import memory

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a healthcare insurance assistant for a US health insurance company.

You help hospital staff, doctors, and patients with:
- Insurance coverage questions (what is covered, copays, deductibles)
- Prior authorization requirements and criteria
- Drug interactions and formulary information
- Medicare and CMS coverage questions

You have access to 4 tools:
1. search_policy_coverage — search our insurance policy documents
2. search_prior_auth_criteria — find prior auth requirements
3. check_drug_interaction_fda — get real FDA drug interaction data
4. get_cms_coverage_data — get real Medicare/CMS data

Rules you must follow:
- ALWAYS search before answering policy questions
- ALWAYS cite your source (which tool, which document)
- If you find conflicting information, state both and note the conflict
- Never make up coverage details — only state what tools return
- If a question is outside your scope, say so clearly
- For medical emergencies, always direct to 911 or emergency services

Response format:
- Answer the question directly first
- Then provide supporting details from tools
- End with: Source: [tool name] — [document or API name]"""


class HealthcareAgent:
    """
    Main agent class — one instance per application.
    Handles all incoming questions through the full pipeline.
    """

    def __init__(self):
        print("HealthcareAgent initialized")
        print("  Guardrails: active")
        print("  Cache: active")
        print("  Router: active (simple→gpt-4o-mini, complex→gpt-4o)")
        print("  Tools: 4 tools loaded")
        print("  Memory: Cosmos DB connected")

    def _build_messages(
        self,
        user_message: str,
        session_id: str
    ) -> list[dict]:
        """
        Build the messages array for the LLM call.
        Includes system prompt + conversation history + new message.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Load last 10 turns from Cosmos DB
        history = memory.get_conversation_history(
            session_id=session_id,
            last_n=10
        )
        messages.extend(history)

        # Add the new user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _run_tool_loop(
        self,
        messages: list[dict],
        model: str
    ) -> tuple[str, list[str]]:
        """
        Run the agent reasoning loop.

        GPT-4o decides which tools to call.
        We execute them and return results.
        Loop continues until GPT-4o stops calling tools.

        Returns: (final_answer, list_of_tools_called)
        """
        tools_called = []

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

            # No tool calls — GPT-4o has its final answer
            if not msg.tool_calls:
                return msg.content, tools_called

            # Execute each tool call GPT-4o requested
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  Tool call: {tool_name}({tool_args})")
                tool_result = execute_tool(tool_name, tool_args)
                tools_called.append(tool_name)

                # Add tool result back to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

    def ask(
        self,
        question: str,
        session_id: str
    ) -> dict:
        """
        MAIN ENTRY POINT — process one question from a user.

        Args:
            question:   raw user question
            session_id: unique session identifier for this user

        Returns dict with:
            answer:        the agent's response
            from_cache:    True if answered from cache
            model_used:    which LLM was used
            tools_called:  which tools were invoked
            phi_redacted:  True if PHI was found and removed
            blocked:       True if question was blocked
            block_reason:  why it was blocked (if blocked)
        """

        print(f"\n{'='*60}")
        print(f"Session: {session_id}")
        print(f"Question: {question[:80]}")
        print(f"{'='*60}")

        # ─────────────────────────────────────
        # LAYER 1: Guardrails
        # ─────────────────────────────────────
        guardrail_result = process_input(question)

        if guardrail_result["blocked"]:
            print(f"BLOCKED: {guardrail_result['block_reason']}")
            memory.save_message(
                session_id=session_id,
                role="user",
                content=question,
                phi_redacted=False
            )
            block_response = (
                f"I'm unable to process this request: "
                f"{guardrail_result['block_reason']}"
            )
            memory.save_message(
                session_id=session_id,
                role="assistant",
                content=block_response
            )
            return {
                "answer": block_response,
                "from_cache": False,
                "model_used": None,
                "tools_called": [],
                "phi_redacted": False,
                "blocked": True,
                "block_reason": guardrail_result["block_reason"]
            }

        # Use safe message (PHI redacted if needed)
        safe_question = guardrail_result["safe_message"]
        phi_redacted = guardrail_result["phi_redacted"]

        if phi_redacted:
            print(f"PHI redacted: {guardrail_result['phi_detected']}")

        # ─────────────────────────────────────
        # LAYER 2: Semantic Cache
        # ─────────────────────────────────────
        cached = cache.get(safe_question)
        if cached:
            print(f"Cache HIT — returning cached answer")
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
                "similarity": cached["similarity"],
                "model_used": "cache",
                "tools_called": [],
                "phi_redacted": phi_redacted,
                "blocked": False,
                "block_reason": None
            }

        # ─────────────────────────────────────
        # LAYER 3: Query Router
        # ─────────────────────────────────────
        routing = router.route(safe_question)
        model = routing["model"]
        print(f"Router: {routing['category']} → {model} "
              f"(tier {routing['tier']})")

        # ─────────────────────────────────────
        # LAYER 4: Agent Tool Loop
        # ─────────────────────────────────────
        messages = self._build_messages(safe_question, session_id)
        answer, tools_called = self._run_tool_loop(messages, model)

        print(f"Answer generated using {len(tools_called)} tool(s)")
        print(f"Tools: {tools_called}")

        # ─────────────────────────────────────
        # LAYER 2: Update Cache
        # ─────────────────────────────────────
        cache.set(safe_question, answer)

        # ─────────────────────────────────────
        # LAYER 5: Save to Memory
        # ─────────────────────────────────────
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

        return {
            "answer": answer,
            "from_cache": False,
            "model_used": model,
            "tools_called": tools_called,
            "phi_redacted": phi_redacted,
            "blocked": False,
            "block_reason": None,
            "routing": routing
        }


# Global agent instance
agent = HealthcareAgent()


if __name__ == "__main__":
    print("=== HEALTHCARE AGENT END-TO-END TEST ===\n")

    session = "test_e2e_001"

    # Test 1: Simple coverage question
    print("\nTEST 1: Simple coverage question")
    result = agent.ask(
        "What is my copay for a specialist visit?",
        session_id=session
    )
    print(f"\nANSWER: {result['answer'][:300]}")
    print(f"Model: {result['model_used']}")
    print(f"Tools: {result['tools_called']}")
    print(f"Cache: {result['from_cache']}")

    # Test 2: Same question — should hit cache
    print("\nTEST 2: Same question (cache hit expected)")
    result = agent.ask(
        "What is my copay for a specialist visit?",
        session_id=session
    )
    print(f"From cache: {result['from_cache']}")

    # Test 3: Prior auth question
    print("\nTEST 3: Prior auth question")
    result = agent.ask(
        "What documentation do I need for knee replacement prior auth?",
        session_id=session
    )
    print(f"\nANSWER: {result['answer'][:300]}")
    print(f"Tools: {result['tools_called']}")

    # Test 4: Drug interaction — calls FDA API
    print("\nTEST 4: Drug interaction (FDA API)")
    result = agent.ask(
        "What are the drug interactions between metformin and lisinopril?",
        session_id=session
    )
    print(f"\nANSWER: {result['answer'][:300]}")
    print(f"Tools: {result['tools_called']}")

    # Test 5: PHI detection
    print("\nTEST 5: PHI detection")
    result = agent.ask(
        "My SSN is 123-45-6789. Does my plan cover physical therapy?",
        session_id=session
    )
    print(f"PHI redacted: {result['phi_redacted']}")
    print(f"\nANSWER: {result['answer'][:300]}")

    # Test 6: Multi-turn follow-up
    print("\nTEST 6: Multi-turn follow-up")
    result = agent.ask(
        "How many physical therapy visits am I covered for per year?",
        session_id=session
    )
    print(f"\nANSWER: {result['answer'][:300]}")

    # Test 7: Blocked question
    print("\nTEST 7: Harmful content blocked")
    result = agent.ask(
        "How to overdose on medication?",
        session_id=session
    )
    print(f"Blocked: {result['blocked']}")
    print(f"Reason: {result['block_reason']}")

    # Show cache stats
    print("\n=== CACHE STATS ===")
    print(cache.get_stats())

    # Show session stats
    print("\n=== SESSION STATS ===")
    print(memory.get_session_stats(session))

    # Clean up test session
    memory.delete_session(session)
    print("\n✅ All tests complete")