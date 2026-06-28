"""
Conversation Memory — Layer 5 of the pipeline
Stores and retrieves conversation history from Azure Cosmos DB.

Why Cosmos DB for conversation memory?
- Every user session = multiple turns (multi-turn agent)
- Each turn stored as a JSON document
- Session retrieved by session_id (partition key)
- Serverless = pay per read/write, not per hour
- Globally distributed = low latency anywhere in the world

Document structure per turn:
{
    "id": "unique-message-id",
    "session_id": "user_sarah_001",     ← partition key
    "role": "user" or "assistant",
    "content": "Does my plan cover PT?",
    "timestamp": "2026-06-27T14:23:11Z",
    "model_used": "gpt-4o-mini",
    "from_cache": false,
    "tool_calls": ["search_policy_coverage"],
    "tokens_used": 234,
    "phi_redacted": false
}
"""

import os
import uuid
from datetime import datetime, timezone
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv()


class ConversationMemory:
    """
    Manages conversation history in Azure Cosmos DB.

    One instance per application startup.
    Methods called per conversation turn.

    Cosmos DB concepts used:
    - Database: healthcare_agent
    - Container: conversations
    - Partition key: /session_id
      (all messages from one session stored together)
    - Document: one JSON object per message turn
    """

    def __init__(self):
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        db_name = os.getenv("COSMOS_DB", "healthcare_agent")
        container_name = os.getenv(
            "COSMOS_CONTAINER", "conversations"
        )

        # Connect to Cosmos DB
        self.client = CosmosClient(endpoint, key)

        # Get or create database
        self.database = self.client.create_database_if_not_exists(
            id=db_name
        )

        # Get or create container with session_id as partition key
        self.container = (
            self.database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/session_id"),
                offer_throughput=None  # None = serverless
            )
        )

        print(f"ConversationMemory connected to Cosmos DB")
        print(f"  Database: {db_name}")
        print(f"  Container: {container_name}")

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_used: str = None,
        from_cache: bool = False,
        tool_calls: list = None,
        tokens_used: int = 0,
        phi_redacted: bool = False
    ) -> str:
        """
        Save one conversation turn to Cosmos DB.

        Called after every user message AND agent response.
        Returns the message_id for reference.

        Role must be "user" or "assistant".
        """
        message_id = str(uuid.uuid4())

        document = {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_used": model_used,
            "from_cache": from_cache,
            "tool_calls": tool_calls or [],
            "tokens_used": tokens_used,
            "phi_redacted": phi_redacted
        }

        self.container.create_item(body=document)
        return message_id

    def get_conversation_history(
        self,
        session_id: str,
        last_n: int = 10
    ) -> list[dict]:
        """
        Retrieve recent conversation history for a session.

        Returns last N turns in chronological order.
        Used to build the messages array for the LLM call.

        Why last_n=10?
        - Older context is less relevant
        - Reduces token cost
        - 10 turns = 5 exchanges = enough context for follow-ups
        """
        query = (
            f"SELECT * FROM c WHERE c.session_id = '{session_id}' "
            f"ORDER BY c.timestamp DESC OFFSET 0 LIMIT {last_n}"
        )

        items = list(
            self.container.query_items(
                query=query,
                partition_key=session_id
            )
        )

        # Reverse to get chronological order (oldest first)
        items.reverse()

        # Format for LLM messages array
        messages = []
        for item in items:
            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

        return messages

    def get_session_stats(self, session_id: str) -> dict:
        """
        Get statistics for a conversation session.
        Used for monitoring and cost tracking.
        """
        query = (
            f"SELECT "
            f"COUNT(1) as total_messages, "
            f"SUM(c.tokens_used) as total_tokens "
            f"FROM c WHERE c.session_id = '{session_id}'"
        )

        items = list(
            self.container.query_items(
                query=query,
                partition_key=session_id
            )
        )

        if items:
            stats = items[0]
            return {
                "session_id": session_id,
                "total_messages": stats.get("total_messages", 0),
                "total_tokens": stats.get("total_tokens", 0),
                "estimated_cost": (
                    f"${stats.get('total_tokens', 0) * 0.000002:.4f}"
                )
            }
        return {"session_id": session_id, "total_messages": 0}

    def delete_session(self, session_id: str) -> int:
        """
        Delete all messages for a session.
        Called when user requests to clear history.
        Returns number of messages deleted.
        """
        query = (
            f"SELECT c.id FROM c "
            f"WHERE c.session_id = '{session_id}'"
        )

        items = list(
            self.container.query_items(
                query=query,
                partition_key=session_id
            )
        )

        count = 0
        for item in items:
            self.container.delete_item(
                item=item["id"],
                partition_key=session_id
            )
            count += 1

        return count


# Global memory instance
memory = ConversationMemory()


if __name__ == "__main__":
    print("=== CONVERSATION MEMORY TEST ===\n")

    session_id = "test_session_001"

    # Test 1: Save a conversation
    print("Test 1: Save user message")
    msg_id = memory.save_message(
        session_id=session_id,
        role="user",
        content="Does my plan cover knee replacement?",
        phi_redacted=False
    )
    print(f"  Saved message ID: {msg_id}")

    print("Test 2: Save agent response")
    msg_id = memory.save_message(
        session_id=session_id,
        role="assistant",
        content=(
            "Yes, total knee replacement (CPT 27447) is covered "
            "after prior authorization. You must meet medical "
            "necessity criteria including 3 months conservative "
            "treatment failure."
        ),
        model_used="gpt-4o-mini",
        from_cache=False,
        tool_calls=["search_prior_auth_criteria"],
        tokens_used=245
    )
    print(f"  Saved message ID: {msg_id}")

    # Test 3: Save follow-up question
    print("Test 3: Save follow-up (multi-turn)")
    memory.save_message(
        session_id=session_id,
        role="user",
        content="What documentation do I need to submit?",
        phi_redacted=False
    )

    # Test 4: Retrieve history
    print("\nTest 4: Retrieve conversation history")
    history = memory.get_conversation_history(session_id)
    print(f"  Retrieved {len(history)} messages:")
    for msg in history:
        print(f"  [{msg['role']}]: {msg['content'][:60]}...")

    # Test 5: Session stats
    print("\nTest 5: Session statistics")
    stats = memory.get_session_stats(session_id)
    print(f"  Stats: {stats}")

    # Test 6: Delete session
    print("\nTest 6: Delete session")
    deleted = memory.delete_session(session_id)
    print(f"  Deleted {deleted} messages")

    # Verify deletion
    history = memory.get_conversation_history(session_id)
    print(f"  Messages after deletion: {len(history)}")

    print("\n✅ Memory tests complete")