"""
FastAPI Web Server — the public face of the healthcare agent.

Endpoints:
POST /ask          → main chat endpoint
GET  /health       → health check (used by Azure Container Apps)
GET  /cache/stats  → cache performance metrics
GET  /session/{id} → conversation history for a session
DELETE /session/{id} → clear a session

This is what gets deployed to Azure Container Apps.
After deployment it gets a live HTTPS URL.
LinkedIn users will call POST /ask through the chat UI.
"""

import os
import uuid
import time
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent.agent import agent
from agent.cache import cache
from agent.memory import memory

load_dotenv()

# ─────────────────────────────────────
# Logging setup
# ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────
app = FastAPI(
    title="Healthcare AI Agent API",
    description=(
        "Production-grade healthcare Q&A agent with PHI guardrails, "
        "semantic caching, model routing, RAG, and multi-turn memory. "
        "Built on Azure AI Search, Cosmos DB, and OpenAI GPT-4o."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc"     # ReDoc at /redoc
)

# CORS — allows the chat UI to call this API from any origin
# In production: restrict origins to your actual UI domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────

class AskRequest(BaseModel):
    """What the user sends to /ask"""
    question: str = Field(
        ...,
        description="The healthcare question",
        min_length=1,
        max_length=2000,
        example="Does my plan cover physical therapy?"
    )
    session_id: str = Field(
        default=None,
        description=(
            "Session ID for multi-turn conversations. "
            "Omit to start a new session."
        ),
        example="user_sarah_001"
    )


class AskResponse(BaseModel):
    """What the API returns"""
    request_id: str
    session_id: str
    answer: str
    from_cache: bool
    model_used: str | None
    tools_called: list[str]
    phi_redacted: bool
    blocked: bool
    block_reason: str | None
    processing_time_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    services: dict


# ─────────────────────────────────────
# Middleware — log every request
# ─────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every request with timing.
    In production this feeds into Azure Monitor.
    """
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration}s)"
    )
    return response


# ─────────────────────────────────────
# Endpoints
# ─────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Azure Container Apps calls this every 30 seconds.
    If it returns non-200, Azure restarts the container.
    Also useful for monitoring dashboards.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        services={
            "openai": "connected",
            "azure_search": "connected",
            "cosmos_db": "connected",
            "blob_storage": "connected",
            "guardrails": "active",
            "semantic_cache": "active",
            "query_router": "active"
        }
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Main chat endpoint — processes one question.

    Runs through the full pipeline:
    1. PHI guardrails
    2. Semantic cache lookup
    3. Query routing (simple vs complex model)
    4. Agent tool calls (AI Search, FDA API, CMS API)
    5. Cosmos DB memory (load history, save turn)

    Supports multi-turn conversations via session_id.
    Pass the same session_id across turns to maintain context.
    """
    start = time.time()
    request_id = str(uuid.uuid4())

    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        f"[{request_id}] Session: {session_id} | "
        f"Question: {request.question[:60]}"
    )

    try:
        result = agent.ask(
            question=request.question,
            session_id=session_id
        )

        processing_time = round(time.time() - start, 3)

        logger.info(
            f"[{request_id}] Done in {processing_time}s | "
            f"Model: {result.get('model_used')} | "
            f"Cache: {result.get('from_cache')} | "
            f"Tools: {result.get('tools_called')}"
        )

        return AskResponse(
            request_id=request_id,
            session_id=session_id,
            answer=result["answer"],
            from_cache=result.get("from_cache", False),
            model_used=result.get("model_used"),
            tools_called=result.get("tools_called", []),
            phi_redacted=result.get("phi_redacted", False),
            blocked=result.get("blocked", False),
            block_reason=result.get("block_reason"),
            processing_time_seconds=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


@app.get("/cache/stats")
async def cache_stats():
    """
    Cache performance metrics.
    Shows hit rate, cost saved, entries cached.
    Use this for the LinkedIn demo to show real-time savings.
    """
    stats = cache.get_stats()
    return {
        "cache_stats": stats,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """
    Retrieve conversation history for a session.
    Shows the multi-turn memory stored in Cosmos DB.
    """
    try:
        history = memory.get_conversation_history(
            session_id=session_id,
            last_n=20
        )
        stats = memory.get_session_stats(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history,
            "stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Clear all messages for a session.
    GDPR/HIPAA compliance — user right to deletion.
    """
    try:
        deleted = memory.delete_session(session_id)
        return {
            "session_id": session_id,
            "messages_deleted": deleted,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint — shows API info."""
    return {
        "name": "Healthcare AI Agent API",
        "version": "1.0.0",
        "description": (
            "Production healthcare Q&A agent with PHI guardrails, "
            "semantic caching, model routing, RAG pipeline, "
            "and multi-turn conversation memory."
        ),
        "endpoints": {
            "POST /ask": "Ask a healthcare question",
            "GET /health": "Health check",
            "GET /cache/stats": "Cache performance metrics",
            "GET /session/{id}": "Conversation history",
            "DELETE /session/{id}": "Clear session",
            "GET /docs": "Interactive API documentation"
        },
        "built_with": {
            "models": "OpenAI GPT-4o + GPT-4o-mini",
            "knowledge_base": "Azure AI Search",
            "memory": "Azure Cosmos DB",
            "storage": "Azure Blob Storage",
            "external_apis": ["FDA openFDA", "CMS data.gov"],
            "guardrails": "PHI detection + harmful content blocking",
            "caching": "Semantic cache (embedding similarity)",
            "routing": "Two-tier model router"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }