"""
FastAPI Web Server — Healthcare AI Agent API v1.0.3
Lazy initialization, full pipeline metrics in response.
"""

import os
import uuid
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
_agent = None
_cache = None
_memory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all connections after app starts."""
    global _agent, _cache, _memory

    logger.info("=" * 50)
    logger.info("Healthcare AI Agent v1.0.3 starting...")
    logger.info("=" * 50)

    env_vars = [
        "OPENAI_API_KEY", "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY", "COSMOS_ENDPOINT", "COSMOS_KEY"
    ]
    for var in env_vars:
        status = "✅ SET" if os.getenv(var) else "❌ MISSING"
        logger.info(f"  {var}: {status}")

    try:
        from agent.cache import SemanticCache
        _cache = SemanticCache()
        logger.info("  ✅ Semantic cache initialized")

        from agent.memory import ConversationMemory
        _memory = ConversationMemory()
        logger.info("  ✅ Cosmos DB memory connected")

        from agent.agent import HealthcareAgent
        _agent = HealthcareAgent()
        logger.info("  ✅ Healthcare agent ready")

        logger.info("🚀 Agent READY")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Healthcare AI Agent API",
    description=(
        "Production-grade healthcare Q&A agent with PHI guardrails, "
        "semantic caching, two-tier model routing, RAG pipeline, "
        "multi-turn memory, intent detection, and NPI doctor search."
    ),
    version="1.0.3",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────
# Models
# ─────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        example="Does my plan cover physical therapy?"
    )
    session_id: str = Field(
        default=None,
        example="user_001"
    )


class AskResponse(BaseModel):
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
    intent: str | None = None
    metrics: dict | None = None


# ─────────────────────────────────────
# Middleware
# ─────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration}s)"
    )
    return response


# ─────────────────────────────────────
# Endpoints
# ─────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check — called by Azure Container Apps every 30s."""
    return {
        "status": "healthy" if _agent is not None else "starting",
        "version": "1.0.3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "agent": "ready" if _agent is not None else "initializing",
            "cache": "ready" if _cache is not None else "initializing",
            "memory": "ready" if _memory is not None else "initializing",
            "openai_key": "set" if os.getenv("OPENAI_API_KEY") else "missing",
            "search": "set" if os.getenv("AZURE_SEARCH_ENDPOINT") else "missing",
            "cosmos": "set" if os.getenv("COSMOS_ENDPOINT") else "missing"
        }
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Main chat endpoint — processes one healthcare question.

    Pipeline:
    0. Intent detection (off-topic redirect)
    1. PHI guardrails (HIPAA compliance)
    2. Semantic cache (cost optimization)
    3. Query router (model selection)
    4. Agent tool calls (RAG + external APIs)
    5. Conversation memory (multi-turn context)

    Returns full pipeline metrics — safe for display.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent initializing. Please try again in 30 seconds."
        )

    start = time.time()
    request_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        f"[{request_id[:8]}] Q: {request.question[:60]}"
    )

    try:
        result = _agent.ask(
            question=request.question,
            session_id=session_id
        )

        processing_time = round(time.time() - start, 3)

        logger.info(
            f"[{request_id[:8]}] Done {processing_time}s | "
            f"Model: {result.get('model_used')} | "
            f"Cache: {result.get('from_cache')} | "
            f"Tools: {result.get('tools_called')} | "
            f"Intent: {result.get('intent')}"
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
            timestamp=datetime.now(timezone.utc).isoformat(),
            intent=result.get("intent"),
            metrics=result.get("metrics")
        )

    except Exception as e:
        logger.error(f"[{request_id[:8]}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def cache_stats():
    """Cache performance metrics."""
    if _cache is None:
        return {"status": "cache not ready"}
    return {
        "cache_stats": _cache.get_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """Conversation history from Cosmos DB."""
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory not ready")
    try:
        history = _memory.get_conversation_history(
            session_id=session_id, last_n=20
        )
        stats = _memory.get_session_stats(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history,
            "stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session — HIPAA right to deletion."""
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory not ready")
    try:
        deleted = _memory.delete_session(session_id)
        return {
            "session_id": session_id,
            "messages_deleted": deleted,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Healthcare AI Agent API",
        "version": "1.0.3",
        "status": "running",
        "agent_ready": _agent is not None,
        "capabilities": [
            "Insurance coverage Q&A",
            "Prior authorization criteria",
            "FDA drug interactions",
            "Medicare/CMS data",
            "NPI doctor search",
            "PHI detection and redaction",
            "Semantic caching",
            "Multi-turn conversation memory",
            "Intent detection"
        ],
        "endpoints": {
            "POST /ask": "Ask a healthcare question",
            "GET /health": "Health check",
            "GET /cache/stats": "Cache performance",
            "GET /session/{id}": "Conversation history",
            "DELETE /session/{id}": "Clear session (HIPAA)",
            "GET /docs": "Swagger UI"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }