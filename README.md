# Healthcare AI Agent

Production-grade healthcare Q&A AI agent built on Azure.

## Features
- PHI guardrails (HIPAA compliant)
- Semantic caching (80% cost reduction)
- Two-tier model routing (gpt-4o-mini + gpt-4o)
- RAG pipeline (Azure AI Search)
- Multi-turn memory (Azure Cosmos DB)
- Live external APIs (FDA, CMS)
- CI/CD pipeline (GitHub Actions)

## Architecture
User → PHI Guardrails → Semantic Cache → Query Router → Agent Tools → Cosmos DB Memory → Answer

## Tech Stack
- OpenAI GPT-4o + GPT-4o-mini
- Azure AI Search (RAG)
- Azure Cosmos DB (conversation memory)
- Azure Blob Storage (document store)
- Azure Container Apps (deployment)
- FastAPI + Docker
