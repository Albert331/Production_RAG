# Production RAG API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, LangGraph, and Docker. Features intelligent query routing, hybrid retrieval, security pipeline, and response caching.

## Architecture

```
User Query → Security Pipeline → Cache Check → LangGraph Agent
                                                      ↓
                                            Query Router (LLM)
                                           ↙              ↘
                                    Retrieval          Conversational
                                       ↓                    ↓
                                  PGVector              Direct LLM
                                  Retriever               Response
                                       ↓
                                  LLM + Context
                                       ↓
                                Output Validator → Cache → Response
```

## Features

- **LangGraph Agent** — stateful agent with primary/fallback LLM and error handling
- **Intelligent Query Routing** — LLM decides whether to retrieve or respond directly
- **Security Pipeline** — prompt injection detection, PII redaction, output validation
- **Response Caching** — SHA-256 keyed cache with configurable TTL
- **Rate Limiting** — per-IP rate limiting via SlowAPI
- **PDF Ingestion** — upload PDFs via REST endpoint, chunked and stored in PGVector
- **Dockerized** — fully containerized with PostgreSQL + pgvector

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Agent | LangGraph |
| LLM | Ollama (llama3.2) |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | PostgreSQL + pgvector |
| Observability | LangSmith |
| Deployment | Docker + Docker Compose |

## Getting Started

### Prerequisites

- Docker Desktop
- Ollama running locally with `llama3.2` and `nomic-embed-text` pulled

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Setup

1. Clone the repo

```bash
git clone https://github.com/Albert331/Production_RAG.git
cd Production_RAG
```

2. Create `.env` file

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=production-api
DATABASE_URL=postgresql+psycopg2://postgres:password@db:5432/ragdb
```

3. Run with Docker

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`

## API Endpoints

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what is self attention?", "thread_id": "123"}'
```

### Upload PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"
```


## Security

- Prompt injection detection with regex pattern matching
- PII masking (email, phone, SSN, credit card, IP)
- Output validation before returning to user
- Rate limiting: 20 requests/minute per IP

## Project Structure

```
production_api/
├── main.py          # FastAPI app, endpoints, lifespan
├── agent.py         # LangGraph agent with query routing
├── security.py      # Input sanitizer, PII detector, output validator
├── cache.py         # In-memory response cache
├── ingestor.py      # PDF ingestion pipeline
├── monitoring.py    # Metrics collector
├── models.py        # Pydantic models
├── config.py        # Settings via pydantic-settings
└── utils.py         # RequestTimer
docker-compose.yml
Dockerfile
```
