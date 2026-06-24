import time 
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI,Request,HTTPException,UploadFile,File
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv  
from config import get_settings
from models import (
    ChatResponse, chatRequest, HealthResponse, MetricResponse, errorResponse
)
from security import SecurePipeline, RequestTimer
from cache import ResponseCache
from monitoring import MetricsCollector
from agent import productionAgent
from ingestor import ingest_pdf

load_dotenv()


@asynccontextmanager
async def lifespan(app:FastAPI):
    global security,cache,metrics,agent

    settings = get_settings()

    security = SecurePipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = productionAgent()

    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title='production Langraph API',
    description = 'production ready chat api',
    version='1.0.0',
    lifespan=lifespan

)
app.state.limiter = limiter

@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDFs allowed')
    
    file_bytes = await file.read()
    chunks = ingest_pdf(file_bytes, file.filename, agent.vector_store)
    
    return {'message': f'ingested {chunks} chunks from {file.filename}'}

@app.post('/chat',response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name='chat_endpoint')
async def chat(request:Request,body:chatRequest):
    with RequestTimer() as timer:
        security_notes = []

        is_allowed,cleaned,notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            raise HTTPException(status_code=400,detail='your message was blocked by security filters')
        

        cached_response = cache.get(cleaned)
        if cached_response is not None:
            return ChatResponse(
                response = cached_response,
                thread_id=body.thread_id,
                model_used='cache',
                cached=True,
                processing_time_ms=0,
                timestamp=datetime.utcnow().isoformat()
            )
         
        try:
            result = agent.invoke(cleaned)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail = "an error occured whie processig thy request"
            )    

        response_text = result['response']
        model_used= result['model_used']

        is_secure,validated_response, output_warnings = security.check_output(response_text)
        security_notes.append(output_warnings)
       
        cache.set(cleaned,validated_response)


        return ChatResponse(
            response=validated_response,
            thread_id=body.thread_id,
            model_used=model_used,
            cached=False,
            processing_time_ms=round(timer.elapsed_ms,2),
            timestamp=datetime.utcnow().isoformat()
        )
    


print('hello world')    