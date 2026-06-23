from pydantic import BaseModel, Field
from datetime import datetime


class chatRequest(BaseModel):
    message:str= Field(
        ...,
        min_length=1,
        max_length=1000,
        description='the users message to the agent'
    )

    thread_id:str = Field(
        default='default',
        description='conversation thread ID'
    )


class ChatResponse(BaseModel):
    response:str
    thread_id:str
    model_used:str
    cached:bool=False
    processing_time_ms:float
    timestamp:str = Field(directory_factory=lambda:datetime.now())    

class HealthResponse(BaseModel) :
    status:str='healthy'   
    environment:str
    version:str='1.0.0'
    checks:dict={}


class MetricResponse(BaseModel):
    total_req:int
    total_errors:int
    error_rate:str
    avg_latency_ms:float
    cache_hit_rate:str
    total_input_tokens:int
    total_output_tokens:int

class errorResponse(BaseModel):
    error:str
    detail:str|None = None
    request_id = str|None = None