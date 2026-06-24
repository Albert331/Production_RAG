from pydantic_settings import BaseSettings
from functools import lru_cache


class settings(BaseSettings):
    primary_model:str = 'llama3.2:latest'
    fallback_model:str = 'llama3.2:latest'

    #langsmith 
    langchain_tracing_v2:bool = True
    langchain_api_key:str = 'kuh'
    langchain_project:str = 'production-api'

    app_env:str = 'development'
    log_level:str = 'info'
    rate_limit:str= '20/minute'
    cache_ttl_seconds:int=300
    max_retries:int=3

    database_url:str='postgresql://postgres:UE_i2U4XNM92GyV@db.olrojxeczwyhyisprpop.supabase.co:5432/postgres'
    model_config={'env_file':'.env','extra':'ignore'}

    @property
    def is_production(self)-> bool:
        return self.app_env == 'production'

@lru_cache
def get_settings() -> settings:
    return settings()