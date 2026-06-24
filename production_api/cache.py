import hashlib
import time
from typing import Optional

class ResponseCache:
    def __init__(self,ttl_seconds: int=300):
        self.ttl = ttl_seconds
        self.cache:dict[str,dict] = {}
        self.hits = 0
        self._misses = 0

    def _make_key(self,query:str) -> str:
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self,query:str)-> Optional[str]:
        key = self._make_key(query)

        if key in self.cache:
            entry = self.cache[key]

            if time.time() - entry['timestamp'] < self.ttl:
                self.hits += 1
                return entry['response']
            else:
                del self.cache[key]

        self._misses+=1
        return None        
    
    def set(self, query: str, response: str) -> None:
        key = self._make_key(query)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
    }

    @property
    def stats(self)-> dict:
        total = self.hits + self._misses 
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits':self.hits,
            'misses':self._misses,
            'hit_rate':f"{hit_rate:.1%}",
            'cached_entries': len(self.cache),
        }