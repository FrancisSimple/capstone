from pydantic import BaseModel
from typing import List

class IngestRequest(BaseModel):
    texts: List[str] # list of strings to learn

class QueryRequest(BaseModel):
    question:str
    