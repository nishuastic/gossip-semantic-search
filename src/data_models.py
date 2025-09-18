from typing import List, Optional

from pydantic import BaseModel

class Article(BaseModel):
    title: str
    link: str
    published: str
    summary: str
    content: Optional[str] = None
    category: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    categories: Optional[List[str]] = None


class SearchResult(BaseModel):
    title: str
    url: str
    summary: str
    category: str
    published: str
    score: Optional[float] = None


class SearchMetrics(BaseModel):
    elapsed_ms: int
    top_k: int
    total_vectors: int
    filtered: bool


class SearchResponse(BaseModel):
    results: List[SearchResult]
    metrics: SearchMetrics

class Query(BaseModel):
    """Search request payload for the semantic search endpoint."""

    query: str
    top_k: int = 5
    categories: Optional[List[str]] = None