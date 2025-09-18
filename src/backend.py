import os
import time

import pinecone
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from data_models import SearchRequest, SearchResponse, SearchResult, SearchMetrics
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

_model = None
_index = None


def get_model():
    """Return a cached `SentenceTransformer` model, initializing on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_index():
    """Return a cached Pinecone `Index`, initializing on first use."""
    global _index
    if _index is None:
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_KEY"))
        _index = pc.Index("gossip-semantic-search")
    return _index


@app.post("/search")
async def search(query: SearchRequest) -> SearchResponse:
    """Search the Pinecone index for results similar to the input query.

    Returns a list of result items and request/engine metrics.
    """
    # Generate query embedding
    started_at = time.perf_counter()
    query_embedding = get_model().encode(query.query, convert_to_tensor=True).tolist()

    # Optional metadata filter
    pinecone_filter = None
    if query.categories:
        if len(query.categories) == 1:
            pinecone_filter = {"category": {"$eq": query.categories[0]}}
        else:
            pinecone_filter = {"category": {"$in": query.categories}}

    # Query Pinecone index
    pc_response = get_index().query(
        vector=query_embedding,
        top_k=query.top_k,
        include_metadata=True,
        filter=pinecone_filter,
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    # Index stats
    try:
        stats = get_index().describe_index_stats()
        total_vectors = stats.get("total_vector_count") or 0
    except Exception:
        total_vectors = 0

    results = [
        {
            "title": match["metadata"].get("title", "Untitled"),
            "url": match["id"],
            "summary": match["metadata"].get("summary", ""),
            "category": match["metadata"].get("category", "unknown"),
            "published": match["metadata"].get("published", ""),
            "score": match.get("score"),
        }
        for match in pc_response.get("matches", [])
    ]

    return SearchResponse(
        results=[
            SearchResult(
                title=item["title"],
                url=item["url"],
                summary=item["summary"],
                category=item["category"],
                published=item["published"],
                score=item.get("score"),
            )
            for item in results
        ],
        metrics=SearchMetrics(
            elapsed_ms=elapsed_ms,
            top_k=query.top_k,
            total_vectors=total_vectors,
            filtered=bool(pinecone_filter is not None),
        ),
    )
