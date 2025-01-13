from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import pinecone
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_KEY"), environment="us-west1-gcp")
index = pc.Index("gossip-semantic-search")

class Query(BaseModel):
    query: str

@app.post("/search")
async def search(query: Query):
    # Assuming your embeddings are generated using a sentence-transformer model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate query embedding
    query_embedding = model.encode(query.query, convert_to_tensor=True).tolist()

    # Query Pinecone index
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Return results
    return [
        {
            "title": match["metadata"]["title"],
            "url": match["id"],
            "summary": match["metadata"]["summary"],
            "category": match["metadata"]["category"],
            "published": match["metadata"]["published"],
        }
        for match in results["matches"]
    ]

