import pytest
from load_articles import initialize_pinecone, fetch_articles, process_feeds_with_cache
from sentence_transformers import SentenceTransformer
import pinecone
import os
from fastapi.testclient import TestClient
from backend import app
from pinecone import ServerlessSpec
import pickle

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configurations
API_KEY = os.getenv("PINECONE_KEY")
ENVIRONMENT = "us-west1-gcp"
INDEX_NAME = "gossip-semantic-search"
EMBEDDING_DIM = 384
LOG_FILE = "cached_urls_test.pkl"
TEST_FEED = "https://vsd.fr/actu-people/feed/"
TEST_QUERY = "celebrity news"

# Initialize TestClient for FastAPI
client = TestClient(app)


@pytest.fixture(scope="module")
def pinecone_index():
    """Fixture to initialize Pinecone and create the test index."""
    pc = pinecone.Pinecone(api_key=API_KEY, environment=ENVIRONMENT)
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    except pinecone.core.openapi.shared.exceptions.PineconeApiException as e:
        if e.status == 409:  
            print(f"Index creation conflict: {e.body}")
        else:
            raise
    return pc.Index(INDEX_NAME)


@pytest.fixture(scope="module")
def test_model():
    """Fixture for the SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def test_fetch_articles():
    """Test the fetch_articles function."""
    articles = fetch_articles(TEST_FEED)
    assert isinstance(articles, list), "fetch_articles did not return a list."
    if articles:
        assert "title" in articles[0], "Article does not contain 'title'."
        assert "link" in articles[0], "Article does not contain 'link'."
        assert "summary" in articles[0], "Article does not contain 'summary'."


def test_process_feeds_with_cache(pinecone_index, test_model):
    """Test process_feeds_with_cache with a small feed."""
    feeds = {"test_feed": TEST_FEED}
    process_feeds_with_cache(feeds, test_model, pinecone_index, LOG_FILE)
    # Verify that articles from the feed are added
    assert os.path.exists(LOG_FILE), "Cache file was not created."
    with open(LOG_FILE, "rb") as f:
        cached_urls = pickle.load(f)
    assert len(cached_urls) > 0, "No URLs were cached."


def test_search_endpoint(pinecone_index, test_model):
    """Test the /search endpoint."""
    query = {"query": TEST_QUERY}

    # Generate embeddings for testing
    embedding = test_model.encode(TEST_QUERY, convert_to_tensor=True).tolist()
    pinecone_index.upsert([
        (
            "test-url",
            embedding,
            {
                "title": "Test Title",
                "summary": "Test Summary",
                "category": "Test Category",
                "published": "2025-01-01"
            }
        )
    ])

    # Test the endpoint
    response = client.post("/search", json=query)
    assert response.status_code == 200, "/search did not return a 200 status code."
    results = response.json()
    assert isinstance(results, list), "/search did not return a list."
    if results:
        assert "title" in results[0], "Result does not contain 'title'."
        assert "url" in results[0], "Result does not contain 'url'."
        assert "summary" in results[0], "Result does not contain 'summary'."
        assert "category" in results[0], "Result does not contain 'category'."
        assert "published" in results[0], "Result does not contain 'published'."


def test_invalid_query():
    """Test the /search endpoint with an invalid query."""
    response = client.post("/search", json={})
    assert response.status_code == 422, "/search should return 422 for invalid query."
