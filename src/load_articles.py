import os
import pickle

import feedparser
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from .data_models import Article

load_dotenv()


def initialize_pinecone(api_key, environment, index_name, embedding_dim):
    """Create Pinecone index if missing and return an Index handle."""
    pc = Pinecone(api_key=api_key)

    # Create index only if it doesn't already exist
    try:
        existing_indexes = pc.list_indexes().names()
    except Exception:
        existing_indexes = []

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

    return pc.Index(index_name)


def fetch_articles(feed_url):
    """Fetch and parse entries from the given RSS/Atom feed URL."""
    results = []
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        results.append(
            Article(
                title=entry.title,
                link=entry.link,
                published=entry.published,
                summary=entry.summary,
                content=entry.get("content", [{"value": ""}])[0]["value"],
            )
        )
    return results


def load_cached_urls(log_file):
    """Load a set of cached URLs from pickle at `log_file`. One-liner for small attr."""
    path = Path(log_file)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return set()


def save_cached_urls(cached_urls, log_file):
    """Persist the set of cached URLs to pickle at `log_file`. One-liner for small attr."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cached_urls, f)


def process_feeds_with_cache(feeds, model, index, log_file):
    """Iterate feeds, embed new articles, upsert to Pinecone, and update cache."""
    if index is None:
        print("Error: Pinecone index is not initialized.")
        return

    cached_urls = load_cached_urls(log_file)

    for category, url in feeds.items():
        articles = fetch_articles(url)
        for article in articles:
            article.category = category
            # Skip if the URL is already cached
            if article.link in cached_urls:
                print(f"Skipping cached URL: {article.link}")
                continue
            # Generate embeddings
            if article.title and article.summary:
                embedding = model.encode(
                    article.title + " " + article.summary, convert_to_tensor=True
                ).tolist()
                # Add to Pinecone
                index.upsert(
                    [
                        (
                            article.link,
                            embedding,
                            {
                                "title": article.title,
                                "summary": article.summary,
                                "category": article.category,
                                "published": article.published,
                            },
                        )
                    ]
                )
                # Cache the URL
                cached_urls.add(article.link)
                print(f"Upserted and cached URL: {article.link}")
            else:
                print(f"Skipping article with missing title or summary: {article}")

    # Save updated cache
    save_cached_urls(cached_urls, log_file)
    print("Processing completed.")


# Main function
def main():
    """Entry point for ingestion: ensure index, fetch feeds, and upsert."""
    # Configurations
    api_key = os.getenv("PINECONE_KEY")
    environment = None  # Not used with current Pinecone client
    index_name = "gossip-semantic-search"
    embedding_dim = 384
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(data_dir / "cached_urls.pkl")

    # Initialize Pinecone
    index = initialize_pinecone(api_key, environment, index_name, embedding_dim)

    # Define feeds
    feeds = {
        "vsd_people": "https://vsd.fr/actu-people/feed/",
        "vsd_tv": "https://vsd.fr/tele/feed/",
        "vsd_company": "https://vsd.fr/societe/feed/",
        "vsd_culture": "https://vsd.fr/culture/feed/",
        "vsd_leisure": "https://vsd.fr/loisirs/feed/",
        "public_news": "https://www.public.fr/feed",
        "public_people": "https://www.public.fr/people/feed",
        "public_tv": "https://www.public.fr/tele/feed",
        "public_fashion": "https://www.public.fr/mode/feed",
        "public_royalty": "https://www.public.fr/people/familles-royales/feed",
    }

    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Process feeds with caching
    process_feeds_with_cache(feeds, model, index, log_file)


if __name__ == "__main__":
    main()
