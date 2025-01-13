import os
import pickle
import feedparser
from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec


def initialize_pinecone(api_key, environment, index_name, embedding_dim):
    """Initialize Pinecone and create an index if it doesn't exist."""
    pc = pinecone.Pinecone(api_key=api_key, environment=environment)
    try:
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created.")
    except pinecone.core.openapi.shared.exceptions.PineconeApiException as e:
        if e.status == 409:  
            print(f"Index creation conflict: {e.body}")
        else:
            raise
    return pc.Index(index_name)

def fetch_articles(feed_url):
    """Fetch articles from an RSS feed."""
    articles = []
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary,
            "content": entry.get("content", [{"value": ""}])[0]["value"]
        })
    return articles

def load_cached_urls(log_file):
    """Load cached URLs from a pickle file."""
    if os.path.exists(log_file):
        with open(log_file, "rb") as f:
            return pickle.load(f)
    return set()

def save_cached_urls(cached_urls, log_file):
    """Save cached URLs to a pickle file."""
    with open(log_file, "wb") as f:
        pickle.dump(cached_urls, f)

def process_feeds_with_cache(feeds, model, index, log_file):
    """Process RSS feeds and upsert articles to Pinecone, skipping cached URLs."""
    if index is None:
        print("Error: Pinecone index is not initialized.")
        return

    cached_urls = load_cached_urls(log_file)

    all_articles = []
    for category, url in feeds.items():
        articles = fetch_articles(url)
        for article in articles:
            article["category"] = category
            # Skip if the URL is already cached
            if article["link"] in cached_urls:
                print(f"Skipping cached URL: {article['link']}")
                continue
            # Generate embeddings
            if article.get("title") and article.get("summary"):
                article["embedding"] = model.encode(
                    article["title"] + " " + article["summary"],
                    convert_to_tensor=True
                ).tolist()
                # Add to Pinecone
                index.upsert([(
                    article["link"],
                    article["embedding"],
                    {
                        "title": article["title"],
                        "summary": article["summary"],
                        "category": article["category"],
                        "published": article["published"]
                    }
                )])
                # Cache the URL
                cached_urls.add(article["link"])
                print(f"Upserted and cached URL: {article['link']}")
            else:
                print(f"Skipping article with missing title or summary: {article}")

    # Save updated cache
    save_cached_urls(cached_urls, log_file)
    print("Processing completed.")

# Main function
def main():
    load_dotenv()

    # Configurations
    api_key = os.getenv("PINECONE_KEY")
    environment = "us-west1-gcp"
    index_name = "gossip-semantic-search"
    embedding_dim = 384
    log_file = "cached_urls.pkl"

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
        "public_royalty": "https://www.public.fr/people/familles-royales/feed"
    }

    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process feeds with caching
    process_feeds_with_cache(feeds, model, index, log_file)

if __name__ == "__main__":
    main()
