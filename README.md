 # Gossip Semantic Search

Gossip Semantic Search is a project designed to enable semantic search functionality for articles from the VSD and Public RSS feeds. This application leverages Sentence Transformers and Pinecone to provide efficient and accurate search results.

## Features
- Fetches articles from VSD and Public RSS feeds.
- Processes articles using Sentence Transformers to generate embeddings.
- Stores article embeddings in Pinecone for semantic search.
- Provides a FastAPI backend for search functionality.
- Includes a Streamlit-based frontend for user interaction.

## Installation
### Prerequisites
- Python 3.10 or higher
- Pinecone account and API key

### Clone the Repository
```bash
git clone https://github.com/nishuastic/gossip-semantic-search
cd gossip-semantic-search
```

### Install Dependencies (uv recommended)
```bash
uv sync
```

### Environment Variables
Create a `.env` file in the project root directory and add the following:
```env
PINECONE_KEY=<your-pinecone-api-key>
```

## Running the Application

### Step 1: Fetch and Process Articles
Run the ingestion to fetch articles, embed, and upsert to Pinecone.
```bash
uv run python -m src.load_articles
```

### Step 2: Start the Backend Server
Start the FastAPI backend:
```bash
uv run uvicorn src.backend:app --reload --port 8000
```
By default, the backend will be available at `http://localhost:8000`.

### Step 3: Start the Frontend
Run the Streamlit app:
```bash
uv run streamlit run src/frontend.py
```
The frontend will be available at `http://localhost:8501`.

## Usage
1. Open the frontend in your browser (`http://localhost:8501`).
2. Enter a search query into the input field and click **Search**.
3. View the search results, which include the title, summary, category, and publication date of matching articles.

## Running Tests
Run mocked tests with coverage enforced at 100% for `src` (UI omitted):
```bash
uv run pytest
```

## Project Structure
```
├── Dockerfile
├── start.sh
├── pyproject.toml
├── src/
│   ├── backend.py        # FastAPI backend for semantic search
│   ├── frontend.py       # Streamlit frontend for user interaction
│   ├── load_articles.py  # Ingestion script
│   └── main.py
├── tests/
│   ├── conftest.py
│   ├── test_backend.py
│   └── test_load_articles.py
└── README.md
```

The app persists history and caches under `DATA_DIR` (default `data/`). In Docker we map it to a volume.

Build image:
```bash
docker build -t gossip-app .
```

Run with a named volume (persists across restarts):
```bash
docker run -p 8000:8000 -p 8501:8501 \
  -e PINECONE_KEY="YOUR_KEY" \
  -e DATA_DIR=/data \
  -v gossip_data:/data \
  gossip-app
```

Open UI: http://localhost:8501

Then start:
```bash
PINECONE_KEY=YOUR_KEY docker compose up --build
```

## Acknowledgements
- [Pinecone](https://www.pinecone.io/) for vector database services.
- [Hugging Face](https://huggingface.co/) for pre-trained Sentence Transformers.

---
