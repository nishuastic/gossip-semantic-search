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
- Python 3.8 or higher
- Pinecone account and API key

### Clone the Repository
```bash
git clone https://github.com/nishuastic/gossip-semantic-search
cd gossip-semantic-search
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root directory and add the following:
```env
PINECONE_KEY=<your-pinecone-api-key>
```

## Running the Application

### Step 1: Fetch and Process Articles
Run the `load_articles.py` script to fetch articles, generate embeddings, and upload them to Pinecone.
```bash
python load_articles.py
```

### Step 2: Start the Backend Server
Start the FastAPI backend:
```bash
uvicorn backend:app --reload
```
By default, the backend will be available at `http://localhost:8000`.

### Step 3: Start the Frontend
Run the Streamlit app:
```bash
streamlit run frontend.py
```
The frontend will be available at `http://localhost:8501`.

## Usage
1. Open the frontend in your browser (`http://localhost:8501`).
2. Enter a search query into the input field and click **Search**.
3. View the search results, which include the title, summary, category, and publication date of matching articles.

## Running Tests
To ensure everything works as expected, run the tests:
```bash
pytest tests.py
```

## Project Structure
```
├── load_articles.py  # Script for fetching and processing articles
├── backend.py        # FastAPI backend for semantic search
├── frontend.py       # Streamlit frontend for user interaction
├── tests.py          # Test suite
├── requirements.txt  # Python dependencies
├── .env              # Environment variables
└── README.md         # Project documentation
```

## Acknowledgements
- [Pinecone](https://www.pinecone.io/) for vector database services.
- [Hugging Face](https://huggingface.co/) for pre-trained Sentence Transformers.

---
