import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# Ensure project root is importable so `import src.*` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def mock_heavy_imports(monkeypatch):
    """Mock heavy imports automatically for all tests"""

    # Mock pinecone module
    mock_pinecone = MagicMock()
    mock_pinecone_class = MagicMock()
    mock_pinecone_class.Index.return_value = MagicMock()
    mock_pinecone.Pinecone = mock_pinecone_class
    mock_pinecone.ServerlessSpec = MagicMock()
    monkeypatch.setitem(sys.modules, "pinecone", mock_pinecone)

    # Mock sentence_transformers module
    mock_st = MagicMock()
    mock_encoder = MagicMock()

    # Configure the encode method to return a mock object with tolist()
    class MockVector:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    mock_encoder.encode.return_value = MockVector()
    mock_st.SentenceTransformer.return_value = mock_encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)

    # Mock data_models module
    mock_data_models = MagicMock()

    class StubArticle:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    mock_data_models.Article = StubArticle
    monkeypatch.setitem(sys.modules, "data_models", mock_data_models)

    # Mock feedparser
    mock_feedparser = MagicMock()
    monkeypatch.setitem(sys.modules, "feedparser", mock_feedparser)
