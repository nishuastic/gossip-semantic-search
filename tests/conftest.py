import sys
from pathlib import Path
import types

# Ensure project root is importable so `import src.*` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a fake pinecone module to avoid import-time exceptions during tests
if "pinecone" not in sys.modules:
    pinecone_stub = types.ModuleType("pinecone")

    class _StubPinecone:
        def __init__(self, *args, **kwargs):
            pass

        def Index(self, *args, **kwargs):
            return object()

    class _StubServerlessSpec:
        def __init__(self, *args, **kwargs):
            pass

    pinecone_stub.Pinecone = _StubPinecone
    pinecone_stub.ServerlessSpec = _StubServerlessSpec
    sys.modules["pinecone"] = pinecone_stub

# Provide a fake sentence_transformers module to avoid heavy imports
if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")

    class _StubSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            class _Vec:
                def tolist(self_inner):
                    return [0.1, 0.2, 0.3]

            return _Vec()

    st_stub.SentenceTransformer = _StubSentenceTransformer
    sys.modules["sentence_transformers"] = st_stub
