"""Runs before any test module is imported: every folder the app writes to is a temporary one, so that the tests can
never touch the real indexes, graphs, traces or uploaded documents."""
import os
import sys
import tempfile

os.environ["FAKE_EMBED"] = "1"  # no embedding model download in tests
for var in ("CHROMA_PATH", "BM25_DIR", "GRAPH_DIR", "TRACE_DIR", "USER_CORPORA_DIR"):
    os.environ[var] = tempfile.mkdtemp()
for var in ("LLM_URL", "LLM_API_KEY", "LLM_MODEL", "GEMINI_MODEL", "GOOGLE_API_KEY", "GOOGLE_API_KEYS", "PII_NAMES"):
    os.environ.pop(var, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
