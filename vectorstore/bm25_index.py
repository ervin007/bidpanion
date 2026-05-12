import re
import pickle
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def tokenize_german(text: str) -> list[str]:
    """Lowercase tokenisation preserving German umlauts."""
    text = text.lower()
    return re.findall(r"[a-zäöüß0-9]+", text)

def build_bm25(documents: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """Build BM25 index aligned with document list."""
    tokenized = [tokenize_german(d.page_content) for d in documents]
    bm25 = BM25Okapi(tokenized)
    return bm25, documents

def bm25_search(
    bm25: BM25Okapi,
    docs: list[Document],
    query: str,
    k: int = 30,
) -> list[tuple[Document, float]]:
    """Return top-k (doc, score) pairs for a query."""
    tokens = tokenize_german(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], float(scores[i])) for i in top_indices]

def save_bm25(bm25: BM25Okapi, path: str):
    with open(path, "wb") as f:
        pickle.dump(bm25, f)

def load_bm25(path: str) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)
