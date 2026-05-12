from langchain_core.documents import Document

def rrf_merge(
    dense_results: list[Document],
    sparse_results: list[tuple[Document, float]],
    k: int = 60,
    top_n: int = 100,
) -> list[Document]:
    """
    Reciprocal Rank Fusion of dense and sparse ranked lists.
    """
    scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for rank, doc in enumerate(dense_results):
        cid = doc.metadata["chunk_index"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    for rank, (doc, _) in enumerate(sparse_results):
        cid = doc.metadata["chunk_index"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]
    return [doc_map[cid] for cid in sorted_ids if cid in doc_map]
