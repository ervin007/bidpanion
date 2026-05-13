import os
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from config import VERTEX_PROJECT_ID, VERTEX_LOCATION, EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from vectorstore.bm25_index import bm25_search
from vectorstore.fusion import rrf_merge

def get_embeddings():
    return VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION
    )

def build_vectorstore(documents, collection_name=None):
    embeddings = get_embeddings()
    collection = collection_name or CHROMA_COLLECTION
    
    # Isolation: clear existing collection if it exists to prevent mixing tenders
    try:
        existing_vstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=collection,
            embedding_function=embeddings
        )
        existing_vstore.delete_collection()
        print(f"  Cleared existing collection: {collection}")
    except Exception:
        pass

    # Vertex AI has a strict limit of 250 instances per prediction request,
    # but ALSO a token limit (e.g. 20,000 tokens for some models/regions).
    # Since our chunks can be large (~1,900 tokens), we use a small batch size.
    batch_size = 5
    
    # Initialize the vectorstore with the first batch to set up the collection
    first_batch = documents[:batch_size]
    vectorstore = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=collection
    )
    
    # Add the rest of the documents in batches
    if len(documents) > batch_size:
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vectorstore.add_documents(batch)
            print(f"  Indexed {min(i + batch_size, len(documents))}/{len(documents)} dense chunks...")
        
    return vectorstore

def hybrid_retrieve(
    vectorstore,
    bm25_index,
    bm25_docs: list,
    query: str,
    dense_k: int = 60,
    sparse_k: int = 40,
    final_k: int = 100,
    metadata_filter: dict = None,
) -> list:
    """
    Retrieve using both dense MMR and BM25, fuse with RRF.
    """
    search_kwargs = {"k": dense_k, "fetch_k": dense_k * 3, "lambda_mult": 0.6}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs=search_kwargs
    )
    dense_results = retriever.invoke(query)
    sparse_results = bm25_search(bm25_index, bm25_docs, query, k=sparse_k)

    return rrf_merge(dense_results, sparse_results, top_n=final_k)
