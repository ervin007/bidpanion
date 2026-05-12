import warnings
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, VERTEX_PROJECT_ID, VERTEX_LOCATION
from vectorstore.bm25_index import bm25_search
from vectorstore.fusion import rrf_merge

# Suppress annoying deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_embeddings():
    return VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION
    )

def build_vectorstore(documents: list[Document], collection_name: str = None) -> Chroma:
    collection = collection_name or CHROMA_COLLECTION
    embeddings = get_embeddings()

    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=collection,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Vertex AI embedding limit is 20000 tokens total per request
    batch_size = 30
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
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
) -> list:
    search_kwargs = {"k": dense_k, "fetch_k": dense_k * 3, "lambda_mult": 0.6}

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs=search_kwargs
    )
    dense_results = retriever.invoke(query)
    sparse_results = bm25_search(bm25_index, bm25_docs, query, k=sparse_k)

    return rrf_merge(dense_results, sparse_results, top_n=final_k)
