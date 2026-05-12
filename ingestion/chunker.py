import re
from langchain_core.documents import Document
from ingestion.table_reconstructor import reconstruct_tables

def load_and_chunk(txt_path: str) -> list[Document]:
    """
    Load a text file and chunk it by pages (5 pages per chunk, 1 page overlap).
    Expects '=== Page X ===' markers in the text.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Reconstruct tables first
    raw_text = reconstruct_tables(raw_text)

    # Split the text by page markers
    # Pattern looks for '=== Page \d+ ==='
    page_splits = re.split(r'=== Page \d+ ===', raw_text)
    
    # The first element might be header info before the first page marker
    header_info = page_splits[0]
    pages = page_splits[1:] # Actual page contents

    # If no page markers found, fall back to character splitting
    if not pages:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )
        raw_chunks = splitter.split_text(raw_text)
        return [Document(page_content=c, metadata={"chunk_index": i}) for i, c in enumerate(raw_chunks)]

    documents = []
    chunk_size_pages = 5
    overlap_pages = 1
    
    # Sliding window over pages
    i = 0
    chunk_idx = 0
    while i < len(pages):
        # Take up to 5 pages
        end_idx = min(i + chunk_size_pages, len(pages))
        chunk_pages = pages[i:end_idx]
        
        # Merge pages into a single chunk
        chunk_content = ""
        page_range = []
        for j, page_text in enumerate(chunk_pages):
            page_num = i + j + 1
            page_range.append(page_num)
            chunk_content += f"\n=== Page {page_num} ===\n{page_text}"
        
        documents.append(Document(
            page_content=chunk_content.strip(),
            metadata={
                "chunk_index": chunk_idx,
                "pages": page_range,
                "source": txt_path
            }
        ))
        
        chunk_idx += 1
        # Advance by (size - overlap)
        i += (chunk_size_pages - overlap_pages)
        
        # Break if we've reached the end and can't make a full new step
        if i >= len(pages):
            break
            
    return documents
